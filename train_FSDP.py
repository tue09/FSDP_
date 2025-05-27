# train_FSDP.py

import os
import argparse
import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# PyTorch FSDP imports
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    ShardingStrategy,
    BackwardPrefetch,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# 1) Data prep
def preprocess(example, tokenizer, max_length=512):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
    text = prompt + example["output"]
    tok = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    tok["labels"] = tok["input_ids"].copy()
    return tok

def collate_fn(batch):
    return {
        k: torch.tensor([ex[k] for ex in batch], dtype=torch.long)
        for k in batch[0]
    }

def main():
    # ——————————————
    # 0) Distributed init
    # ——————————————
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl")

    # ——————————————
    # 1) Tokenizer + Model (load trên CPU)
    # ——————————————
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,               # chừa RAM
    )
    model.gradient_checkpointing_enable()    # tiết kiệm activation

    # ——————————————
    # 2) Wrap FSDP
    # ——————————————
    fsdp = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,      # shard param + grads + opt-state
        auto_wrap_policy=transformer_auto_wrap_policy,      # chỉ wrap các block Transformer lớn
        cpu_offload=CPUOffload(offload_params=True),        # offload params về CPU khi không dùng
        mixed_precision=MixedPrecision(                     # bfloat16 nội bộ
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,    # prefetch gradients
        device_id=torch.cuda.current_device(),              # mỗi rank giữ shard trên GPU riêng
    )
    model = fsdp

    # ——————————————
    # 3) Dataset + DataLoader
    # ——————————————
    raw = load_dataset("yahma/alpaca-cleaned", split="train")
    ds = raw.map(
        lambda ex: preprocess(ex, tokenizer),
        batched=False,
        remove_columns=raw.column_names,
    )

    train_loader = DataLoader(
        ds,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # ——————————————
    # 4) Optimizer & (optional) Scaler
    # ——————————————
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scaler = torch.cuda.amp.GradScaler()  # vẫn dùng AMP wrapper nếu muốn

    # ——————————————
    # 5) Training loop
    # ——————————————
    model.train()
    for epoch in range(3):
        for batch in train_loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.cuda.amp.autocast():  # AMP context
                outputs = model(**batch)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if dist.get_rank() == 0:
                print(f"[Epoch {epoch}] loss={loss.item():.4f}")

    # ——————————————
    # 6) Save checkpoint (rank0)
    # ——————————————
    if dist.get_rank() == 0:
        # Move full model to CPU để save
        model.cpu()
        # unwrap FSDP và lưu
        model.module.save_pretrained("qwen3_fsdp_sft")

if __name__ == "__main__":
    main()
