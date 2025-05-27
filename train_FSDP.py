# train_FSDP.py

import os
import argparse
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

def preprocess(example, tokenizer, max_length=512):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
    text = prompt + example["output"]
    tok = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    # labels = input_ids (model sẽ ignore token_id = -100 nếu cần)
    tok["labels"] = tok["input_ids"].copy()
    return tok

def collate_fn(batch):
    # batch là list of dicts có keys: input_ids, attention_mask, labels
    return {
        k: torch.tensor([example[k] for example in batch], dtype=torch.long)
        for k in batch[0]
    }

def main():
    # 1. Parse args & init process group
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", 0)),
        help="Torch local rank"
    )
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl")

    # 2. Load tokenizer & model
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True
    ).to(torch.cuda.current_device())

    # 3. Wrap with FSDP
    model = FSDP(model)

    # 4. Load & preprocess dataset
    raw_ds = load_dataset("yahma/alpaca-cleaned", split="train")
    # map với tokenization + padding
    ds = raw_ds.map(
        lambda ex: preprocess(ex, tokenizer),
        batched=False,
        remove_columns=raw_ds.column_names,
    )

    # 5. DataLoader với custom collate_fn
    train_loader = DataLoader(
        ds,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # 6. Optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # 7. Training loop
    model.train()
    for epoch in range(3):
        for batch in train_loader:
            # Move batch to current GPU
            batch = {k: v.cuda() for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if dist.get_rank() == 0:
                print(f"[Epoch {epoch}] loss = {loss.item():.4f}")

    # 8. Save checkpoint (only rank 0)
    if dist.get_rank() == 0:
        model.module.save_pretrained("qwen3_fsdp_sft")

if __name__ == "__main__":
    main()
