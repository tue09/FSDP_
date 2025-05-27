# train_fsdp_sft.py

import os
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    AdamW,
)

def main():
    # 1. Parse local rank để phân GPU
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", 0)),
        help="Torch local rank"
    )
    args = parser.parse_args()

    # 2. Khởi tạo process group và chỉ định GPU cho mỗi rank
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl")

    # 3. Load tokenizer và model
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True
    ).to(torch.cuda.current_device())

    # 4. Quấn model với FSDP
    model = FSDP(model)

    # 5. Load và tiền xử lý dataset
    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    def preprocess(example):
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
        text = prompt + example["output"]
        tok = tokenizer(
            text, truncation=True, max_length=512
        )
        tok["labels"] = tok["input_ids"].copy()
        return tok

    ds = ds.map(preprocess, remove_columns=ds.column_names)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    train_loader = DataLoader(
        ds, batch_size=2, shuffle=True, collate_fn=collator
    )

    # 6. Khởi tạo optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # 7. Training loop
    model.train()
    for epoch in range(3):
        for batch in train_loader:
            # Chuyển batch sang GPU tương ứng
            batch = {k: v.cuda() for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if dist.get_rank() == 0:
                print(f"[Epoch {epoch}] loss = {loss.item():.4f}")

    # 8. Lưu checkpoint chỉ ở rank 0
    if dist.get_rank() == 0:
        model.module.save_pretrained("qwen3_fsdp_sft")

if __name__ == "__main__":
    main()
