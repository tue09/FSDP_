# train_fsdp_qwen.py
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import default_auto_wrap_policy
from torch.distributed.fsdp import BackwardPrefetch, CPUOffload, MixedPrecision
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from torch.cuda.amp import autocast, GradScaler

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=1024):
        self.encodings = tokenizer(texts, truncation=True, 
                                   padding="max_length", 
                                   max_length=max_length)
    def __len__(self):
        return len(self.encodings["input_ids"])
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) 
                for k, v in self.encodings.items()}
        # causal LM: labels = input_ids
        item["labels"] = item["input_ids"].clone()
        return item

def setup_process(rank, world_size):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def main():
    # 1. Khởi tạo multi-GPU
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    setup_process(rank, world_size)

    # 2. Tokenizer + Model (mixed-precision)
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B", trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        torch_dtype=torch.float16,
        device_map=None,  # để FSDP manage device
    )

    # 3. Wrap model với FSDP
    auto_wrap = default_auto_wrap_policy
    fsdp_cfg = {
        "mixed_precision": MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        ),
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        "cpu_offload": CPUOffload(offload_params=False),
        "auto_wrap_policy": auto_wrap,
    }
    model = FSDP(model, **fsdp_cfg)

    # 4. Chuẩn bị DataLoader
    #    Giả sử bạn đã có list `train_texts`
    train_texts = ["Your first training example...", "..."]  # thay bằng data thật
    dataset = TextDataset(train_texts, tokenizer, max_length=1024)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler)

    # 5. Optimizer + Scheduler + AMP
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(dataloader) * 3  # ví dụ 3 epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=total_steps
    )
    scaler = GradScaler()

    # 6. Training loop
    model.train()
    for epoch in range(3):
        sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader):
            # đưa batch lên GPU
            batch = {k: v.cuda() for k, v in batch.items()}
            with autocast():
                outputs = model(**batch)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            if rank == 0 and step % 20 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss {loss.item():.4f}")

    # 7. Lưu checkpoint (chỉ GPU 0)
    if rank == 0:
        model.module.save_pretrained("qwen3_sft_fsdp")
        tokenizer.save_pretrained("qwen3_sft_fsdp")

    cleanup()

if __name__ == "__main__":
    main()
