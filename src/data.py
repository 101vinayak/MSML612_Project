from typing import Tuple

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding


MODEL_NAME = "distilbert-base-uncased"


def get_tokenizer(model_name: str = MODEL_NAME):
    return AutoTokenizer.from_pretrained(model_name)


def load_sst2(tokenizer, max_length: int = 128):
    dataset = load_dataset("glue", "sst2")

    def tokenize_fn(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=max_length)

    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.remove_columns([c for c in tokenized["train"].column_names if c not in ["input_ids", "attention_mask", "label"]])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(type="torch")
    return tokenized


def build_dataloaders(tokenizer, max_length: int = 128, train_batch_size: int = 16, eval_batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset = load_sst2(tokenizer=tokenizer, max_length=max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader = DataLoader(dataset["train"], batch_size=train_batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(dataset["validation"], batch_size=eval_batch_size, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(dataset["validation"], batch_size=eval_batch_size, shuffle=False, collate_fn=collator)
    return train_loader, val_loader, test_loader
