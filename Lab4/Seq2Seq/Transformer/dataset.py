from torch.utils.data import Dataset


class Seq2SeqDataset(Dataset):
    def __init__(self, data, tokenizer, max_source_len, max_target_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source = self.data.iloc[idx]["description"]
        target = self.data.iloc[idx]["diagnosis"]

        source_ids = self.tokenizer(
            source,
            max_length=self.max_source_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        with self.tokenizer.as_target_tokenizer():
            target_ids = self.tokenizer(
                target,
                max_length=self.max_target_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        return {
            "input_ids": source_ids["input_ids"].squeeze(),
            "attention_mask": source_ids["attention_mask"].squeeze(),
            "labels": target_ids["input_ids"].squeeze(),
        }
