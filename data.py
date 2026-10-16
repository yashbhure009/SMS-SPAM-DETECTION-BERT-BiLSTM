import torch
from torch.utils.data import Dataset


class SMSDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._printed_sample_shape = False

        # Tokenize everything once (faster than tokenizing per item every epoch).
        # encode_plus was removed in transformers v5; call the tokenizer directly.
        encodings = tokenizer(
            self.data["text"].astype(str).tolist(),
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels = torch.tensor(self.data["label"].astype(int).values, dtype=torch.float)

        print(
            f"[SMSDataset] Created dataset with {len(self.data)} samples "
            f"(max_length={self.max_length})"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if not self._printed_sample_shape:
            print(
                "[SMSDataset] First encoded sample: "
                f"input_ids={tuple(self.input_ids[index].shape)}, "
                f"attention_mask={tuple(self.attention_mask[index].shape)}, "
                f"label={self.labels[index].item()}"
            )
            self._printed_sample_shape = True

        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "label": self.labels[index],
        }