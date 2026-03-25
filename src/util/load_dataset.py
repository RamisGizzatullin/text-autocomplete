import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class LoadDataset():
    def __init__(self, tokenizer, max_len=10, batch_size=5):
        self.max_len = max_len
        self.batch_size = batch_size
        self.tokenizer = tokenizer

    def getText(self, path):
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')

        texts = pd.read_csv(os.path.join(data_dir, path), index_col=False)

        return [line[0] for line in texts.values.tolist()]

    def getDataLoader(self, path, shuffle=True):
        dataset = GetTargedDataset(self.getText(path), self.tokenizer, self.max_len)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
    
    def getTokenText(self, text):
        token_ids = self.tokenizer.encode(text, add_special_tokens=False, max_length=512, truncation=True)
        mask = [1 if t != 0 else 0 for t in token_ids]

        return torch.tensor([token_ids], dtype=torch.long), torch.tensor([mask], dtype=torch.long)

class GetTargedDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len = 4):
        self.samples = []
        self.seq_len = seq_len

        for line in texts:
            token_ids = tokenizer.encode(line, add_special_tokens=False, max_length=512, truncation=True)

            for i in range(1, len(token_ids) - 1):
                context = token_ids[max(0, len(token_ids)-i-self.seq_len):-i]
                context.extend([0] * (self.seq_len - len(context)))

                target = token_ids[len(token_ids)-i]

                self.samples.append((context, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]

        mask = [1 if t != 0 else 0 for t in x]

        return {
            'texts': torch.tensor(x, dtype=torch.long),
            'masks': torch.tensor(mask, dtype=torch.long),
            'labels': torch.tensor(y, dtype=torch.long)
        }

