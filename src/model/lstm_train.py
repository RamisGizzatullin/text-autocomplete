import torch.nn as nn
from tqdm import tqdm

class LSTMTrain():
    def __init__(self, model, loader, device):
        self.model = model
        self.loader = loader
        self.device = device

    def train(self, optimizer, criterion, tokenizer):
        self.model.train()

        total_loss = 0
        
        for batch in tqdm(self.loader):
            ids = batch['texts'].to(self.device)
            mask = batch['masks'].to(self.device)
            labels = batch['labels'].to(self.device)

            optimizer.zero_grad()
            logits = self.model(ids, mask)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.loader)