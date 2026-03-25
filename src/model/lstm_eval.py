import torch
from sklearn.metrics import accuracy_score

class LSTMEval():
    def __init__(self, model, loader, device):
        self.model = model
        self.loader = loader
        self.device = device

    def evaluate(self):
        self.model.eval()
        preds, trues = [], []

        with torch.no_grad():
            for batch in self.loader:
                ids = batch['texts'].to(self.device)
                mask = batch['masks'].to(self.device)
                labels = batch['labels'].to(self.device)
                logits = self.model(ids, mask)
                preds += torch.argmax(logits, dim=1).cpu().tolist()
                trues += labels.tolist()

        return accuracy_score(trues, preds)

