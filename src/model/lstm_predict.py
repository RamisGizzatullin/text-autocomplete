import torch
from src.util.load_dataset import LoadDataset

class LSTMPredict():
    def __init__(self, model, tokenizer, device, max_length = 1):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    def generate(self, text):
        self.model.eval()
        input_tokens = []

        load_dataset = LoadDataset(self.tokenizer)

        predic = text
        with torch.no_grad():
            for idx in range(self.max_length):
                token_ids, masks = load_dataset.getTokenText(predic)

                ids = token_ids.to(self.device)
                masks = masks.to(self.device)

                logits = self.model(ids, masks)
                preds = torch.argmax(logits, dim=1)

                input_tokens = self.tokenizer.convert_ids_to_tokens(preds.tolist())
                predic = predic + " " + input_tokens[0]

        return predic