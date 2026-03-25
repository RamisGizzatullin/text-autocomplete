import evaluate
import torch
from tqdm import tqdm
from src.model.lstm_predict import LSTMPredict

class LSTMRouge():
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.device = device
        self.tokenizer = tokenizer

    def get_metrics(self, texts):
        generated_summaries = []
        lstm_predict = LSTMPredict(self.model, self.tokenizer, self.device)

        with torch.no_grad():
            for text in texts:
                text = text.split(" ")[:-1]
                if len(text) == 0: 
                     continue
                
                summary = lstm_predict.generate(" ".join(text))
                generated_summaries.append(summary)

        rouge = evaluate.load("rouge")

        results = rouge.compute(predictions=generated_summaries, references=texts)

        return results   