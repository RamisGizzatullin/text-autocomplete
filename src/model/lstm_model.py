import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=128):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)
        self.dropout = nn.Dropout(0.5)
        self.init_weights()


    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, inputs, masks):
        x = self.embedding(inputs)
        rnn_out, _ = self.rnn(x)

        mask = masks.unsqueeze(2).expand_as(rnn_out)
        masked_out = rnn_out * mask
        summed = masked_out.sum(dim=1)
        lengths = masks.sum(dim=1).unsqueeze(1)
        mean_pooled = summed / lengths
        out = self.dropout(mean_pooled)

        return self.fc(out)


    
    # def generate(self, text, tokenizer, max_length):

    #     token_ids = tokenizer.encode(text, add_special_tokens=False, max_length=max_length, truncation=True)
    #     mask = [1 if t != 0 else 0 for t in token_ids]

    #     with torch.no_grad():
    #         emb = self.embedding(torch.tensor(token_ids, dtype=torch.long))
    #         rnn_out, _ = self.rnn(emb)

    #     tokenizer.convert_ids_to_tokens(
    #     print(self.fc(rnn_out)) 

   