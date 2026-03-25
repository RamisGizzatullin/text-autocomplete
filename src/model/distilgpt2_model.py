from transformers import pipeline

class Distilgpt2Model():
    def __init__(self, max_length):
        self.generator = pipeline("text-generation", model="distilgpt2")
        self.max_length = max_length

    def generate(self, text):
        
        result = self.generator(text, max_new_tokens=self.max_length, truncation=True, do_sample=True, pad_token_id=10, top_k=50)

        return result[0]["generated_text"]