import os
import re
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

class ProcessedDataset(): 
    def __init__(self, path, data_files, clean_path, train_path, val_path, test_path, split):
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')
        
        self.path = path
        self.data_files = os.path.join(data_dir, data_files)
        self.clean_path = os.path.join(data_dir, clean_path)
        self.train_path = os.path.join(data_dir, train_path)
        self.val_path = os.path.join(data_dir, val_path)
        self.test_path = os.path.join(data_dir, test_path)

        self.split = split

    def clean_text(self, row):
        text = row["text"]

        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        row["text"] = text
    
        return row

    def process(self):
        raw_dataset = load_dataset(self.path, data_files=self.data_files, split=self.split)

        clean_dataset = raw_dataset.map(self.clean_text)
        clean_dataset.to_csv(self.clean_path)

        texts = [line for line in clean_dataset["text"] if len(line.split()) > 2]

        train_texts, val_texts = train_test_split(texts, test_size=0.2, random_state=42)
        val_texts, test_texts = train_test_split(val_texts, test_size=0.5, random_state=42)

        train_df = pd.DataFrame(train_texts)
        train_df.to_csv(self.train_path, index=False)

        val_df = pd.DataFrame(val_texts)
        val_df.to_csv(self.val_path, index=False)

        test_df = pd.DataFrame(test_texts)
        test_df.to_csv(self.test_path, index=False)
