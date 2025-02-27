import pandas as pd
from transformers import DistilBertTokenizerFast, BertTokenizerFast


def tokenizer_text(path_dataset):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    df = pd.read_csv(path_dataset)

    sentences = df['Output'].tolist()

    tokens = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    return tokens