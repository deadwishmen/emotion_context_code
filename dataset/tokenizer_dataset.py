import pandas as pd
from transformers import DistilBertTokenizerFast


def tokenizer_text(path_dataset):
    tokenizer = DistilBertTokenizerFast.from_pretrained('bert-base-uncased')
    df = pd.read_csv(path_dataset)

    sentences = df['Output'].tolist()

    tokens = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    print(tokens)
    return tokens