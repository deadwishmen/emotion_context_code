import pandas as pd
from trasformers import DistilBertTokenizerFast


def tokenizer_dataset(path_dataset):
    tokenizer = DistilBertTokenizerFast.from_pretrained('bert-base-uncased')
    df = pd.read_csv(path_dataset)

    sentences = df['text'].tolist()

    tokens = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    return tokens



