import pandas as pd
from transformers import DistilBertTokenizerFast, BertTokenizerFast, RobertaTokenizerFast


def tokenizer_text(path_dataset, model_text):
    if model_text == "distilbert":
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    elif model_text == "bert":
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    elif model_text == "roberta":
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

    df = pd.read_csv(path_dataset)

    sentences = df['Output'].tolist()

    tokens = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    return tokens