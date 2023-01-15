import pandas as pd
import spacy
import re
from unidecode import unidecode


def tokenizer(text, nlp=spacy.load('en_core_web_md')):
    # remove special character
    text = re.sub(r'[^\w\s]', '', text)

    # lowercase
    text = text.lower()

    # remove space
    text = text.strip()

    # remove accent
    text = unidecode(text)

    # lemmatize and remove stop_words
    tokens = nlp(text.lower().strip())
    tokens = [word.lemma_ for word in tokens if not word.is_stop and not word.is_punct and len(word) > 2]

    return tokens


def cleaner(text, nlp=spacy.load('en_core_web_md')):
    tokens = tokenizer(text, nlp)
    new_text = ""
    compteur = 0
    for word in tokens:
      new_text += word
      compteur += 1
      if compteur < len(tokens):
        new_text += " "
    return new_text


def preprocessor(path = "./Data/train.csv"):
    df = pd.read_csv(path)
    df['cleaned-text'] = df['text'].apply(lambda x: cleaner(x))
    X = df['cleaned-text']
    if 'target' in list(df.columns):
        y = df['target']
    else:
        y = None
    return X, y


if __name__ == '__main__':
    X, y = preprocessor()
