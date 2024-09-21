# src/trainRegression.py
import numpy as np
import pandas as pd 
import joblib

import string
import spacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS

import argparse

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Fix Seed
seed = 123
np.random.seed(seed)

punctuations = string.punctuation
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# Load English pre-trained model
nlp = spacy.load("en_core_web_sm")

def load_data(file_path):
    df = pd.read_csv(file_path, sep="@", encoding='latin-1', header=None)
    df.rename(columns={0: 'sentence', 1: 'target'}, inplace=True)
    mapper = {'positive':1, 'neutral':0, 'negative':-1}
    df['target'] = df['target'].apply(lambda x : mapper[x])
    return df


# Creating our tokenizer function
def spacy_processor(sentence):
    
    mytokens = nlp(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    return mytokens


def main(args):
    # Load data
    df = load_data(args.data_file)
    bow_vector = CountVectorizer(tokenizer = spacy_processor, ngram_range=(1,2))

    X_train, X_test, y_train, y_test = train_test_split(df.sentence.values, 
                                                    df.target.values, 
                                                    test_size = 0.2,
                                                    random_state = seed,
                                                    stratify = df.target.values)
    if args.model == 'linear_regression':
        regressor = LinearRegression()
    else:
        regressor = RandomForestRegressor()

    pipe = Pipeline([('vectorizer', bow_vector),
                    ('classifier', regressor)])


    pipe.fit(X_train, y_train)
    predicted = pipe.predict(X_test)

    # Model performance 
    print("Regression MAE:", mean_absolute_error(y_test, predicted))

    joblib.dump(pipe, args.saving_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a regression models.')
    parser.add_argument('--data_file', type=str, default='data/Sentences_75Agree.txt', help='Path to the data file.')
    parser.add_argument('--model', type=str, default='random_forest', help='Regression model.')
    parser.add_argument('--saving_path', type=str, default='models/pipe_random_forest_regressor.sav', help='Path to save the best model.')
    args = parser.parse_args()
    main(args)