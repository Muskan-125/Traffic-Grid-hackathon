import pandas as pd

def load_data():

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample = pd.read_csv("data/sample_submission.csv")

    return train, test, sample