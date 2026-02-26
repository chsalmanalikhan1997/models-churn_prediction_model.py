import pandas as pd
from sklearn.linear_model import LogisticRegression

def train_model(data):
    X = data[['balance', 'tenure', 'age']]
    y = data['churn']
    model = LogisticRegression()
    model.fit(X, y)
    return model
