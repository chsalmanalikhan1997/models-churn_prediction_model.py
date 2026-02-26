import pandas as pd
from sklearn.linear_model import LogisticRegression

def train_model(data):
    X = data[['balance', 'tenure', 'age']]
    y = data['churn']
    model = LogisticRegression()
    model.fit(X, y)
    return model
# Model version 2: Feature scaling added
# Final version for France application
