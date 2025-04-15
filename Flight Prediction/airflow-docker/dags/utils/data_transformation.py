from sklearn.preprocessing import StandardScaler
import pandas as pd

class DataTransformer:
    def __init__(self, data):
        self.data = data

    def transform(self):
        data = self.data.copy()
        X = data.drop(columns=["price"])
        y = data["price"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_df = pd.DataFrame(X_scaled, columns=X.columns)
        return X_df, y
