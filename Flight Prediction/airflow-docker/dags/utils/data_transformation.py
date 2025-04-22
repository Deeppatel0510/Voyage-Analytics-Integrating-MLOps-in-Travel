from sklearn.preprocessing import StandardScaler
import pandas as pd
import logging

class DataTransformer:
    def __init__(self, data):
        self.original_data = data.copy()
        self.scaler = StandardScaler()
        self.columns_to_encode = ["from", "to", "flightType", "agency"]
        self.fitted_columns = None

    def fit_transform(self):
        data = self.original_data.copy()

        # Drop irrelevant columns
        drop_cols = ["travelCode", "userCode", "date"]
        data.drop(columns=drop_cols, inplace=True, errors='ignore')

        # Separate features and target
        X = data.drop(columns=["price"])
        y = data["price"]

        # Encode categorical columns
        X_encoded = pd.get_dummies(X, columns=self.columns_to_encode, drop_first=True)

        # Save column structure for later use in transform
        self.fitted_columns = X_encoded.columns

        # Scale features
        X_scaled = self.scaler.fit_transform(X_encoded)
        X_df = pd.DataFrame(X_scaled, columns=self.fitted_columns)

        logging.info(f"✅ Fit-transform completed: {X_df.shape[0]} rows, {X_df.shape[1]} features.")
        return X_df, y

    def transform(self, new_data):
        data = new_data.copy()

        # Drop irrelevant columns
        drop_cols = ["travelCode", "userCode", "date"]
        data.drop(columns=drop_cols, inplace=True, errors='ignore')

        # Separate features and target
        X = data.drop(columns=["price"])
        y = data["price"]

        # Encode categorical columns
        X_encoded = pd.get_dummies(X, columns=self.columns_to_encode, drop_first=True)

        # Align columns with training data
        X_encoded = X_encoded.reindex(columns=self.fitted_columns, fill_value=0)

        # Scale features
        X_scaled = self.scaler.transform(X_encoded)
        X_df = pd.DataFrame(X_scaled, columns=self.fitted_columns)

        logging.info(f"✅ Transform completed on new data: {X_df.shape[0]} rows.")
        return X_df, y
