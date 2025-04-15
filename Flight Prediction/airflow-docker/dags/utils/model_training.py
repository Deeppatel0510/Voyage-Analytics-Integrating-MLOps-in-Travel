import pickle
import os
from sklearn.ensemble import RandomForestRegressor

class RandomForestModel:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def random_forest(self):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(self.X, self.y)

        os.makedirs("/opt/airflow/dags/models", exist_ok=True)
        with open("/opt/airflow/dags/models/random_forest.pkl", "wb") as f:
            pickle.dump(model, f)
        print("📦 Model saved to /opt/airflow/dags/models/random_forest.pkl")
