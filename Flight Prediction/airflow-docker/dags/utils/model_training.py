import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt
import mlflow
import mlflow.sklearn

class RandomForestModel:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def random_forest(self):  
        # split data  
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)  

        model = RandomForestRegressor(n_estimators=100, random_state=42)  
        model.fit(X_train, y_train)  

        # evaluate  
        preds = model.predict(X_test)  
        rmse = sqrt(mean_squared_error(y_test, preds))  

        # log to MLflow  
        mlflow.set_tracking_uri("http://mlflow:5000")  # optional if already set in env  
        with mlflow.start_run():  
            mlflow.log_param("model", "RandomForestRegressor")  
            mlflow.log_param("n_estimators", 100)  
            mlflow.log_metric("rmse", rmse)  
            mlflow.sklearn.log_model(model, "model", registered_model_name="flight_rf_model")  

        # also save locally (for Flask API if needed)  
        model_dir = "/opt/airflow/dags/models"  
        os.makedirs(model_dir, exist_ok=True)  
        with open(os.path.join(model_dir, "random_forest.pkl"), "wb") as f:  
            pickle.dump(model, f)  

