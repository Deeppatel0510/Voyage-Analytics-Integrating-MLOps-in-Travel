import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# ✅ Tell MLflow to use the tracking server
mlflow.set_tracking_uri("http://localhost:5000")

# Data
data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)

# Model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict & Evaluate
preds = model.predict(X_test)
rmse = sqrt(mean_squared_error(y_test, preds))

# Log to MLflow
with mlflow.start_run():
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(model, "model")