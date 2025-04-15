from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd

from utils.data_ingestion import DataLoader
from utils.data_transformation import DataTransformer
from utils.model_training import RandomForestModel

# Path to your CSV file
DATA_FILE_PATH = '/opt/airflow/dags/data/flights.csv'

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

def extract_data():
    loader = DataLoader(DATA_FILE_PATH)
    data = loader.load_data()
    data.to_csv('/opt/airflow/dags/data/cleaned.csv', index=False)
    print("✅ Data extracted and saved to cleaned.csv")

def transform_data():
    data = pd.read_csv('/opt/airflow/dags/data/cleaned.csv')
    transformer = DataTransformer(data)
    X, y = transformer.transform()
    
    pd.DataFrame(X).to_csv('/opt/airflow/dags/data/X.csv', index=False)
    y.to_csv('/opt/airflow/dags/data/y.csv', index=False)
    print("✅ Data transformed and saved as X.csv and y.csv")

def train_model():
    X = pd.read_csv('/opt/airflow/dags/data/X.csv')
    y = pd.read_csv('/opt/airflow/dags/data/y.csv')
    
    model = RandomForestModel(X, y)
    model.random_forest()
    print("✅ Model training completed and saved")

with DAG('flight_price_prediction',
         default_args=default_args,
         description='Daily flight price prediction pipeline',
         schedule_interval='@daily',
         catchup=False) as dag:

    extract = PythonOperator(
        task_id='extract_data',
        python_callable=extract_data
    )
    
    transform = PythonOperator(
        task_id='transform_data',
        python_callable=transform_data
    )
    
    train = PythonOperator(
        task_id='train_model',
        python_callable=train_model
    )

    extract >> transform >> train
