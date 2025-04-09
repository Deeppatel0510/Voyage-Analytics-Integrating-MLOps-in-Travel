from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'deep',
    'depends_on_past': False,
    'email': ['your@email.com'],
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG('flight_price_deploy_only',
         default_args=default_args,
         description='A DAG to deploy flight price Flask API container',
         schedule_interval='@weekly',
         start_date=datetime(2025, 4, 1),
         catchup=False) as dag:

    deploy_model = DockerOperator(
        task_id='deploy_model',
        image='deep0510/flight-price-api',  # Your Docker image with Flask app
        api_version='auto',
        auto_remove=True,
        command="python Flight_Price.py",   # Run the Flask API
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge"
    )
