# Use the official Apache Airflow image as the base
FROM apache/airflow:2.10.5

# Switch to root to install dependencies
USER root

# (Optional) Install system packages here
# RUN apt-get update && apt-get install -y gcc libpq-dev curl && apt-get clean

# Switch back to airflow user
USER airflow

# Copy requirements.txt and install any custom Python packages
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
