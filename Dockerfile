# Use Python 3.13 Alpine as the base image
FROM python:3.13-alpine

# Install build dependencies
RUN apk add --no-cache build-base

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Flask app port (assuming port 5000 for Flask)
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "Flight_Price.py"]