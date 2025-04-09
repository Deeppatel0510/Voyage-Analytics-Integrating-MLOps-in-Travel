# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the application files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Flask app port
EXPOSE 8000

# # Define environment variables for Flask
# ENV FLASK_APP=Flight_Price.py
# ENV FLASK_RUN_HOST=0.0.0.0

# Run the application
CMD ["python", "Flight_Price.py"]