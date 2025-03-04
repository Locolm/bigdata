# Use an official Python runtime as a base image
FROM python:3.12-rc-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app
COPY api.py .
COPY pipeline.pkl .

# Expose FastAPI port
EXPOSE 8080

# Start FastAPI server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
