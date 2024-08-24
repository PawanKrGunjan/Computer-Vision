# Use the official Python image as a base
FROM python:3.8-slim

# Set environment variables to prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install the necessary dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the application code and pre-trained models into the container
COPY License_Plate_Recognizer.py /app/License_Plate_Recognizer.py
COPY app.py /app/app.py
COPY ANPR.py /app/ANPR.py
COPY saved_model /app/saved_model

# Set the command to run the application
CMD ["python", "app.py","License_Plate_Recognizer.py","ANPR.py"]
