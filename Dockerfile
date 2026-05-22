# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PORT 5000

# Set work directory
WORKDIR /app

# Install system dependencies (required for some Python packages like PyMuPDF or compiling extensions)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy the rest of the application
COPY . /app/

# Create uploads directory
RUN mkdir -p uploads chroma_db

# Expose port
EXPOSE $PORT

# Command to run the application using gunicorn
CMD gunicorn app:app --workers 2 --timeout 120 --bind 0.0.0.0:$PORT
