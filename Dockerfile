# Dockerfile
FROM python:3.9.17-bookworm
# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED True
# Copy local code to the container image.
ENV APP_HOME /back-end
WORKDIR $APP_HOME
COPY . ./

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app

# # Use an official lightweight Python image
# FROM python:3.10-slim

# # Set the working directory in the container
# WORKDIR /app

# # Copy the project files into the container
# COPY . /app

# # Install dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Download the spaCy model
# RUN python -m spacy download en_core_web_sm

# # Expose port 8080 (GCP Cloud Run uses 8080 by default)
# EXPOSE 8080

# # Set environment variables
# ENV FLASK_APP=app.py
# ENV FLASK_RUN_HOST=0.0.0.0
# ENV FLASK_RUN_PORT=8080

# # Run the application
# CMD ["python", "app.py"]
