# Base image
FROM continuumio/miniconda3:latest

WORKDIR /app

# Copy environment file first (Docker layer caching)
COPY environment.yml .

# Create the conda environment
RUN conda env create -f environment.yml

# Activate environment for subsequent commands
SHELL ["conda", "run", "-n", "human_detection_env", "/bin/bash", "-c"]

# Copy the rest of the project
COPY . .

# Default command to run your Python script
CMD ["conda", "run", "-n", "human_detection_env", "python", "main.py"]
