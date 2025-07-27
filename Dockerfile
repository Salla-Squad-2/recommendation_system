# Base image with Python
FROM python:3.10-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full app
COPY . .

# Run the data upload script
CMD ["python", "upload_to_opensearch.py"]
