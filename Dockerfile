FROM python:3.12-slim

WORKDIR /app
# Install OS-level dependencies needed by ML packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*


# Copy requirements and install
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

# install app 
COPY . /app

# Expose port
EXPOSE 8501

# CMD
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

