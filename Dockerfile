# Menggunakan base image python yang ringan
FROM python:3.10-slim

# Mencegah Python membuat file .pyc dan memastikan output log langsung muncul
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Menentukan folder kerja di dalam container
WORKDIR /app

# Menginstal dependensi sistem yang diperlukan untuk pandas/openpyxl
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements terlebih dahulu agar Docker bisa melakukan caching layer
COPY requirements.txt .

# Install semua library dari requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh kode project ke dalam container
COPY . .

# Ekspos port yang digunakan FastAPI
EXPOSE 8000

# Perintah untuk menjalankan aplikasi menggunakan uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
