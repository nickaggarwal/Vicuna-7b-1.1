FROM nvcr.io/nvidia/tritonserver:22.11-py3

RUN apt update && apt -y install libssl-dev tesseract-ocr libtesseract-dev ffmpeg

RUN pip install --upgrade pip
 
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .

CMD ["python3", "app.py"]