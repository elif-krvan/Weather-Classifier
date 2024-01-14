FROM python:3.10

COPY src/ /app
WORKDIR /app

COPY requirements.txt .
RUN apt-get update && \ 
apt-get install -y libgl1

RUN pip install -r requirements.txt

ENV PORT 8080

CMD gunicorn --bind :$PORT --workers 1 --threads 8 app:app & python clean.py