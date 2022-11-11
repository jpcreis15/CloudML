# syntax=docker/dockerfile:1

FROM python:3.10-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
COPY Makefile Makefile

EXPOSE 8000

RUN python -m pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "server_app:app", "--host", "0.0.0.0", "--port", "8000"]