FROM python:3.8-slim

RUN apt-get update && \
    apt-get install -y fluidsynth && \
    apt-get clean

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

CMD ["streamlit", "run", "app.py"]
