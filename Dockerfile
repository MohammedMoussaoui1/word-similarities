FROM python:3.10-slim
WORKDIR /app

COPY . .
RUN pip install --no-cache-dir nltk gensim

CMD ["python", "similaritty_vectors..py"]
