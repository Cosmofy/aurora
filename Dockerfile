FROM python:3.11-slim
WORKDIR /app # sets the working directory inside the container
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 2260
CMD ["gunicorn", "inference.app:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:2260", "--backlog", "2048", "--timeout", "120"]
