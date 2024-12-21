FROM python:3.9-slim
WORKDIR /app
COPY ./model /app/model
COPY requirements.txt /app/
RUN pip install -r requirements.txt
COPY app.py /app/
EXPOSE 5000
CMD ["python", "app.py"]
