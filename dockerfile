FROM python:3.13-slim

WORKDIR /app

COPY Assessmentagent/Backend/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "Assessmentagent.Backend.main:app", "--host", "0.0.0.0", "--port", "8000"]