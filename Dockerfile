#Dockerfile

# --- Etapa 1: entrena el modelo con el MISMO scikit-learn que usará la API ---
FROM python:3.12-slim AS trainer
WORKDIR /app
COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt
COPY ./data ./data
COPY ./model/train.py ./model/train.py
RUN python model/train.py

# --- Etapa 2: imagen final, solo la API y el modelo (sin el dataset) ---
FROM python:3.12-slim
WORKDIR /app
COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt
COPY ./app.py .
COPY --from=trainer /app/model ./model
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
