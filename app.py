from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from joblib import load
import hmac
import os
import pathlib
import re
from urllib.parse import urlparse

import requests
from fastapi.middleware.cors import CORSMiddleware

origins = ["*"]

app = FastAPI(title='Used Car Price Prediction')

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

bundle = load(pathlib.Path('model/car-price-v1.joblib'))
model = bundle['pipeline']
options = bundle['options']
metrics = bundle['metrics']

# --- Configuración del reentrenamiento (se define como variables de entorno en Render) ---
GITHUB_REPO = os.environ.get('GITHUB_REPO', '')    # ej. evelinpatlani/heart-model
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '')  # token con permiso para repository_dispatch
RETRAIN_KEY = os.environ.get('RETRAIN_KEY', '')    # clave que debe escribir quien pulsa el botón
ALLOWED_HOSTS = ('res.cloudinary.com', 'firebasestorage.googleapis.com')


class InputData(BaseModel):
    brand: str = 'audi'
    model: str = 'A3'
    year: int = Field(2017, ge=1995, le=2020)
    transmission: str = 'Manual'
    mileage: int = Field(25000, gt=0)
    fuelType: str = 'Petrol'
    engineSize: float = Field(1.4, gt=0)


class OutputData(BaseModel):
    price: float = 15000.0
    currency: str = 'GBP'


class RetrainData(BaseModel):
    dataseturl: str
    sha: str


@app.get('/options')
def get_options():
    """Marcas, modelos, transmisiones y combustibles válidos."""
    return options


@app.get('/info')
def get_info():
    """Datos del modelo que está sirviendo la API ahora mismo."""
    return metrics


@app.post('/score', response_model=OutputData)
def score(data: InputData):
    brand = data.brand.strip().lower()
    car_model = data.model.strip()

    if brand not in options['models_by_brand']:
        raise HTTPException(422, f"Marca desconocida: '{data.brand}'. Ver GET /options")
    if car_model not in options['models_by_brand'][brand]:
        raise HTTPException(422, f"Modelo '{car_model}' no existe para {brand}. Ver GET /options")
    if data.transmission not in options['transmissions']:
        raise HTTPException(422, f"transmission debe ser una de {options['transmissions']}")
    if data.fuelType not in options['fuel_types']:
        raise HTTPException(422, f"fuelType debe ser una de {options['fuel_types']}")

    model_input = pd.DataFrame([{
        'brand': brand, 'model': car_model,
        'transmission': data.transmission, 'fuelType': data.fuelType,
        'year': data.year, 'mileage': data.mileage, 'engineSize': data.engineSize,
    }])
    result = model.predict(model_input)

    # Convertimos el elemento del array de numpy a un float de Python
    return {'price': round(float(result[0]), 2), 'currency': 'GBP'}


@app.post('/retrain', status_code=202)
def retrain(data: RetrainData, x_admin_key: str = Header(default='')):
    """Dispara el workflow de GitHub Actions que reentrena y publica una imagen nueva."""
    if not (GITHUB_REPO and GITHUB_TOKEN and RETRAIN_KEY):
        raise HTTPException(503, 'El reentrenamiento no está configurado en el servidor')
    if not hmac.compare_digest(x_admin_key, RETRAIN_KEY):
        raise HTTPException(401, 'Clave incorrecta')

    url = urlparse(data.dataseturl)
    if url.scheme != 'https' or url.hostname not in ALLOWED_HOSTS:
        raise HTTPException(422, f'dataseturl debe ser https y de: {", ".join(ALLOWED_HOSTS)}')
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]{0,60}', data.sha):
        raise HTTPException(422, 'sha solo admite letras, números, _ . - (máx. 61)')

    try:
        r = requests.post(
            f'https://api.github.com/repos/{GITHUB_REPO}/dispatches',
            headers={'Authorization': f'Bearer {GITHUB_TOKEN}',
                     'Accept': 'application/vnd.github.v3+json'},
            json={'event_type': 'ml_ci_cd',
                  'client_payload': {'dataseturl': data.dataseturl, 'sha': data.sha}},
            timeout=15)
    except requests.RequestException:
        raise HTTPException(502, 'No se pudo contactar a GitHub')
    if r.status_code != 204:
        raise HTTPException(502, f'GitHub respondió {r.status_code}')
    return {'status': 'started', 'sha': data.sha,
            'message': 'Reentrenamiento iniciado; el modelo nuevo se desplegará en unos minutos'}