from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from joblib import load
import pathlib
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


@app.get('/options')
def get_options():
    """Marcas, modelos, transmisiones y combustibles válidos."""
    return options


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
