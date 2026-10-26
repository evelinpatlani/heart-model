"""Entrena el modelo de precio de autos usados.

Ejecutar desde la raíz del proyecto:
    python model/train.py
"""
import json
import os
import pathlib

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

DATASET_PATH = pathlib.Path(os.environ.get('DATASET_PATH', 'data/cars.csv'))
MODEL_PATH = pathlib.Path('model/car-price-v1.joblib')

CAT_FEATURES = ['brand', 'model', 'transmission', 'fuelType']
NUM_FEATURES = ['year', 'mileage', 'engineSize']
TARGET = 'price'

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    missing = set(CAT_FEATURES + NUM_FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f'Al dataset le faltan columnas: {sorted(missing)}')

    df['brand'] = df['brand'].str.strip().str.lower()
    df['model'] = df['model'].str.strip()
    df = df.drop_duplicates()
    # Limpieza de valores absurdos (p. ej. year=2060, engineSize=0)
    df = df[df['year'].between(1995, 2020)
            & (df['engineSize'] > 0)
            & (df['mileage'] > 0)]
    return df.reset_index(drop=True)


def build_pipeline() -> Pipeline:
    encoder = ColumnTransformer(
        [('cat', OrdinalEncoder(handle_unknown='use_encoded_value',
                                unknown_value=np.nan), CAT_FEATURES)],
        remainder='passthrough')
    regressor = HistGradientBoostingRegressor(
        categorical_features=[0, 1, 2, 3],  # las 4 primeras tras el encoder
        max_iter=400, learning_rate=0.08, random_state=0)
    # Se predice log(precio) para que el error sea proporcional
    target_model = TransformedTargetRegressor(
        regressor=regressor, func=np.log1p, inverse_func=np.expm1)
    return Pipeline([('prep', encoder), ('reg', target_model)])


if __name__ == '__main__':
    df = load_data()
    print(f'Dataset: {DATASET_PATH}')
    print(f'Filas: {len(df)} | Marcas: {df.brand.nunique()} | Modelos: {df.model.nunique()}')

    X = df[CAT_FEATURES + NUM_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print('Training model.. ')
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)
    mae, r2 = mean_absolute_error(y_test, pred), r2_score(y_test, pred)
    print(f'MAE: £{mae:,.0f} | R2: {r2:.3f}')

    # Valores válidos para que la API pueda validar la entrada
    options = {
        'brands': sorted(df['brand'].unique().tolist()),
        'transmissions': sorted(df['transmission'].unique().tolist()),
        'fuel_types': sorted(df['fuelType'].unique().tolist()),
        'models_by_brand': {b: sorted(g['model'].unique().tolist())
                            for b, g in df.groupby('brand')},
    }

    print('Saving model..')
    dump({'pipeline': pipeline,
          'options': options,
          'metrics': {'mae': float(mae), 'r2': float(r2)}},
         MODEL_PATH, compress=3)
    (MODEL_PATH.parent / 'options.json').write_text(
        json.dumps(options, indent=2), encoding='utf-8')
