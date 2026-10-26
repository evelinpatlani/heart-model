import requests
body = {
    "brand": "audi",
    "model": "A3",
    "year": 2017,
    "transmission": "Manual",
    "mileage": 25000,
    "fuelType": "Petrol",
    "engineSize": 1.4
    }
response = requests.post(url = 'http://127.0.0.1:8000/score',
              json = body)
print (response.json())
# output (aprox.): {'price': 15500.0, 'currency': 'GBP'}
