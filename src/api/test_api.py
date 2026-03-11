import requests
import json

BASE_URL = "http://127.0.0.1:5000"

#Test 1 - health check

print("Test 1: Health Check")
response = requests.get(f"{BASE_URL}/health")
print(json.dumps(response.json(), indent=2))

#Test 2 - Model Info

print("\nTest 2 — Model Info")
response = requests.get(f"{BASE_URL}/model-info")
print(json.dumps(response.json(), indent=2))

# ================================
# Test 3 — Single Prediction
# ================================
print("\nTest 3 — Single Prediction")
sensor_data = {
    "engine_id": 1,
    "cycle": 150,
    "s2": 0.65,
    "s3": 0.55,
    "s4": 0.82,
    "s7": 0.45,
    "s8": 0.71,
    "s9": 0.63,
    "s11": 0.78,
    "s12": 0.34,
    "s13": 0.56,
    "s14": 0.67,
    "s15": 0.72,
    "s17": 0.58,
    "s20": 0.43,
    "s21": 0.39
}
response = requests.post(
    f"{BASE_URL}/predict",
    json=sensor_data
)
print(json.dumps(response.json(), indent=2))

# ================================
# Test 4 — Batch Prediction
# ================================
print("\nTest 4 — Batch Prediction")
batch_data = {
    "engines": [
        {
            "engine_id": 1, "cycle": 150,
            "s2": 0.65, "s3": 0.55, "s4": 0.82,
            "s7": 0.45, "s8": 0.71, "s9": 0.63,
            "s11": 0.78, "s12": 0.34, "s13": 0.56,
            "s14": 0.67, "s15": 0.72, "s17": 0.58,
            "s20": 0.43, "s21": 0.39
        },
        {
            "engine_id": 2, "cycle": 50,
            "s2": 0.32, "s3": 0.28, "s4": 0.35,
            "s7": 0.71, "s8": 0.42, "s9": 0.81,
            "s11": 0.29, "s12": 0.68, "s13": 0.41,
            "s14": 0.78, "s15": 0.35, "s17": 0.29,
            "s20": 0.72, "s21": 0.65
        },
        {
            "engine_id": 3, "cycle": 200,
            "s2": 0.91, "s3": 0.88, "s4": 0.95,
            "s7": 0.22, "s8": 0.93, "s9": 0.31,
            "s11": 0.96, "s12": 0.18, "s13": 0.89,
            "s14": 0.42, "s15": 0.94, "s17": 0.87,
            "s20": 0.21, "s21": 0.19
        }
    ]
}
response = requests.post(
    f"{BASE_URL}/predict/batch",
    json=batch_data
)
print(json.dumps(response.json(), indent=2))
