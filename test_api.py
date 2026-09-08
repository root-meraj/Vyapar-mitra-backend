import requests

try:
    res = requests.get('http://127.0.0.1:8000/api/health')
    print("Health:", res.json())
except Exception as e:
    print("Health failed:", e)

try:
    res2 = requests.get('http://127.0.0.1:8000/api/ai/health')
    print("AI Health:", res2.json())
except Exception as e:
    print("AI Health failed:", e)
