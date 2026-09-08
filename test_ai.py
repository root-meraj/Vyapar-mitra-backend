import requests

try:
    res = requests.get('http://127.0.0.1:8000/api/health')
    print("Health:", res.json())
except Exception as e:
    print("Health failed:", e)

try:
    res = requests.get('http://127.0.0.1:8000/api/ai/health')
    print("AI Health:", res.json())
except Exception as e:
    print("AI Health failed:", e)

try:
    res = requests.get('http://127.0.0.1:8000/api/providers/health')
    print("Providers Health:", res.json())
except Exception as e:
    print("Providers failed:", e)
