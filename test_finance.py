import requests

try:
    # Test valid
    res = requests.post('http://127.0.0.1:8000/api/finance/plan', json={
        "projectCost": 100000,
        "applicantMargin": 10000,
        "requiredCredit": 90000,
        "annualInterestRate": 10.5,
        "tenureMonths": 36,
        "monthlyRevenue": 15000,
        "monthlyOperatingCost": 5000
    })
    print("Finance Valid:", res.json())
    
    # Test incomplete
    res2 = requests.post('http://127.0.0.1:8000/api/finance/plan', json={
        "projectCost": 100000,
        "applicantMargin": -10000,
        "requiredCredit": 90000,
        "annualInterestRate": 0,
        "tenureMonths": 36,
        "monthlyRevenue": 15000,
        "monthlyOperatingCost": 5000
    })
    print("Finance Incomplete:", res2.json())
except Exception as e:
    print("Failed:", e)
