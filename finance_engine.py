def calculate_emi(principal: float, annual_rate: float, tenure_months: int) -> float:
    if tenure_months <= 0: return 0
    r = annual_rate / 100 / 12
    if r == 0: return principal / tenure_months
    factor = (1 + r) ** tenure_months
    return (principal * r * factor) / (factor - 1)

def calculate_financials(margin_capital: float) -> dict:
    # Scheme Definitions
    MICRO_FINANCE = {
        'type': 'micro_finance',
        'name': 'Micro Finance Scheme',
        'nameHi': 'माइक्रो फाइनेंस योजना',
        'maxLoan': 1_25_000,
        'annualInterestRate': 6.5,
        'tenureYears': 3,
        'moratoriumMonths': 3,
        'description': 'For small-scale micro enterprises with project cost up to ₹1,40,000. Maximum loan of ₹1,25,000 at 6.5% annual interest with 3-year repayment.'
    }

    TERM_LOAN = {
        'type': 'term_loan',
        'name': 'Term Loan Scheme',
        'nameHi': 'सावधि ऋण योजना',
        'maxLoan': 45_00_000,
        'annualInterestRate': 8.0,
        'tenureYears': 7,
        'moratoriumMonths': 6,
        'description': 'For medium enterprises with project cost between ₹1,40,001 and ₹50,00,000. Maximum loan of ₹45,00,000 at 8% annual interest with 7-year repayment.'
    }

    if margin_capital <= 0:
        return {
            'isEligible': False,
            'marginCapital': 0,
            'projectCost': 0,
            'loanAmount': 0,
            'ineligibleReason': 'Margin capital must be greater than 0.'
        }

    project_cost = margin_capital * 10
    loan_amount = project_cost * 0.9

    if project_cost <= 1_40_000:
        scheme = MICRO_FINANCE
    elif project_cost <= 50_00_000:
        scheme = TERM_LOAN
    else:
        return {
            'isEligible': False,
            'marginCapital': margin_capital,
            'projectCost': project_cost,
            'loanAmount': loan_amount,
            'ineligibleReason': 'Project cost exceeds maximum limit of ₹50,00,000 for available schemes.'
        }

    capped_loan_amount = min(loan_amount, scheme['maxLoan'])
    
    annual_rate = scheme['annualInterestRate']
    tenure_years = scheme['tenureYears']
    moratorium_months = scheme['moratoriumMonths']
    
    monthly_rate = annual_rate / 100 / 12
    balance = capped_loan_amount
    
    moratorium_interest = 0
    amortization_table = []
    
    # Moratorium period
    for m in range(1, moratorium_months + 1):
        interest_accrued = balance * monthly_rate
        moratorium_interest += interest_accrued
        new_balance = balance + interest_accrued
        amortization_table.append({
            'month': m,
            'openingBalance': round(balance, 2),
            'emi': 0,
            'principalPaid': 0,
            'interestPaid': 0,
            'closingBalance': round(new_balance, 2),
            'isMoratorium': True
        })
        balance = new_balance
        
    capitalized_principal = balance
    repayment_months = tenure_years * 12 - moratorium_months
    emi = calculate_emi(capitalized_principal, annual_rate, repayment_months)
    
    total_repayment = 0
    total_interest_paid = moratorium_interest
    
    for m in range(moratorium_months + 1, tenure_years * 12 + 1):
        interest = balance * monthly_rate
        principal = emi - interest
        
        # Adjust last month rounding
        if m == tenure_years * 12:
            principal = balance
            emi = principal + interest
            
        new_balance = balance - principal
        total_repayment += emi
        total_interest_paid += interest
        
        amortization_table.append({
            'month': m,
            'openingBalance': round(balance, 2),
            'emi': round(emi, 2),
            'principalPaid': round(principal, 2),
            'interestPaid': round(interest, 2),
            'closingBalance': max(0, round(new_balance, 2)),
            'isMoratorium': False
        })
        balance = new_balance

    return {
        'isEligible': True,
        'marginCapital': margin_capital,
        'projectCost': project_cost,
        'loanAmount': loan_amount,
        'cappedLoanAmount': capped_loan_amount,
        'scheme': scheme,
        'monthlyEMI': emi,
        'moratoriumInterest': moratorium_interest,
        'capitalizedPrincipal': capitalized_principal,
        'totalRepayment': total_repayment,
        'totalInterestPaid': total_interest_paid,
        'amortizationTable': amortization_table
    }

def calculate_scenario_financials(margin_capital: float, avg_revenue: float, avg_operating_cost: float) -> dict:
    fin = calculate_financials(margin_capital)
    if not fin.get('isEligible'):
        return None
        
    emi = fin.get('monthlyEMI', 0)
    
    def calc_scenario(rev, cost):
        surplus = max(0, rev - cost)
        ratio = (emi / surplus * 100) if surplus > 0 else (100 if emi > 0 else 0)
        return {
            'revenue': rev,
            'operatingCost': cost,
            'surplus': surplus,
            'emiRatio': ratio,
            'isSafe': ratio <= 50
        }
        
    return {
        'optimistic': calc_scenario(avg_revenue * 1.2, avg_operating_cost * 0.9),
        'expected': calc_scenario(avg_revenue, avg_operating_cost),
        'conservative': calc_scenario(avg_revenue * 0.8, avg_operating_cost * 1.1)
    }

def calculate_repayment_readiness(margin_capital: float, avg_revenue: float, avg_operating_cost: float) -> dict:
    fin = calculate_financials(margin_capital)
    if not fin.get('isEligible'):
        return {
            'status': 'High Risk',
            'statusHi': 'उच्च जोखिम',
            'ratio': 100,
            'message': fin.get('ineligibleReason', 'Ineligible'),
            'messageHi': 'अपात्र'
        }
        
    emi = fin.get('monthlyEMI', 0)
    surplus = max(0, avg_revenue - avg_operating_cost)
    ratio = (emi / surplus * 100) if surplus > 0 else 100
    
    if ratio < 35:
        return {
            'status': 'Comfortable',
            'statusHi': 'आरामदायक',
            'ratio': ratio,
            'message': 'Your expected surplus easily covers the EMI. You have a strong buffer for unexpected expenses.',
            'messageHi': 'आपका अपेक्षित अधिशेष आसानी से EMI को कवर करता है।'
        }
    elif ratio <= 50:
        return {
            'status': 'Caution',
            'statusHi': 'सावधानी',
            'ratio': ratio,
            'message': 'EMI is manageable, but takes up a significant portion of surplus. Maintain an emergency fund.',
            'messageHi': 'EMI प्रबंधनीय है, लेकिन अधिशेष का एक महत्वपूर्ण हिस्सा लेती है।'
        }
    else:
        return {
            'status': 'High Risk',
            'statusHi': 'उच्च जोखिम',
            'ratio': ratio,
            'message': 'EMI exceeds safe limits. Consider starting smaller or increasing margin capital to reduce loan size.',
            'messageHi': 'EMI सुरक्षित सीमा से अधिक है। ऋण का आकार कम करने के लिए मार्जिन पूंजी बढ़ाने पर विचार करें।'
        }
