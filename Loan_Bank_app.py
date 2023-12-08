import joblib
import uvicorn
from fastapi import FastAPI
import pandas as pd
from prometheus_client import Counter, make_asgi_app

eligible_counter = Counter("eligible", "Counter for eligible")
not_eligible_counter = Counter("not_eligible", "Counter for not eligible")

app = FastAPI()
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/reda")
def prediction_api(
    checking_account_status: int,
    duration_months: int,
    credit_history: int,
    purpose: int,
    credit_amount: int,
    savings_account: int,
    employment_duration: int,
    installment_rate: int,
    personal_status_sex: int,
    other_debtors_guarantors: int,
    residence_duration: int,
    property: int,
    age_years: int,
    other_installment_plans: int,
    housing: int,
    existing_credits_at_bank: int,
    job: int,
    people_liable_for_maintenance: int,
    telephone: int,
    foreign_worker: int
):
    
    loan_model = joblib.load("./loan_SVC.joblib")
    x = [
    'checking_account_status', 'duration_months', 'credit_history', 'purpose', 'credit_amount',
    'savings_account', 'employment_duration', 'installment_rate', 'personal_status_sex',
    'other_debtors_guarantors', 'residence_duration', 'property', 'age_years',
    'other_installment_plans', 'housing', 'existing_credits_at_bank', 'job',
    'people_liable_for_maintenance', 'telephone', 'foreign_worker'
        ]
#The output of our model is either 1 (bad loan), hence not eligible or 2 (good loan), eligible.

    prediction = loan_model.predict(pd.DataFrame(x).transpose())
    eligible = int(prediction) == 2
    if eligible:
        eligible_counter.inc()
        return "this client will be a good candidate for a bank loan"
    else:
        not_eligible_counter.inc()
        return "this client will be a bad candidate for a bank loan"
    return eligible


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)