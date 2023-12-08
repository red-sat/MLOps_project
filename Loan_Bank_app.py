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
def prediction_api(pclass: int, sex: int, age: int):
    loan_model = joblib.load("./loan_SVC.joblib")
    x = ['status of existing checking account','Duration in months','Credit History','Purpose','Credit Amount','Savings account/bonds','Present employement since','installement rate in percentage of disposable income','Personal status and sex','Other debtors / guarantors','present residence since','Property','Age in years','Other installment plans', 'Housing', 'Number of existing credits at this bank', 'Job', 'Number of people being liable to provide maintenance for', 'Telephone', 'foreign worker']
    prediction = loan_model.predict(pd.DataFrame(x).transpose())
    eligible = int(prediction) == 1
    if eligible:
        eligible_counter.inc()
    else:
        not_eligible_counter.inc()
    return eligible


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)