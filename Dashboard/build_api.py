from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load model đã train
model = joblib.load("D:\ml\Customer_Churn\saved_models\logreg_best_model.pkl")

# Khởi tạo app
app = FastAPI()

# Định nghĩa input schema (các cột phải khớp lúc training)
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def predict(data: CustomerData):
    # Convert input → DataFrame
    df = pd.DataFrame([data.dict()])
    
    # Predict probability churn
    prob = model.predict_proba(df)[0][1]
    label = "Churn" if prob > 0.5 else "Not Churn"
    
    return {"probability": prob, "prediction": label}
