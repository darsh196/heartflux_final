from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import jwt
import os
from fastapi import Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
import os
from datetime import datetime
from tensorflow.keras.models import load_model
import tensorflow as tf
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras import backend as K
import math
import numpy as np
import shap

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all domains 
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (POST, GET, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Database connection
MONGO_URI = "mongodb+srv://darshan:PkHmZoXR8e4cJgYF@cluster0.eafea.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["heartflux"]
users_collection = db["users"]

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT secret key 
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # Token expires in 1 hour


# User Schema
class User(BaseModel):
    name: str
    email: EmailStr
    password: str

@app.post("/signup")
async def signup(user: User):
    # Check if email already exists
    if users_collection.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Hash password before storing
    hashed_password = pwd_context.hash(user.password)
    
    # Store user in database
    users_collection.insert_one({
        "name": user.name,
        "email": user.email,
        "password": hashed_password
    })
    # Generate JWT token after signup
    access_token = create_access_token(
        data={"sub": user.name}, expires_delta=timedelta(minutes=60)
    )

    return {"access_token": access_token, "token_type": "bearer", "message": "User registered successfully"}

class UserLogin(BaseModel):
    name: str
    password: str

def create_access_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

@app.post("/login")
async def login(user: UserLogin):
    # Find user in database
    db_user = users_collection.find_one({"name": user.name})
    
    if not db_user:
        raise HTTPException(status_code=400, detail="Invalid username or password")

    # Verify password
    if not pwd_context.verify(user.password, db_user["password"]):
        raise HTTPException(status_code=400, detail="Invalid email or password")
    
    # Generate JWT token
    access_token = create_access_token(
        data={"sub": user.name}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    return {"access_token": access_token, "token_type": "bearer"}

# <---------------- Handling Health Data ------------------>

security = HTTPBearer()

# Pydantic schema for health data
class HealthData(BaseModel):
    age: int
    gender: str
    height: int
    weight: float
    systolic_bp: int
    diastolic_bp: int
    cholesterol: int
    glucose: float
    smoker: str
    alcohol: str
    activity: str
# Dependency to get current user from JWT
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/submit-health-data")
def submit_health_data(data: HealthData, username: str = Depends(get_current_user)):
    result = users_collection.update_one(
        {"name": username},
        {"$push": {
            "health_data": {
                "age": data.age,
                "gender": data.gender,
                "height": data.height,
                "weight": data.weight,
                "systolic_bp": data.systolic_bp,
                "diastolic_bp": data.diastolic_bp,
                "cholesterol": data.cholesterol,
                "glucose": data.glucose,
                "smoker": data.smoker,
                "alcohol": data.alcohol,
                "activity": data.activity
            }
        }}
    )

    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="User not found")

    return {"message": "Health data submitted successfully"}

# <---------------- Getting Risk Prediction ------------------>

# Load the model from model folder 
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "xgb_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "model", "scaler.pkl")

xgb_model = joblib.load(MODEL_PATH)
explainer_xgb = shap.Explainer(xgb_model)
scaler = joblib.load(SCALER_PATH)

@app.get("/predict-risk")
def predict_risk(username: str = Depends(get_current_user)):
    user = users_collection.find_one({"name": username})
    if not user or "health_data" not in user or len(user["health_data"]) == 0:
        raise HTTPException(status_code=404, detail="No health data available")

    latest = user["health_data"][-1]  # Use the most recent entry

    # Convert cholesterol mg/dL to model category:
    # Normal < 200 mg/dL → 1
    # 200–239 → 2
    # 240+ → 3
    chol_cat = 1 if latest["cholesterol"] < 200 else 2 if latest["cholesterol"] < 240 else 3

    # Convert glucose mmol/L to model category:
    # Normal < 5.6 → 1
    # 5.6–6.9 → 2
    # ≥7.0 → 3
    glucose_cat = 1 if latest["glucose"] < 5.6 else 2 if latest["glucose"] < 7.0 else 3

    # Gender: female = 1, male = 2 (as per your model)
    gender_code = 2 if latest["gender"].lower() == "male" else 1

    features = pd.DataFrame([{
        "age": latest["age"],
        "gender": gender_code,
        "height": latest["height"],
        "weight": latest["weight"],
        "ap_hi": latest["systolic_bp"],
        "ap_lo": latest["diastolic_bp"],
        "cholesterol": chol_cat,
        "gluc": glucose_cat,
        "smoke": 1 if latest["smoker"].lower() == "yes" else 0,
        "alco": 1 if latest["alcohol"].lower() == "yes" else 0,
        "active": 1 if latest["activity"].lower() == "yes" else 0
    }])

    # Scale the features before prediction
    features_scaled = scaler.transform(features)

    # Predict using scaled input
    prediction = xgb_model.predict_proba(features_scaled)[0][1]
    # Save prediction to risk history
    users_collection.update_one(
        {"name": username},
        {"$push": {
            "risk_history": {
                "value": float(round(prediction * 100, 1)),
                "timestamp": datetime.now().isoformat()
            }
        }}
    )

    # Generate SHAP values
    shap_values = explainer_xgb(features_scaled)
    shap_contributions = {
        feature: round(float(shap_values[0, i].values), 4)
        for i, feature in enumerate(features.columns)
    }


    return {
        "risk_percent": float(round(prediction * 100, 1)),
        "shap_contributions": shap_contributions
    }

    

# <---------------- Get Health History ------------------>

@app.get("/health-history")
def get_health_history(username: str = Depends(get_current_user)):
    user = users_collection.find_one({"name": username})
    if not user or "health_data" not in user:
        raise HTTPException(status_code=404, detail="No health data found")
    return {"history": user["health_data"]}

@app.get("/risk-history")
def get_risk_history(username: str = Depends(get_current_user)):
    user = users_collection.find_one({"name": username})
    if not user or "risk_history" not in user:
        raise HTTPException(status_code=404, detail="No risk history found")
    return {"history": user["risk_history"]}


# <---------------- Recommendations ------------------>
# Load LSTM model and scaler
LSTM_MODEL_PATH = os.path.join("model", "lstm_model_better2.h5")
SCALER_PATH = os.path.join("model", "lstm_scaler_better.pkl")

lstm_model = load_model(LSTM_MODEL_PATH)
lstm_scaler = joblib.load(SCALER_PATH)

@app.get("/recommendations")
def get_recommendations(username: str = Depends(get_current_user)):
    user = users_collection.find_one({"name": username})
    if not user or "health_data" not in user or len(user["health_data"]) < 5:
        raise HTTPException(status_code=404, detail="Not enough health data to generate recommendations.")

    recent_health = user["health_data"][-5:]
    recent_risk = user.get("risk_history", [])[-5:]

    if len(recent_health) != len(recent_risk):
        raise HTTPException(status_code=400, detail="Mismatch between health data and risk history.")

    # Merge risk into health entries
    combined = []
    for h, r in zip(recent_health, recent_risk):
        h["risk_percent"] = r["value"]
        combined.append(h)

    df = pd.DataFrame(combined)

    # Convert to binary
    df["smoker"] = df["smoker"].str.lower().map({"yes": 1, "no": 0})
    df["alcohol"] = df["alcohol"].str.lower().map({"yes": 1, "no": 0})
    df["activity"] = df["activity"].str.lower().map({"active": 1, "not active": 0})

    # Add delta columns
    # for col in ['risk_percent', 'systolic_bp', 'diastolic_bp', 'cholesterol', 'glucose', 'weight']:
    #     df[f"{col}_delta"] = df[col].diff().fillna(0)


    # Check columns AFTER everything is ready
    # features = ['systolic_bp', 'diastolic_bp', 'cholesterol', 'glucose',
    #     'weight', 'risk_percent', 'smoker', 'alcohol', 'activity',
    #     'risk_percent_delta', 'systolic_bp_delta', 'diastolic_bp_delta',
    #     'cholesterol_delta', 'glucose_delta', 'weight_delta']
    features = ['systolic_bp', 'diastolic_bp', 'cholesterol', 'glucose',
         'weight', 'risk_percent', 'smoker', 'alcohol', 'activity']
    if not all(f in df.columns for f in features):
        raise HTTPException(status_code=400, detail="Missing required fields in final data.")

    # Reshape and scale
    X = df[features].values
    
    X_scaled = lstm_scaler.transform(X).reshape(1, 5, len(features))
    print("Scaled input:", X_scaled)
    print("Any NaN?", np.isnan(X_scaled).any())
    print("Any Inf?", np.isinf(X_scaled).any())

    # Predict trend
    # prediction = lstm_model.predict(X_scaled)[0][0]

    # raw_confidence = float(prediction) * 100
    # confidence = round(raw_confidence, 1) if math.isfinite(raw_confidence) else 0.0

     # Predict
    prediction_vector = lstm_model.predict(X_scaled)[0]
    predicted_class = int(np.argmax(prediction_vector))  # 0: improving, 1: stable, 2: worsening
    confidence = round(float(np.max(prediction_vector)) * 100, 1)

    trend_mapping = {0: "improving", 1: "stable", 2: "worsening"}
    trend = trend_mapping[predicted_class]



    # Generate recommendations
    latest = combined[-1]
    recs = []

     # === Integrated Gradients (IG) simplified approach ===
    try:
        # Use a more sensitive approach that adds noise rather than zeroing out
        feature_importance = {}
        baseline_pred = lstm_model.predict(X_scaled)[0][predicted_class]
        
        # Number of perturbation samples per feature
        n_samples = 5
        
        # For each feature, calculate importance by perturbation
        for i, feature in enumerate(features):
            total_impact = 0.0
            
            for _ in range(n_samples):
                # Create a copy with random noise for this feature only
                X_modified = X_scaled.copy()
                
                # Get the standard deviation of the feature
                feature_std = np.std(X_modified[0, :, i])
                if feature_std == 0:  # If std is 0, use a small value
                    feature_std = 0.1
                
                # Add random noise to this feature (different for each time step)
                for t in range(X_modified.shape[1]):  # For each time step
                    noise = np.random.normal(0, feature_std)
                    X_modified[0, t, i] += noise
                
                # Get prediction with feature perturbed
                modified_pred = lstm_model.predict(X_modified)[0][predicted_class]
                
                # Calculate impact as absolute change in prediction
                impact = abs(baseline_pred - modified_pred)
                total_impact += impact
            
            # Average impact across samples
            feature_importance[feature] = float(total_impact / n_samples)
        
        # If all values are 0, use a simpler approach
        if sum(feature_importance.values()) == 0:
            # Try larger perturbations
            for i, feature in enumerate(features):
                X_modified = X_scaled.copy()
                # Shuffle the feature values across time steps
                feature_values = X_modified[0, :, i].copy()
                np.random.shuffle(feature_values)
                X_modified[0, :, i] = feature_values
                
                modified_pred = lstm_model.predict(X_modified)[0][predicted_class]
                impact = abs(baseline_pred - modified_pred)
                feature_importance[feature] = float(impact)
        
        # Normalize scores if any non-zero values exist
        total = sum(feature_importance.values())
        if total > 0:
            feature_importance = {k: round((v / total) * 100, 2) for k, v in feature_importance.items()}
            
            # Format display names
            feature_display_names = {
                'systolic_bp': 'systolic blood pressure',
                'diastolic_bp': 'diastolic blood pressure',
                'cholesterol': 'cholesterol level',
                'glucose': 'glucose level',
                'weight': 'weight',
                'risk_percent': 'overall risk score',
                'smoker': 'smoking status',
                'alcohol': 'alcohol consumption',
                'activity': 'physical activity'
            }
            
            # Get top features and add explanations
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_features = sorted_features[:3]
            
            # Add explanation based on top features
            if top_features[0][1] > 50:  # If one feature dominates
                dominant_feature = top_features[0][0]
                dominant_name = feature_display_names.get(dominant_feature, dominant_feature)
                
                # Add specific feature-based explanation
                if trend == "worsening":
                    recs.append(f"Your {dominant_name} appears to be the main factor affecting your health trend.")
                    
                    # Add more specific advice based on the dominant feature
                    if dominant_feature == 'risk_percent':
                        recs.append("Your overall cardiovascular risk score is the key indicator to monitor.")
                    elif dominant_feature in ['systolic_bp', 'diastolic_bp']:
                        recs.append("Blood pressure management should be your priority for improvement.")
                    else:
                        recs.append(f"Your {dominant_name} should be carefully monitored.")
                else:
                    recs.append(f"Your {dominant_name} is significantly influencing your health trend.")
        
        print("Feature importance:", feature_importance)
        
    except Exception as e:
        print(f"Error in feature importance calculation: {e}")
        feature_importance = {f: 0 for f in features}  # Fallback empty importance

    if trend == "worsening":
        recs.append("⚠️ Your cardiovascular risk appears to be increasing. Please take action.")
    elif trend == "stable":
        recs.append("➖  Your risk appears stable. Keep maintaining your healthy habits and try to improve your risk score!")
    else:
        recs.append("✅ Your risk is improving. Keep maintaining your healthy habits!")

    bmi = latest["weight"]/(latest["height"]/100)**2

    if latest["systolic_bp"] > 140:
        recs.append("💧 Reduce salt and caffeine intake to manage your blood pressure.")
    if latest["cholesterol"] > 200:
        recs.append("🥬 Eat more fiber and reduce saturated fats.")
    if latest["glucose"] > 6.9:
        recs.append("🍭 Cut sugary foods and monitor your glucose levels closely.")
    if bmi > 25:
        recs.append("🏃‍♂️ Moderate exercise and portion control can help with weight.")
    if latest["smoker"].lower() == "yes":
        recs.append("🚭 Quitting smoking significantly reduces cardiovascular risk.")
    if latest["alcohol"].lower() == "yes":
        recs.append("🍷 Try to reduce alcohol consumption to lower liver and heart strain.")
    if latest["activity"].lower() == "not active":
        recs.append("🕺 Aim for at least 30 minutes of activity, 3–5 days a week.")

    return {
        "trend": trend,
        "confidence": confidence,
        "recommendations": recs,
        "prediction_vector": prediction_vector.tolist(),
        "feature_importance": feature_importance
    }

# ✅ Place this LAST
app.mount("/", StaticFiles(directory="static", html=True), name="static")

