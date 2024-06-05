from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError, Field
import joblib
import numpy as np

# Load the model
model = joblib.load('best_model.joblib')
scaler = joblib.load('scaler.joblib')
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.mount("/static", StaticFiles(directory="./static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("./static/index.html")

# Define the input data model with validation
class InputData(BaseModel):
    attempts: float = Field(..., ge=0)  # Ensure attempts is >= -1
    score_rate: float = Field(..., ge=-1, le=1)  # Ensure score_rate is between 0 and 1
    teacher_rank: float = Field(..., ge=1, le=5)  # Ensure teacher_rank is between 1 and 5
    school_rank: float = Field(..., ge=1, le=5)  # Ensure school_rank is between 1 and 5
    users_school_rank: float = Field(..., ge=1, le=5)  # Ensure users_school_rank is between 1 and 5
    sentiment: float = Field(..., ge=0, le=5)  # Ensure sentiment is between 1 and 5
    num_of_user: float = Field(..., ge=0)  # Ensure num_of_user is >= 0

# Define mapping for prediction labels
prediction_labels = {
    1: "Tệ",
    2: "Trung bình",
    3: "Tốt"
}

@app.post("/predict/")
async def predict(data: InputData):
    try:
        print("Received data:", data)

        # Extract features from the input data
        features = np.array([
            data.attempts, data.score_rate, data.teacher_rank,
            data.school_rank, data.users_school_rank, data.sentiment,
            data.num_of_user
        ]).reshape(1, -1)

        # Scale the features
        features_scaled = scaler.transform(features)
        
        # Log scaled data
        print("Scaled input data:", features_scaled)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        print("Prediction:", prediction)

        # Map prediction to label
        prediction_label = prediction_labels.get(prediction, "Không xác định")
        
        # Log prediction
        print("Prediction:", prediction_label)
        
        # Return the prediction label
        return JSONResponse(content={"prediction": prediction_label})
    except ValidationError as e:
        print("Validation Error:", e.json())
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        print("Internal Server Error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))