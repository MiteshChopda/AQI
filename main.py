
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
try:
    from aqi import pollutants_mean, linearRegressionModel, RandomForestRegressorModel, DecisionTreeRegressorModel, AQI_Range
except Exception as e:
    from .aqi import pollutants_mean, linearRegressionModel, RandomForestRegressorModel, DecisionTreeRegressorModel, AQI_Range
from  pandas import DataFrame

# Initialize FastAPI app
app = FastAPI(
    title="AQI Predictor API",
    description="API for predicting Air Quality Index based on pollutant values",
    version="1.0.0"
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PollutantData(BaseModel):
    SO2i: Optional[float] = None
    Noi: Optional[float] = None
    RSPMi: Optional[float] = None
    SPMi: Optional[float] = None
    PM2_5i: Optional[float] = None
    model: Optional[str] = None

class AQIResponse(BaseModel):
    AQI: float
    AQI_range: str

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "AQI Predictor API",
        "status": "running"
    }

# Main prediction endpoint
@app.post("/predict", response_model=AQIResponse)
async def predict_aqi(data: PollutantData):
    try:
        print(f"\n\n\nrequest from frontend\n\n{data}\n\n\n")
        if data.Noi == None:
            data.Noi = pollutants_mean["Noi"]
        if data.PM2_5i == None:
            data.PM2_5i = pollutants_mean["PM2_5i"] 
        if data.RSPMi == None:
            data.RSPMi = pollutants_mean["RSPMi"] 
        if data.SO2i == None:
            data.SO2i = pollutants_mean["SO2i"] 
        if data.SPMi == None:
            data.SPMi = pollutants_mean["SPMi"]
            
        if data.model == "linearRegressionModel":
            dataframe = {
                'SO2i':data.SO2i,
                'Noi' :data.Noi,
                'RSPMi' :data.RSPMi,
                'SPMi' :data.SPMi,
                "PM2_5i":  data.PM2_5i
            }
            X = DataFrame([dataframe])
            prediction = linearRegressionModel.predict(X)
            result = AQIResponse(
                AQI=prediction,
                AQI_range=AQI_Range(prediction)
            )
            print("Linear:",result)
            return result
        if data.model == "DecisionTreeRegressorModel":
            dataframe = {
                'SO2i':data.SO2i,
                'Noi' :data.Noi,
                'RSPMi' :data.RSPMi,
                'SPMi' :data.SPMi,
                "PM2_5i":  data.PM2_5i
            }
            X = DataFrame([dataframe])
            prediction = DecisionTreeRegressorModel.predict(X)
            result = AQIResponse(
                AQI=prediction,
                AQI_range=AQI_Range(prediction)
            )
            print("DecisionTreeRegressor:",result)
            return result
        if data.model == "RandomForestRegressorModel":
            dataframe = {
                'SO2i':data.SO2i,
                'Noi' :data.Noi,
                'RSPMi' :data.RSPMi,
                'SPMi' :data.SPMi,
                "PM2_5i":  data.PM2_5i
            }
            X = DataFrame([dataframe])
            prediction = RandomForestRegressorModel.predict(X)
            result = AQIResponse(
                AQI=prediction,
                AQI_range=AQI_Range(prediction)
            )
            print("RandomForestRegressor:",result)
            return result
        
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=500,
            detail=f"Error in AQI prediction: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )