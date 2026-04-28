from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from contextlib import asynccontextmanager
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import httpx
import os
 
IOT_BASE_URL = os.getenv("IOT_BASE_URL", "http://iot-cloud-service:8000")
 
scheduler = AsyncIOScheduler()
 
# last prediction storage 
latest_prediction: dict | None = None
 
 
# ml затычка
 
def run_model(measurements: list[dict]) -> dict:
    return {
        "prefTemperature": 21.0,
        "prefHumidity": 50.0,
        "prefLight": 300.0,
    }
 
 
# core logic
 
async def fetch_and_predict():
    global latest_prediction
    async with httpx.AsyncClient() as http:
        #get data from IOT
        resp = await http.get(f"{IOT_BASE_URL}/sensor-data/current")
        resp.raise_for_status()
        measurements = resp.json()
 
        # dummy model prediction
        scenario = run_model(measurements)
        latest_prediction = {
            "scenario": scenario,
            "createdAt": datetime.utcnow().isoformat(),
        }
 
        # post mock action 
        await http.post(f"{IOT_BASE_URL}/actuator/action", json=scenario)
 
 
#sheduler setup 
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.add_job(fetch_and_predict, "interval", minutes=15, id="predict_job")
    scheduler.start()
    yield
    scheduler.shutdown()
 
 
app = FastAPI(title="MAL Service API", version="1.0.0", lifespan=lifespan)
 
 
#dummy model
 
class ScenarioOut(BaseModel):
    prefTemperature: float
    prefHumidity: float
    prefLight: float
    createdAt: str
 
 
class FeedbackIn(BaseModel):
    value: bool
 
 
#endpoints for IM
 
@app.get("/scenario", response_model=ScenarioOut,
         summary="IM last predicted scenarion")
async def get_scenario():
    #return last prediction to IM
    if not latest_prediction:
        raise HTTPException(404, "No predictions yet")
    return ScenarioOut(**latest_prediction["scenario"], createdAt=latest_prediction["createdAt"])
 
 
@app.post("/feedback", status_code=201,
          summary="IM sends feedback on prediction (used for model retraining)")
async def post_feedback(payload: FeedbackIn):
    #pass feedback to model  when model is ready
    return {"value": payload.value, "savedAt": datetime.utcnow().isoformat()}