import os
import warnings

# Suppress warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
from backend.routes import router
from utils.logger import logger

app = FastAPI(title="Thyroid Cancer Detection API")

# Mount static files
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# Include Routes
app.include_router(router)

if __name__ == "__main__":
    logger.info("Starting Thyroid Cancer Detection API...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)