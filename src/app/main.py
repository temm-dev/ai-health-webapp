from fastapi import FastAPI
from app.api.endpoints import health, upload

app = FastAPI(title="Health Face Analysis API")

app.include_router(health.router, prefix="/api/v1")
app.include_router(upload.router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Health Face Analysis API is running!"}