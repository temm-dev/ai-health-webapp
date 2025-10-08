from fastapi import FastAPI

from app.api.endpoints import upload

app = FastAPI(title="Health Face Analysis API")

app.include_router(upload.router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"message": "Health Face Analysis API is running!"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
