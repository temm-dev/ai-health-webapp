from fastapi import APIRouter, UploadFile, File

router = APIRouter(tags=["upload"])

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    return {"message": f"Файл {file.filename} получен!", "status": "success"}