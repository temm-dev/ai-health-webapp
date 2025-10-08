import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from PIL import Image
import io

# Импортируем наши сервисы (создадим их дальше)
from app.services.face_analysis import FaceAnalysis

router = APIRouter(tags=["upload"])

# Директория для сохранения фото
UPLOAD_DIR = "data/temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)

face_analysis = FaceAnalysis()


def compress_image(image_data, max_size=(800, 800), quality=85):
    """"""
    try:
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')
        
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format='JPEG', quality=quality, optimize=True)
        
        return output.getvalue()
        
    except Exception as e:
        raise Exception(f"Image compression error: {str(e)}")

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"): # type: ignore
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Генерация уникального имя файла
        file_extension = os.path.splitext(file.filename)[1] # type: ignore
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)

        content = await file.read()

        compressed_content = compress_image(
            content, 
            max_size=(1200, 1200),  # Максимальный размер
            quality=75              # Качество 75%
        )
        
        # Сохранение файл
        with open(file_path, "wb") as buffer:
            buffer.write(compressed_content)
        
        processed_data = await face_analysis.analyze(unique_filename)
        
        response = {
            "status": "success",
            "filename": unique_filename,
            "analysis": processed_data
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")