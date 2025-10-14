import os
import uuid
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from app.core.schemas import UploadResponse
from app.langgraf.langgraf_logic import analyze_from_json
from app.utils.utils import compress_image, image_processing

router = APIRouter(tags=["upload"])

# Директория для сохранения фото
UPLOAD_DIR = "data/temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    actions: Optional[List[str]] = Query(None),
    anti_spoofing_test: Optional[int] = 0,
):
    try:
        if not file.content_type.startswith("image/"):  # type: ignore
            raise HTTPException(status_code=400, detail="File must be an image")

        # Генерация уникального имя файла
        file_extension = os.path.splitext(file.filename)[1]  # type: ignore
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)

        content = await file.read()

        compressed_content = await compress_image(
            content,
            max_size=(1200, 1200),  # Максимальный размер
            quality=75,  # Качество 75%
        )

        # Сохранение файл
        with open(file_path, "wb") as buffer:
            buffer.write(compressed_content)

        if actions and "all" in actions:
            analysis_actions = ("age", "gender", "race", "emotion")
        elif actions:
            analysis_actions = tuple(actions)
        else:
            analysis_actions = ("age", "gender", "race", "emotion")

        response = await image_processing(
            unique_filename, analysis_actions, bool(anti_spoofing_test)
        )
        print(response)

        llmresponse = await analyze_from_json(response)

        print(llmresponse)

        return JSONResponse(content=llmresponse)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
