import io
import os
import uuid
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from app.models.schemas import UploadResponse
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

        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")

        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=quality, optimize=True)

        return output.getvalue()

    except Exception as e:
        raise Exception(f"Image compression error: {str(e)}")


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

        compressed_content = compress_image(
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

        return_response = {}

        analyze_data = await face_analysis.analyze(unique_filename, analysis_actions)

        return_response.update(analyze_data)  # type: ignore

        if anti_spoofing_test:
            anti_spoofing_test_data = await face_analysis.anti_spoofing_test(
                unique_filename
            )

            return_response.update(anti_spoofing_test_data)  # type: ignore

        response = {
            "status": "success",
            "filename": unique_filename,
            "analyze": return_response,
        }

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
