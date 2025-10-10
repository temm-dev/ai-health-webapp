import io
import json
import os
import uuid
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from app.core.schemas import UploadResponse
from app.services.calculate_metrics import CalculateMetrics
from app.services.face_analysis import FaceAnalysis
from app.services.face_metrics import FaceMetricsAnalysis
from app.services.skin_analysis import SkinTestAnalysis
from app.langgraf.langgraf_logic import analyze_from_json

router = APIRouter(tags=["upload"])

# Директория для сохранения фото
UPLOAD_DIR = "data/temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)

face_analysis = FaceAnalysis()
skin_test = SkinTestAnalysis()
face_metrics_analysis = FaceMetricsAnalysis()
calculate_metrics = CalculateMetrics()


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


def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(element) for element in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(element) for element in obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):  # type: ignore
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):  # type: ignore
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def safe_json_serialize(obj):
    """Безопасная сериализация в JSON с конвертацией numpy типов"""
    return json.loads(json.dumps(obj, default=lambda x: convert_numpy_types(x)))


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

        try:
            face_analysis_data = await face_analysis.analyze(
                unique_filename, analysis_actions
            )
            face_analysis_data = convert_numpy_types(face_analysis_data)
            return_response.update(face_analysis_data)  # type: ignore
        except:
            pass

        try:
            skin_test_data = await skin_test.analyze(unique_filename)
            skin_test_data = convert_numpy_types(skin_test_data)
            return_response.update(skin_test_data)  # type: ignore
        except:
            pass

        try:
            metrics_face_data = face_metrics_analysis.analyze(unique_filename)
            metrics_face_data = convert_numpy_types(metrics_face_data)

            age = face_analysis_data["age"]  # type: ignore
            emotions = return_response["emotion"]
            eyebags = skin_test_data["eyebags"]  # type: ignore
            average_eye_openness = metrics_face_data["average_eye_openness"]  # type: ignore
            muscle_tension = metrics_face_data["muscle_tension"]  # type: ignore
            redness = skin_test_data["redness"]  # type: ignore
            acne = skin_test_data["acne"]  # type: ignore
            facial_symmetry = metrics_face_data["facial_symmetry"] # type: ignore

            stress = calculate_metrics.calculate_stress_level(
                emotions, eyebags, muscle_tension  # type: ignore
            )
            sleep_quality = calculate_metrics.calculate_sleep_quality(
                eyebags, redness, average_eye_openness, age  # type: ignore
            )
            calculate_skin_health_index = calculate_metrics.calculate_skin_health_index(
                acne, redness, age  # type: ignore
            )
            vitality_score = calculate_metrics.calculate_vitality_score(
                {
                    "stress_level": stress,
                    "sleep_quality": sleep_quality,
                    "skin_health_index": calculate_skin_health_index,
                    "facial_symmetry": facial_symmetry,
                    "muscle_tension": muscle_tension
                }  # type: ignore
            )

            stress = {"stress_level": stress}
            sleep_quality = {"sleep_quality": sleep_quality}
            skin_health_index = {"skin_health_index": calculate_skin_health_index}
            vitality_score = {"vitality_score": vitality_score}

            return_response.update(metrics_face_data)  # type: ignore
            return_response.update(stress)
            return_response.update(sleep_quality)
            return_response.update(skin_health_index)
            return_response.update(vitality_score)
        except Exception as e:
            print(e)

        if anti_spoofing_test:
            try:
                anti_spoofing_test_data = await face_analysis.anti_spoofing_test(
                    unique_filename
                )
                return_response.update(anti_spoofing_test_data)  # type: ignore
            except:
                pass

        
        print(return_response)
        
        llmresponse = await analyze_from_json(return_response)

        print(llmresponse)

        return JSONResponse(content=llmresponse)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
