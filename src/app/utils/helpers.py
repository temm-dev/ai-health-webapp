import io

import numpy as np
from PIL import Image

from app.services.calculate_metrics import CalculateMetrics
from app.services.face_analysis import FaceAnalysis
from app.services.face_metrics import FaceMetricsAnalysis
from app.services.skin_analysis import SkinTestAnalysis
from app.utils.helpers import convert_numpy_types

face_analysis = FaceAnalysis()
skin_test = SkinTestAnalysis()
face_metrics_analysis = FaceMetricsAnalysis()
calculate_metrics = CalculateMetrics()


async def compress_image(image_data, max_size=(800, 800), quality=85):
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


async def convert_numpy_types(obj) -> dict | list | tuple | float | bool:
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


async def image_processing(
    filename: str, analysis_actions: tuple, anti_spoofing_test: bool
):
    response = {}

    face_data = await face_analysis.analyze(filename, analysis_actions)
    skin_data = await skin_test.analyze(filename)
    metrics_data = await face_metrics_analysis.analyze(filename)

    face_data = await convert_numpy_types(face_data)
    skin_data = await convert_numpy_types(skin_data)
    metrics_data = await convert_numpy_types(metrics_data)

    response.update(face_data)  # type: ignore
    response.update(skin_data)  # type: ignore

    age = face_data["age"]  # type: ignore
    emotions = response["emotion"]
    eyebags = skin_test_data["eyebags"]  # type: ignore
    average_eye_openness = metrics_face_data["average_eye_openness"]  # type: ignore
    muscle_tension = metrics_face_data["muscle_tension"]  # type: ignore
    redness = skin_test_data["redness"]  # type: ignore
    acne = skin_test_data["acne"]  # type: ignore
    facial_symmetry = metrics_face_data["facial_symmetry"]  # type: ignore

    stress = await calculate_metrics.calculate_stress_level(
        emotions, eyebags, muscle_tension
    )
    sleep_quality = await calculate_metrics.calculate_sleep_quality(
        eyebags, redness, average_eye_openness, age
    )
    skin_health_index = await calculate_metrics.calculate_skin_health_index(
        acne, redness, age
    )
    vitality_score = await calculate_metrics.calculate_vitality_score(
        {
            "stress_level": stress,
            "sleep_quality": sleep_quality,
            "skin_health_index": skin_health_index,
            "facial_symmetry": facial_symmetry,
            "muscle_tension": muscle_tension,
        }
    )

    stress = {"stress_level": stress}
    sleep_quality = {"sleep_quality": sleep_quality}
    skin_health_index = {"skin_health_index": skin_health_index}
    vitality_score = {"vitality_score": vitality_score}

    response.update(metrics_face_data)  # type: ignore
    response.update(stress)
    response.update(sleep_quality)
    response.update(skin_health_index)
    response.update(vitality_score)

    if anti_spoofing_test:
        anti_spoofing_test_data = await face_analysis.anti_spoofing_test(filename)
        response.update(anti_spoofing_test_data)  # type: ignore

    return response
