import io
# import json
import numpy as np

from PIL import Image



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

# def safe_json_serialize(obj):
#     """Безопасная сериализация в JSON с конвертацией numpy типов"""
#     return json.loads(json.dumps(obj, default=lambda x: convert_numpy_types(x)))
