import os
import time

from deepface import DeepFace


class FaceAnalysis:

    async def analyze(
        self, image_path: str, actions=("age", "gender", "race", "emotion")
    ):
        path = f"data/temp/{image_path}"

        if os.path.exists(path):
            t1 = time.time()

            objs = DeepFace.analyze(img_path=path, actions=actions)

            objs[0]["time"] = time.time() - t1  # type: ignore

            return objs[0]

        return {}

    async def anti_spoofing_test(self, image_path: str):
        path = f"data/temp/{image_path}"

        if os.path.exists(path):
            t1 = time.time()
            face_objs = DeepFace.extract_faces(img_path=path, anti_spoofing=True)

            objs = {
                "antispoof_score": face_objs[0]["antispoof_score"],
                "is_real": face_objs[0]["is_real"],
            }

            objs["time"] = time.time() - t1  # type: ignore

            return objs
