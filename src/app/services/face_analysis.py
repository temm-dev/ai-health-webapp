import os
import time
from typing import Any, Dict, List

from deepface import DeepFace

class FaceAnalysis:

    async def base_analyze(self, image_path: str
    ) -> Dict[str, Any] | List[Dict[str, Any]] | None:
        """"""
        path = f"data/temp/{image_path}"

        if os.path.exists(path):
            t1 = time.time()

            objs = DeepFace.analyze(
                img_path=path, actions=['age', 'gender', 'race', 'emotion']
            )

            objs[0]["time"] = time.time() - t1 # type: ignore

            return objs[0]

        return None
    
    async def anti_spoofing_test(self, image_path: str):
        path = f"data/temp/{image_path}"

        if os.path.exists(path):
            t1 = time.time()
            face_objs = DeepFace.extract_faces(img_path=path, anti_spoofing = True)

            objs = {
                "confidence": face_objs[0]["confidence"],
                "is_real": face_objs[0]["is_real"],
                "antispoof_score": face_objs[0]["antispoof_score"]
            }

            objs["time"] = time.time() - t1 # type: ignore

            return objs


