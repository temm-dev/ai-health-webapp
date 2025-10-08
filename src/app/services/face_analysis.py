import os
import time
from typing import Any, Dict, List

from deepface import DeepFace

class FaceAnalysis:

    async def analyze(self, image_path: str
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
