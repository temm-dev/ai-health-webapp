from pathlib import Path

import numpy as np
import torch
import torchvision.models as models
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATH_MODELS = (PROJECT_ROOT / "app" / "models").as_posix() + "/"
# PATH_TEST_IMAGES = (PROJECT_ROOT / "data" / "test_images").as_posix() + "/"


torch.serialization.add_safe_globals([np.core.multiarray.scalar])  # type: ignore


class SkinTestAnalysis:
    def __init__(self) -> None:
        try:
            model_path = PATH_MODELS + "skin_model.pth"
            model = torch.load(model_path, map_location="cpu", weights_only=False)

            self.model = models.vit_b_16(weights=None)
            self.model.heads[0] = torch.nn.Linear(in_features=768, out_features=3)
            self.model.load_state_dict(model["model"])  # type: ignore
            self.model.eval()

            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            print("✅ Модель skin_model.pth успешно загружена!")
        except Exception as e:
            print(
                f"ERROR: Class SkinTestAnalysis(torch) - {self.__init__.__name__}:\n{e}"
            )

    async def normalize_probabilities(self, probabilities, decimals=5):
        try:
            probs = np.array(probabilities)

            if abs(probs.sum() - 1.0) > 0.01:
                probs = probs / probs.sum()

            normalized = [round(float(p), decimals) for p in probs]

            return normalized
        except Exception as e:
            print(
                f"ERROR: Class SkinTestAnalysis(torch) - {self.normalize_probabilities.__name__}:\n{e}"
            )

    async def analyze(self, image_path: str):
        path = f"data/temp/{image_path}"

        image = Image.open(path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0)  # type: ignore

        with torch.no_grad():
            try:
                output = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)

                # Преобразуем tensor в numpy array и нормализуем
                probs_array = probabilities.numpy()
                normalized_probs = await self.normalize_probabilities(probs_array)
            except Exception as e:
                print(
                    f"ERROR: Class SkinTestAnalysis(torch) - {self.analyze.__name__}:\n{e}"
                )

        result_dict = {
            "acne": normalized_probs[0],  # type: ignore
            "redness": normalized_probs[1],  # type: ignore
            "eyebags": normalized_probs[2],  # type: ignore
        }

        return result_dict
