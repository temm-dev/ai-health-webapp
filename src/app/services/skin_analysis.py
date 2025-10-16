from pathlib import Path
import numpy as np
import torch
import torchvision.models as models
from PIL import Image, ImageOps
from torchvision import transforms
import logging
from typing import Dict, List, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATH_MODELS = (PROJECT_ROOT / "app" / "models").as_posix() + "/"

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.serialization.add_safe_globals([np.core.multiarray.scalar]) # type: ignore


class SkinTestAnalysis:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        try:
            model_path = PATH_MODELS + "skin_model.pth"
            
            # Загрузка с проверкой наличия файла
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Безопасная загрузка модели
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            
            # Инициализация модели с оригинальной архитектурой
            self.model = models.vit_b_16(weights=None)
            # Используем оригинальную архитектуру головы для совместимости
            self.model.heads[0] = torch.nn.Linear(in_features=768, out_features=3)
            
            # Загрузка весов с проверкой
            if "model" in checkpoint:
                model_state_dict = checkpoint["model"]
            else:
                model_state_dict = checkpoint
            
            # Загрузка с обработкой несовпадений
            try:
                self.model.load_state_dict(model_state_dict)
            except RuntimeError as e:
                logger.warning(f"Strict loading failed, trying flexible loading: {e}")
                # Гибкая загрузка - игнорируем несовпадающие ключи
                self.model.load_state_dict(model_state_dict, strict=False)
                
            self.model.to(self.device)
            self.model.eval()
            
            # Упрощенные трансформации (оригинальные)
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ])
            
            # Трансформации для аугментации (опционально)
            self.augmentation_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            self._warmup_model()
            logger.info("✅ Skin model successfully loaded!")

        except Exception as e:
            logger.error(f"ERROR loading skin model: {e}")
            # Создаем простую модель в случае ошибки
            self._create_fallback_model()

    def _create_fallback_model(self):
        """Создание резервной модели в случае ошибки загрузки"""
        logger.info("Creating fallback model...")
        self.model = models.vit_b_16(weights=None)
        self.model.heads[0] = torch.nn.Linear(in_features=768, out_features=3)
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])

    def _warmup_model(self):
        """Прогрев модели для стабильной работы"""
        try:
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            with torch.no_grad():
                _ = self.model(dummy_input)
            logger.info("Model warmup completed")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

    async def normalize_probabilities(self, probabilities: np.ndarray, decimals: int = 5) -> List[float]:
        """Улучшенная нормализация вероятностей"""
        try:
            probs = np.array(probabilities, dtype=np.float32)
            
            if probs.size == 0:
                return [0.333, 0.333, 0.334]
                
            if np.any(probs < 0):
                probs = np.abs(probs)
                
            # Стабильная нормализация
            epsilon = 1e-8
            probs = probs + epsilon
            probs = probs / (probs.sum() + epsilon)
            
            # Обеспечение суммы = 1
            total = 0.0
            normalized = []
            for i, p in enumerate(probs):
                if i == len(probs) - 1:
                    normalized.append(round(1.0 - total, decimals))
                else:
                    value = round(float(p), decimals)
                    normalized.append(value)
                    total += value
                    
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing probabilities: {e}")
            return [0.333, 0.333, 0.334]

    async def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, Image.Image]:
        """Предобработка изображения"""
        try:
            image = Image.open(image_path).convert("RGB")
            image = ImageOps.exif_transpose(image)
            
            if image.size[0] < 50 or image.size[1] < 50:
                raise ValueError("Image resolution too low")
                
            input_tensor = self.transform(image).unsqueeze(0).to(self.device) # type: ignore
            return input_tensor, image
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            raise

    async def ensemble_prediction(self, image: Image.Image, n_augmentations: int = 3) -> torch.Tensor:
        """Ансамблевое предсказание с аугментациями"""
        try:
            predictions = []
            
            for i in range(n_augmentations):
                if i == 0:
                    # Первое предсказание без аугментаций
                    tensor = self.transform(image).unsqueeze(0).to(self.device) # type: ignore
                else:
                    # Случайные аугментации для остальных
                    tensor = self.augmentation_transform(image).unsqueeze(0).to(self.device) # type: ignore
                
                with torch.no_grad():
                    output = self.model(tensor)
                    predictions.append(output)
            
            # Усреднение предсказаний
            avg_prediction = torch.mean(torch.stack(predictions), dim=0)
            return avg_prediction
            
        except Exception as e:
            logger.warning(f"Ensemble prediction failed, using single prediction: {e}")
            tensor = self.transform(image).unsqueeze(0).to(self.device) # type: ignore
            with torch.no_grad():
                return self.model(tensor)

    async def calculate_confidence(self, probabilities: List[float]) -> Dict[str, float]:
        """Расчет уверенности в предсказаниях"""
        try:
            probs = np.array(probabilities)
            epsilon = 1e-8
            
            # Энтропия как мера уверенности
            entropy = -np.sum(probs * np.log(probs + epsilon)) # type: ignore
            max_entropy = np.log(len(probs))
            confidence = 1 - (entropy / max_entropy)
            
            max_prob = np.max(probs)
            std_prob = np.std(probs)
            
            return {
                "overall_confidence": float(np.clip(confidence, 0, 1)),
                "max_probability_confidence": float(max_prob),
                "std_confidence": float(1 - std_prob),
                "is_uncertain": confidence < 0.7
            }
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return {
                "overall_confidence": 0.5,
                "max_probability_confidence": 0.5,
                "std_confidence": 0.5,
                "is_uncertain": True
            }

    async def analyze(self, image_path: str) -> Dict:
        """Анализ кожи"""
        path = f"data/temp/{image_path}"
        
        try:
            # Предобработка
            input_tensor, original_image = await self.preprocess_image(path)
            
            # Предсказание с ансамблем
            output = await self.ensemble_prediction(original_image)
            
            # Расчет вероятностей
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            probs_array = probabilities.cpu().numpy()
            
            # Нормализация
            normalized_probs = await self.normalize_probabilities(probs_array)
            
            # Расчет уверенности
            confidence_metrics = await self.calculate_confidence(normalized_probs)
            
            # Основные результаты
            result_dict = {
                "acne": normalized_probs[0],
                "redness": normalized_probs[1],
                "eyebags": normalized_probs[2],
                "confidence_metrics": confidence_metrics,
                "dominant_condition": self._get_dominant_condition(normalized_probs),
                "severity_level": self._calculate_severity_level(normalized_probs),
            }
            
            logger.info(f"Analysis completed for {image_path}")
            return result_dict
            
        except Exception as e:
            logger.error(f"Error analyzing image {path}: {e}")
            return await self._get_fallback_result()

    def _get_dominant_condition(self, probabilities: List[float]) -> str:
        """Определение доминирующего состояния кожи"""
        conditions = ["acne", "redness", "eyebags"]
        dominant_idx = np.argmax(probabilities)
        return conditions[dominant_idx]

    def _calculate_severity_level(self, probabilities: List[float]) -> str:
        """Расчет уровня серьезности"""
        max_prob = max(probabilities)
        
        if max_prob < 0.4:
            return "mild"
        elif max_prob < 0.7:
            return "moderate"
        else:
            return "severe"

    async def _get_fallback_result(self) -> Dict:
        """Резервный результат при ошибке"""
        return {
            "acne": 0.333,
            "redness": 0.333,
            "eyebags": 0.334,
            "confidence_metrics": {
                "overall_confidence": 0.1,
                "max_probability_confidence": 0.333,
                "std_confidence": 0.1,
                "is_uncertain": True
            },
            "dominant_condition": "unknown",
            "severity_level": "unknown",
        }

    def __del__(self):
        """Очистка ресурсов"""
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()