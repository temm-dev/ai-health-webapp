import numpy as np
from typing import Dict, Optional

class CalculateMetrics:
    def __init__(self):
        # Оптимизированные константы
        self.STRESS_WEIGHTS = {
            "emotion": 0.35,
            "eye_bags": 0.30, 
            "facial_tension": 0.25,
            "heart_rate": 0.10
        }
        
        self.SLEEP_WEIGHTS = {
            "eye_bags": 0.40,
            "redness": 0.30,
            "eye_openness": 0.20,
            "pupil_reaction": 0.10
        }
        
        self.SKIN_WEIGHTS = {
            "acne": 0.40,
            "redness": 0.30,
            "smoothness": 0.20,
            "wrinkles": 0.10
        }

    async def calculate_stress_level(
        self, 
        emotion_data: Dict[str, float], 
        eye_bags_score: float, 
        facial_tension: float = 0.5,
        heart_rate_variability: Optional[float] = None
    ) -> float:
        """
        Оптимизированный расчет уровня стресса
        """
        # Быстрая нормализация
        emotion_data = self._normalize_emotions(emotion_data)
        eye_bags_score = np.clip(eye_bags_score, 0, 1)
        facial_tension = np.clip(facial_tension, 0, 1)
        
        # Оптимизированный эмоциональный анализ
        emotion_stress = 0
        for emotion, confidence in emotion_data.items():
            if emotion in ["angry", "fear", "sad", "disgust"]:
                emotion_stress += confidence * 1.0
            elif emotion in ["neutral"]:
                emotion_stress += confidence * 0.5
            elif emotion in ["happy", "surprise"]:
                emotion_stress += confidence * 0.2
        
        emotion_stress = min(1.0, emotion_stress)
        
        # Эффективный расчет с использованием numpy
        components = np.array([
            emotion_stress,
            eye_bags_score, 
            facial_tension,
            heart_rate_variability if heart_rate_variability else 0.5
        ])
        
        weights = np.array([
            self.STRESS_WEIGHTS["emotion"],
            self.STRESS_WEIGHTS["eye_bags"],
            self.STRESS_WEIGHTS["facial_tension"],
            self.STRESS_WEIGHTS["heart_rate"]
        ])
        
        stress_level = np.dot(components, weights)
        return float(np.clip(stress_level, 0, 1))

    async def calculate_sleep_quality(
        self, 
        eye_bags_score: float, 
        redness_score: float, 
        eye_openness: float = 0.7, 
        age: int = 30,
        pupil_reaction: Optional[float] = None
    ) -> float:
        """
        Улучшенный расчет качества сна
        """
        # Валидация входных данных
        eye_bags_score = np.clip(eye_bags_score, 0, 1)
        redness_score = np.clip(redness_score, 0, 1)
        eye_openness = np.clip(eye_openness, 0, 1)
        age = np.clip(age, 0, 120)
        
        # Оптимизированный возрастной фактор
        age_factor = min(1.0, (age / 50) ** 0.8)  # Нелинейная зависимость
        
        # Расчет компонентов
        eye_bags_impact = eye_bags_score * (0.8 + 0.2 * age_factor)
        redness_impact = redness_score * (0.7 + 0.3 * age_factor)
        eye_openness_impact = (1 - eye_openness) * 0.8
        pupil_impact = pupil_reaction if pupil_reaction else 0.5
        
        # Векторный расчет
        impacts = np.array([
            eye_bags_impact,
            redness_impact, 
            eye_openness_impact,
            pupil_impact
        ])
        
        weights = np.array([
            self.SLEEP_WEIGHTS["eye_bags"],
            self.SLEEP_WEIGHTS["redness"],
            self.SLEEP_WEIGHTS["eye_openness"],
            self.SLEEP_WEIGHTS["pupil_reaction"]
        ])
        
        sleep_quality = 1 - np.dot(impacts, weights)
        return float(np.clip(sleep_quality, 0, 1))

    async def calculate_skin_health_index(
        self, 
        acne_score: float, 
        redness_score: float, 
        age: int,
        skin_smoothness: float = 0.7,
        wrinkles_score: Optional[float] = None
    ) -> float:
        """
        Улучшенный расчет индекса здоровья кожи
        """
        # Быстрая валидация
        acne_score = np.clip(acne_score, 0, 1)
        redness_score = np.clip(redness_score, 0, 1)
        skin_smoothness = np.clip(skin_smoothness, 0, 1)
        age = np.clip(age, 0, 120)
        
        # Оптимизированный возрастной фактор
        if age <= 25:
            age_factor = 1.0
        elif age <= 35:
            age_factor = 0.9
        elif age <= 45:
            age_factor = 0.8
        elif age <= 55:
            age_factor = 0.7
        else:
            age_factor = 0.6
        
        # Расчет компонентов
        acne_impact = (1 - acne_score) * self.SKIN_WEIGHTS["acne"]
        redness_impact = (1 - redness_score) * self.SKIN_WEIGHTS["redness"]
        smoothness_impact = skin_smoothness * self.SKIN_WEIGHTS["smoothness"]
        wrinkles_impact = (1 - (wrinkles_score if wrinkles_score else 0.3)) * self.SKIN_WEIGHTS["wrinkles"]
        
        skin_health_index = (
            acne_impact + 
            redness_impact + 
            smoothness_impact + 
            wrinkles_impact +
            age_factor * 0.1  # Возрастной бонус
        )
        
        return float(np.clip(skin_health_index, 0, 1))

    async def calculate_vitality_score(
        self, 
        all_metrics: Dict[str, float], 
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Оптимизированный расчет показателя жизненной силы
        """
        if weights is None:
            weights = {
                "stress_level": 0.25,
                "sleep_quality": 0.25,
                "skin_health_index": 0.20,
                "facial_symmetry": 0.15,
                "muscle_tension": 0.15,
            }
        
        vitality = 0.0
        total_weight = 0.0
        
        # Быстрый итератор по метрикам
        for metric, weight in weights.items():
            if metric in all_metrics:
                value = all_metrics[metric]
                
                # Инвертируем негативные метрики
                if metric in ["stress_level", "muscle_tension"]:
                    vitality += (1 - value) * weight
                else:
                    vitality += value * weight
                    
                total_weight += weight
        
        # Эффективная нормализация
        if total_weight > 0:
            vitality /= total_weight
        
        return float(np.clip(vitality, 0, 1))

    def _normalize_emotions(self, emotion_data: Dict[str, float]) -> Dict[str, float]:
        """Быстрая нормализация эмоций"""
        total = sum(emotion_data.values())
        if total == 0:
            return emotion_data
        return {k: v / total for k, v in emotion_data.items()}

    async def calculate_comprehensive_health_score(
        self,
        emotion_data: Dict[str, float],
        eye_bags_score: float,
        facial_tension: float,
        redness_score: float,
        eye_openness: float,
        age: int,
        acne_score: float,
        skin_smoothness: float = 0.7
    ) -> Dict[str, float]:
        """
        Комплексный расчет всех метрик здоровья за один вызов для оптимизации
        """
        # Параллельные вычисления (в рамках одного процесса)
        stress_level = await self.calculate_stress_level(emotion_data, eye_bags_score, facial_tension)
        sleep_quality = await self.calculate_sleep_quality(eye_bags_score, redness_score, eye_openness, age)
        skin_health = await self.calculate_skin_health_index(acne_score, redness_score, age, skin_smoothness)
        
        # Сбор всех метрик для vitality score
        all_metrics = {
            "stress_level": stress_level,
            "sleep_quality": sleep_quality,
            "skin_health_index": skin_health,
            "facial_symmetry": 0.7,  # Должно приходить извне
            "muscle_tension": facial_tension
        }
        
        vitality_score = await self.calculate_vitality_score(all_metrics)
        
        return {
            "stress_level": stress_level,
            "sleep_quality": sleep_quality,
            "skin_health_index": skin_health,
            "vitality_score": vitality_score,
            "overall_health": (stress_level + sleep_quality + skin_health + vitality_score) / 4
        }