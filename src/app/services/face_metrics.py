from math import sqrt, dist
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATH_MODELS = (PROJECT_ROOT / "app" / "models").as_posix() + "/"


class FaceMetricsAnalysis:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh # type: ignore
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,  # Повышена точность детекции
            min_tracking_confidence=0.6,
        )
        
        # Оптимизированные наборы точек
        self.SYMMETRIC_PAIRS = [
            (162, 389), (234, 454), (130, 359), (93, 323), (58, 288),
            (67, 297), (109, 338), (151, 377), (33, 263), (133, 362),
            (362, 133), (374, 386), (145, 159)  # Добавлены ключевые точки
        ]
        
        # Точки для глаз (EAR - Eye Aspect Ratio)
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        
        # Кэш для нормализации
        self._normalization_cache = {}

    async def analyze(self, image_path: str) -> Optional[Dict]:
        """Оптимизированный анализ всех метрик лица"""
        path = f"data/temp/{image_path}"

        try:
            # Загрузка с оптимизацией
            image = cv2.imread(path)
            if image is None:
                return None

            # Уменьшение размера для скорости (сохраняя пропорции)
            height, width = image.shape[:2]
            if max(height, width) > 800:
                scale = 800 / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
                height, width = new_height, new_width

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)

            if not results.multi_face_landmarks:
                return None

            landmarks = results.multi_face_landmarks[0]
            
            # Параллельные вычисления
            symmetry_task = self._calculate_facial_symmetry(landmarks, width, height)
            eyes_task = self._calculate_eye_openness(landmarks, width, height)
            tension_task = self._calculate_muscle_tension(landmarks, width, height)

            symmetry_results = await symmetry_task
            eyes_results = await eyes_task
            tension_results = await tension_task

            metrics = {
                **symmetry_results,
                **eyes_results,
                **tension_results,
            }

            return metrics

        except Exception as e:
            print(f"ERROR: FaceMetricsAnalysis - {e}")
            return None

    async def _calculate_facial_symmetry(self, landmarks, image_width, image_height) -> Dict:
        """Улучшенный расчет симметрии с использованием большего количества точек"""
        left_points = []
        right_points = []

        for left_idx, right_idx in self.SYMMETRIC_PAIRS:
            left_landmark = landmarks.landmark[left_idx]
            right_landmark = landmarks.landmark[right_idx]

            left_points.append((
                left_landmark.x * image_width, 
                left_landmark.y * image_height
            ))
            right_points.append((
                right_landmark.x * image_width, 
                right_landmark.y * image_height
            ))

        # Векторный подход для вычисления расстояний
        left_array = np.array(left_points)
        right_array = np.array(right_points)
        
        # Отражаем правые точки
        reflected_right = np.copy(right_array)
        reflected_right[:, 0] = image_width - reflected_right[:, 0]
        
        # Вычисляем расстояния между соответствующими точками
        distances = np.linalg.norm(left_array - reflected_right, axis=1)
        
        # Адаптивная нормализация
        face_bbox = self._get_face_bounding_box(landmarks, image_width, image_height)
        face_diagonal = sqrt(face_bbox[2]**2 + face_bbox[3]**2)
        max_possible_diff = face_diagonal * 0.15  # Адаптивный порог

        symmetry_scores = 1 - np.clip(distances / max_possible_diff, 0, 1)
        overall_symmetry = float(np.mean(symmetry_scores))

        return {
            "facial_symmetry": overall_symmetry,
            "symmetry_confidence": float(np.std(symmetry_scores) < 0.08),
            "symmetry_details": {
                "eye_symmetry": float(symmetry_scores[0]),
                "mouth_symmetry": float(symmetry_scores[1]),
                "brow_symmetry": float(symmetry_scores[3])
            }
        }

    async def _calculate_eye_openness(self, landmarks, image_width, image_height) -> Dict:
        """Улучшенный расчет открытости глаз с использованием EAR"""
        def calculate_ear(eye_indices):
            points = []
            for idx in eye_indices:
                landmark = landmarks.landmark[idx]
                points.append((
                    landmark.x * image_width, 
                    landmark.y * image_height
                ))
            
            # EAR формула (Eye Aspect Ratio)
            # Вертикальные расстояния
            v1 = dist(points[1], points[5])
            v2 = dist(points[2], points[4])
            # Горизонтальное расстояние
            h = dist(points[0], points[3])
            
            if h == 0:
                return 0.0
                
            ear = (v1 + v2) / (2.0 * h)
            return min(1.0, ear * 1.5)  # Нормализация

        left_ear = calculate_ear(self.LEFT_EYE_INDICES)
        right_ear = calculate_ear(self.RIGHT_EYE_INDICES)
        avg_ear = (left_ear + right_ear) / 2

        return {
            "left_eye_openness": left_ear,
            "right_eye_openness": right_ear,
            "average_eye_openness": avg_ear,
            "eye_symmetry": 1 - abs(left_ear - right_ear),
            "eyes_closed": avg_ear < 0.2  # Порог для закрытых глаз
        }

    async def _calculate_muscle_tension(self, landmarks, image_width, image_height) -> Dict:
        """Улучшенный расчет напряжения мышц с дополнительными метриками"""
        # Нормализация относительно размера лица
        face_bbox = self._get_face_bounding_box(landmarks, image_width, image_height)
        face_size = max(face_bbox[2], face_bbox[3])

        # 1. Напряжение бровей (расстояние и угол)
        left_brow = landmarks.landmark[46]
        right_brow = landmarks.landmark[276]
        brow_center = landmarks.landmark[9]  # Центр между бровями
        
        brow_distance = dist(
            (left_brow.x * image_width, left_brow.y * image_height),
            (right_brow.x * image_width, right_brow.y * image_height)
        )
        
        # Высота бровей относительно глаз
        left_eye_top = landmarks.landmark[386]
        brow_height_left = abs(left_brow.y - left_eye_top.y) * image_height

        # 2. Напряжение рта (комплексный анализ)
        mouth_upper = landmarks.landmark[13]
        mouth_lower = landmarks.landmark[14]
        mouth_left = landmarks.landmark[78]
        mouth_right = landmarks.landmark[308]

        mouth_height = abs(mouth_upper.y - mouth_lower.y) * image_height
        mouth_width = abs(mouth_left.x - mouth_right.x) * image_width
        
        mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
        
        # 3. Напряжение челюсти
        jaw_left = landmarks.landmark[58]
        jaw_right = landmarks.landmark[288]
        jaw_tension = abs(jaw_left.y - jaw_right.y) * image_height

        # Нормализация относительно размера лица
        normalized_brow_tension = min(1.0, brow_distance / (face_size * 0.3))
        normalized_mouth_tension = min(1.0, mouth_ratio * 4)
        normalized_jaw_tension = min(1.0, jaw_tension / (face_size * 0.2))
        normalized_brow_height = min(1.0, brow_height_left / (face_size * 0.1))

        # Композитный показатель напряжения
        overall_tension = np.mean([
            normalized_brow_tension,
            normalized_mouth_tension,
            normalized_jaw_tension,
            normalized_brow_height
        ])

        return {
            "muscle_tension": float(overall_tension),
            "brow_tension": float(normalized_brow_tension),
            "mouth_tension": float(normalized_mouth_tension),
            "jaw_tension": float(normalized_jaw_tension),
            "forehead_tension": float(normalized_brow_height),
            "tension_confidence": overall_tension > 0.1  # Фильтр ложных срабатываний
        }

    def _get_face_bounding_box(self, landmarks, image_width, image_height) -> Tuple:
        """Вычисляет ограничивающий прямоугольник лица для нормализации"""
        x_coords = [lm.x * image_width for lm in landmarks.landmark]
        y_coords = [lm.y * image_height for lm in landmarks.landmark]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        width = x_max - x_min
        height = y_max - y_min
        
        return (x_min, y_min, width, height)

    def _normalize_to_face_size(self, value, face_size, factor=1.0) -> float:
        """Нормализует значение относительно размера лица"""
        return min(1.0, value / (face_size * factor))

    async def cleanup(self):
        """Очистка ресурсов"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()