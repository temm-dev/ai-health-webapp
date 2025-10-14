from math import sqrt
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATH_MODELS = (PROJECT_ROOT / "app" / "models").as_posix() + "/"
PATH_TEST_IMAGES = (PROJECT_ROOT / "data" / "test_images").as_posix() + "/"


class FaceMetricsAnalysis:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh  # type: ignore
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

    def analyze(self, image_path):
        """Анализ всех метрик лица"""
        path = f"data/temp/{image_path}"

        image = cv2.imread(path)
        if image is None:
            return None

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0]
        image_height, image_width = image.shape[:2]

        metrics = {
            **self._calculate_facial_symmetry(landmarks, image_width, image_height),
            **self._calculate_eye_openness(landmarks, image_width, image_height),
            **self._calculate_muscle_tension(landmarks, image_width, image_height),
        }

        return metrics

    def _calculate_facial_symmetry(self, landmarks, image_width, image_height):
        """Расчет симметрии лица"""
        # Ключевые симметричные точки
        left_points = []
        right_points = []

        # Индексы симметричных точек из MediaPipe Face Mesh
        symmetric_pairs = [
            (162, 389),  # Левый и правый край глаза
            (234, 454),  # Левый и правый угол рта
            (130, 359),  # Левый и правый край носа
            (93, 323),  # Левый и правый край брови
            (58, 288),  # Левый и правый край челюсти
        ]

        for left_idx, right_idx in symmetric_pairs:
            left_landmark = landmarks.landmark[left_idx]
            right_landmark = landmarks.landmark[right_idx]

            left_points.append(
                (left_landmark.x * image_width, left_landmark.y * image_height)
            )
            right_points.append(
                (right_landmark.x * image_width, right_landmark.y * image_height)
            )

        # Рассчитываем разницу между симметричными точками
        differences = []
        for left, right in zip(left_points, right_points):
            # Отражаем правую точку относительно центра
            reflected_right = (image_width - right[0], right[1])
            distance = sqrt(
                (left[0] - reflected_right[0]) ** 2
                + (left[1] - reflected_right[1]) ** 2
            )
            differences.append(distance)

        # Нормализуем и вычисляем общую симметрию
        max_possible_diff = sqrt(image_width**2 + image_height**2) * 0.1
        symmetry_scores = [1 - (diff / max_possible_diff) for diff in differences]
        overall_symmetry = np.mean(symmetry_scores)

        return {
            "facial_symmetry": max(0, min(1, overall_symmetry)),
            "symmetry_confidence": np.std(symmetry_scores)
            < 0.1,  # Низкое std = высокая уверенность
        }

    def _calculate_eye_openness(self, landmarks, image_width, image_height):
        """Расчет открытости глаз"""
        # Индексы для левого и правого глаза (вертикальные измерения)
        left_eye_indices = [386, 374, 263, 362]  # верх-низ левого глаза
        right_eye_indices = [159, 145, 33, 133]  # верх-низ правого глаза

        def get_eye_openness(eye_indices):
            """Вычисляет открытость для одного глаза"""
            points = []
            for idx in eye_indices:
                landmark = landmarks.landmark[idx]
                points.append((landmark.x * image_width, landmark.y * image_height))

            # Вертикальное расстояние между верхом и низом глаза
            vertical_distance = sqrt(
                (points[0][0] - points[1][0]) ** 2 + (points[0][1] - points[1][1]) ** 2
            )

            # Горизонтальное расстояние как референс
            horizontal_distance = sqrt(
                (points[2][0] - points[3][0]) ** 2 + (points[2][1] - points[3][1]) ** 2
            )

            # Отношение вертикального к горизонтальному (нормализованное)
            openness_ratio = (
                vertical_distance / horizontal_distance
                if horizontal_distance > 0
                else 0
            )

            return min(1.0, openness_ratio * 3)  # Нормализуем к 0-1

        left_eye_openness = get_eye_openness(left_eye_indices)
        right_eye_openness = get_eye_openness(right_eye_indices)

        return {
            "left_eye_openness": left_eye_openness,
            "right_eye_openness": right_eye_openness,
            "average_eye_openness": (left_eye_openness + right_eye_openness) / 2,
            "eye_symmetry": 1 - abs(left_eye_openness - right_eye_openness),
        }

    def _calculate_muscle_tension(self, landmarks, image_width, image_height):
        """Расчет напряжения лицевых мышц"""
        # 1. Напряжение бровей (расстояние между бровями)
        left_brow = landmarks.landmark[46]  # Левая бровь
        right_brow = landmarks.landmark[276]  # Правая бровь

        brow_distance = sqrt(
            (left_brow.x - right_brow.x) ** 2 * image_width**2
            + (left_brow.y - right_brow.y) ** 2 * image_height**2
        )

        # 2. Напряжение рта (открытость/сжатость)
        mouth_upper = landmarks.landmark[13]  # Верхняя губа
        mouth_lower = landmarks.landmark[14]  # Нижняя губа
        mouth_left = landmarks.landmark[78]  # Левый угол рта
        mouth_right = landmarks.landmark[308]  # Правый угол рта

        mouth_vertical = abs(mouth_upper.y - mouth_lower.y) * image_height
        mouth_horizontal = abs(mouth_left.x - mouth_right.x) * image_width

        mouth_tension_ratio = (
            mouth_vertical / mouth_horizontal if mouth_horizontal > 0 else 0
        )

        # 3. Напряжение лба (по положению бровей относительно глаз)
        left_eye_center = landmarks.landmark[468]  # Центр левого глаза
        brow_eye_distance_left = abs(left_brow.y - left_eye_center.y) * image_height

        # Нормализуем метрики
        normalized_brow_tension = min(1.0, brow_distance / (image_width * 0.2))
        normalized_mouth_tension = min(1.0, mouth_tension_ratio * 5)
        normalized_forehead_tension = min(
            1.0, brow_eye_distance_left / (image_height * 0.1)
        )

        overall_tension = (
            normalized_brow_tension
            + normalized_mouth_tension
            + normalized_forehead_tension
        ) / 3

        return {
            "muscle_tension": overall_tension,
            "brow_tension": normalized_brow_tension,
            "mouth_tension": normalized_mouth_tension,
            "forehead_tension": normalized_forehead_tension,
        }
