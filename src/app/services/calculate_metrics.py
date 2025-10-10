class CalculateMetrics:
    def calculate_stress_level(self, emotion_data, eye_bags_score, facial_tension=0.5):
        """
        Вычисляет уровень стресса на основе:
        - эмоционального состояния
        - мешков под глазами (признак усталости)
        - мышечного напряжения лица
        """
        # Весовые коэффициенты
        emotion_weight = 0.3
        eye_bags_weight = 0.3
        tension_weight = 0.2

        # Анализ эмоций (стрессовые эмоции имеют больший вес)
        stress_emotions = ["angry", "fear", "sad", "disgust"]
        neutral_emotions = ["neutral"]
        positive_emotions = ["happy", "surprise"]

        stress_score = 0
        for emotion, confidence in emotion_data.items():
            if emotion in stress_emotions:
                stress_score += confidence * 1.0
            elif emotion in neutral_emotions:
                stress_score += confidence * 0.5
            elif emotion in positive_emotions:
                stress_score += confidence * 0.2

        # Нормализуем эмоциональный score
        emotion_stress = min(1.0, stress_score)

        # Общий расчет
        stress_level = (
            emotion_stress * emotion_weight
            + eye_bags_score * eye_bags_weight
            + facial_tension * tension_weight
        )

        return min(1.0, max(0.0, stress_level))

    def calculate_sleep_quality(
        self, eye_bags_score, redness_score, eye_openness=0.7, age=30
    ):
        """
        Вычисляет качество сна на основе:
        - мешков под глазами
        - покраснения глаз
        - открытости глаз
        - возраста (нормализация по возрасту)
        """
        # Возрастная нормализация (чем старше, тем сложнее скрыть недосып)
        age_factor = min(1.0, age / 50)  # после 50 лет фактор = 1

        # Весовые коэффициенты
        eye_bags_weight = 0.4
        redness_weight = 0.3
        eye_openness_weight = 0.3

        # Мешки под глазами (основной индикатор)
        eye_bags_impact = eye_bags_score * (0.8 + 0.2 * age_factor)

        # Покраснение глаз
        redness_impact = redness_score * (0.7 + 0.3 * age_factor)

        # Открытость глаз (чем меньше открыты, тем хуже сон)
        eye_openness_impact = (1 - eye_openness) * 0.8

        sleep_quality = 1 - (
            eye_bags_impact * eye_bags_weight
            + redness_impact * redness_weight
            + eye_openness_impact * eye_openness_weight
        )

        return max(0.0, min(1.0, sleep_quality))

    def calculate_skin_health_index(
        self, acne_score, redness_score, age, skin_smoothness=0.7
    ):
        """
        Вычисляет индекс здоровья кожи на основе:
        - акне
        - покраснений
        - возраста (нормализация)
        - гладкости кожи
        """
        # Возрастные коэффициенты (кожа ухудшается с возрастом)
        if age <= 25:
            age_factor = 1.0
        elif age <= 40:
            age_factor = 0.9
        elif age <= 60:
            age_factor = 0.8
        else:
            age_factor = 0.7

        # Весовые коэффициенты
        acne_weight = 0.4
        redness_weight = 0.3
        smoothness_weight = 0.2
        age_weight = 0.1

        # Расчет компонентов
        acne_impact = (1 - acne_score) * acne_weight
        redness_impact = (1 - redness_score) * redness_weight
        smoothness_impact = skin_smoothness * smoothness_weight
        age_impact = age_factor * age_weight

        skin_health_index = (
            acne_impact + redness_impact + smoothness_impact + age_impact
        )

        return max(0.0, min(1.0, skin_health_index))

    def calculate_vitality_score(self, all_metrics, weights=None):
        """
        Композитный показатель на основе всех метрик
        """
        if weights is None:
            weights = {
                "stress_level": 0.25,
                "sleep_quality": 0.25,
                "skin_health_index": 0.20,
                "facial_symmetry": 0.15,
                "muscle_tension": 0.15,
            }

        vitality = 0
        total_weight = 0

        for metric, weight in weights.items():
            if metric in all_metrics:
                # Для негативных метрик (stress_level, muscle_tension) инвертируем
                if metric in ["stress_level", "muscle_tension"]:
                    vitality += (1 - all_metrics[metric]) * weight
                else:
                    vitality += all_metrics[metric] * weight
                total_weight += weight

        # Нормализуем на случай отсутствия некоторых метрик
        if total_weight > 0:
            vitality /= total_weight

        return max(0.0, min(1.0, vitality))
