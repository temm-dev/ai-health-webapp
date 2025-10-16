import logging
import uuid
from datetime import datetime
from typing import List, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key="AIzaSyDRujHWa_BOpcKmDUZDRnHoVhZX9TftWAM",
    temperature=0.4,
)


# Обновленный TypedDict с учетом всех полей из JSON
class State(TypedDict):
    age: int
    region: dict
    face_confidence: float
    gender: dict
    dominant_gender: str
    race: dict
    dominant_race: str
    emotion: dict
    dominant_emotion: str
    time: float
    antispoof_score: float
    is_real: bool
    acne: float
    redness: float
    eyebags: float
    facial_symmetry: float
    symmetry_confidence: bool
    left_eye_openness: float
    right_eye_openness: float
    average_eye_openness: float
    eye_symmetry: float
    muscle_tension: float
    brow_tension: float
    mouth_tension: float
    forehead_tension: float
    red_flags: List[str]
    recommendations: str
    risk_level: str
    analysis_id: str
    timestamp: str
    stress_level: float
    sleep_quality: float
    skin_health_index: float
    vitality_score: float
    user_message: str


def transform_json_to_state(json_data: dict) -> State:
    """Преобразует JSON данные в формат State"""
    # analyze_data = json_data.get("analyze", {})

    return {
        "age": json_data.get("age", 0),
        "region": json_data.get("region", {}),
        "face_confidence": json_data.get("face_confidence", 0.0),
        "gender": json_data.get("gender", {}),
        "dominant_gender": json_data.get("dominant_gender", "unknown"),
        "race": json_data.get("race", {}),
        "dominant_race": json_data.get("dominant_race", "unknown"),
        "emotion": json_data.get("emotion", {}),
        "dominant_emotion": json_data.get("dominant_emotion", "neutral"),
        "time": json_data.get("time", 0.0),
        "antispoof_score": 0.9,
        "is_real": True,
        "acne": json_data.get("acne", 0.0),
        "redness": json_data.get("redness", 0.0),
        "eyebags": json_data.get("eyebags", 0.0),
        "facial_symmetry": json_data.get("facial_symmetry", 0.0),
        "symmetry_confidence": json_data.get("symmetry_confidence", False),
        "left_eye_openness": json_data.get("left_eye_openness", 0.0),
        "right_eye_openness": json_data.get("right_eye_openness", 0.0),
        "average_eye_openness": json_data.get("average_eye_openness", 0.0),
        "eye_symmetry": json_data.get("eye_symmetry", 0.0),
        "muscle_tension": json_data.get("muscle_tension", 0.0),
        "brow_tension": json_data.get("brow_tension", 0.0),
        "mouth_tension": json_data.get("mouth_tension", 0.0),
        "forehead_tension": json_data.get("forehead_tension", 0.0),
        "red_flags": [],
        "recommendations": "",
        "risk_level": "low",
        "analysis_id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "stress_level": json_data.get("stress_level", 0.0),
        "sleep_quality": json_data.get("sleep_quality", 0.0),
        "skin_health_index": json_data.get("skin_health_index", 0.0),
        "vitality_score": json_data.get("vitality_score", 0.0),
        "user_message": json_data.get("user_message", ""),
    }


async def analyze_parameters(state: State):
    """Анализирует параметры с фокусом на наблюдения, а не диагнозы"""
    try:
        logger.info(f"🔍 Начало анализа для {state.get('analysis_id', 'unknown')}")

        # Преобразуем confidence в проценты для читаемости
        confidence_percent = int(state["face_confidence"] * 100)

        # Получаем уровень нейтральной эмоции
        neutral_emotion_percent = state["emotion"].get("neutral", 0.0)

        analysis_prompt = f"""
        Ты - медицинский ассистент для первичного анализа состояния человека. Анализируй только наблюдаемые параметры, не ставь диагнозы.

        ДАННЫЕ ДЛЯ АНАЛИЗА:
        - Возраст: {state['age']}
        - Пол: {state['dominant_gender']}
        - Уверенность распознавания: {confidence_percent}%
        - Доминирующая эмоция: {state['dominant_emotion']} (нейтральность: {neutral_emotion_percent:.1f}%)
        - Уровень акне: {state['acne']:.4f}/1
        - Мешки под глазами: {state['eyebags']:.4f}/1
        - Симметрия лица: {state['facial_symmetry']:.3f}/1
        - Открытость глаз: {state['average_eye_openness']:.3f}/1
        - Напряжение мышц: {state['muscle_tension']:.3f}/1
        - Уровень стресса: {state['stress_level']:.3f}/1
        - Качество сна: {state['sleep_quality']:.3f}/1
        - Здоровье кожи: {state['skin_health_index']:.3f}/1
        - Индекс жизнеспособности: {state['vitality_score']:.3f}/1

        ПОЛЬЗОВАТЕЛЬСКОЕ СООБЩЕНИЕ: {state.get("user_message", "Не предоставлено")}

        ИНСТРУКЦИЯ:
        1. Проанализируй КАЖДЫЙ параметр отдельно
        2. Формулируй только нейтральные наблюдения
        3. Не делай медицинских заключений
        4. Учитывай возрастные нормы
        5. Если параметр близок к 0 или 1 - укажи на это

        ФОРМАТ ОТВЕТА (только наблюдения):
        - Эмоциональное состояние: [наблюдение]
        - Состояние кожи: [наблюдение] 
        - Область глаз: [наблюдение]
        - Мышечное напряжение: [наблюдение]
        - Общие показатели: [наблюдение]

        Примеры корректных формулировок:
        ✅ "Наблюдается преимущественно нейтральное эмоциональное состояние"
        ✅ "Присутствуют минимальные проявления акне"
        ✅ "Обнаружена выраженная асимметрия черт лица"
        ✅ "Зафиксирован повышенный уровень мышечного напряжения"

        Если все параметры в норме, укажи "Значимых отклонений не обнаружено".
        """

        analysis_result = await llm.ainvoke(analysis_prompt)

        # Упрощенный парсинг ответа
        observations = []
        if "Значимых отклонений не обнаружено" not in analysis_result.content:
            lines = analysis_result.content.strip().split("\n") # type: ignore
            for line in lines:
                line = line.strip()
                if line.startswith("-") and ":" in line:
                    observation = line[1:].strip()
                    if (
                        observation
                        and "Значимых отклонений не обнаружено" not in observation
                    ):
                        observations.append(observation)

        # Оценка уровня риска на основе объективных параметров
        risk_factors = 0
        if state["dominant_emotion"] in ["sad", "angry", "fear"]:
            risk_factors += 1
        if state["acne"] > 0.5:
            risk_factors += 1
        if state["eyebags"] > 0.5:
            risk_factors += 1
        if state["stress_level"] > 0.7:
            risk_factors += 1
        if state["sleep_quality"] < 0.5:
            risk_factors += 1

        risk_level = (
            "high" if risk_factors >= 3 else "medium" if risk_factors >= 1 else "low"
        )

        print("=" * 60)
        print("АНАЛИЗ ПАРАМЕТРОВ:")
        print("=" * 60)
        if observations:
            for i, obs in enumerate(observations, 1):
                print(f"📝 {i}. {obs}")
        else:
            print("✅ Значимых отклонений не обнаружено")
        print(f"📊 Уровень риска: {risk_level.upper()}")
        print("=" * 60)

        state["red_flags"] = observations
        state["risk_level"] = risk_level

        logger.info(
            f"✅ Анализ завершен. Найдено наблюдений: {len(observations)}, риск: {risk_level}"
        )
        return state

    except Exception as e:
        logger.error(f"❌ Ошибка в analyze_parameters: {e}")
        state["red_flags"] = ["Временные технические трудности при анализе"]
        state["recommendations"] = "Пожалуйста, попробуйте позже."
        state["risk_level"] = "error"
        return state


async def generate_recommendations(state: State):
    """Генерирует безопасные практические рекомендации"""
    try:
        if not state["red_flags"]:
            state["recommendations"] = (
                "✅ Все параметры в пределах нормы. Рекомендуется поддерживать текущий образ жизни."
            )
            print("✅ Все параметры в норме")
            return state

        recommendations_prompt = f"""
        На основе предоставленных наблюдений предложи практические рекомендации общего характера.

        НАБЛЮДЕНИЯ ДЛЯ АНАЛИЗА:
        {chr(10).join(f'- {obs}' for obs in state['red_flags'])}

        КОНТЕКСТ:
        - Возраст: {state['age']}
        - Пол: {state['dominant_gender']}
        - Индекс жизнеспособности: {state['vitality_score']:.3f}/1

        СТРОГИЕ ОГРАНИЧЕНИЯ:
        - ЗАПРЕЩЕНО назначать лекарства и БАДы
        - ЗАПРЕЩЕНО ставить диагнозы
        - ЗАПРЕЩЕНО рекомендовать конкретных врачей
        - Только общие советы по улучшению качества жизни
        - Максимум 250 слов
        - Простой, доступный язык

        ТРЕБУЕМЫЙ ФОРМАТ ОТВЕТА:
        Основные рекомендации
        [1-2 самых важных общих совета]

        Конкретные действия
        [3-5 практических шагов, которые можно выполнить]

        Когда стоит проконсультироваться
        [2-3 общие ситуации для обращения к специалистам]

        ВАЖНО: Ответ должен быть единым текстом без Markdown разметки, специальных символов и переносов строк.
        """

        recommendations_result = await llm.ainvoke(recommendations_prompt)

        # Добавляем медицинский дисклеймер
        disclaimer = "\n\n---\n*Важно: данные рекомендации носят общеоздоровительный характер и не заменяют консультацию специалиста.*"

        recommendations = recommendations_result.content + disclaimer # type: ignore

        print("РЕКОМЕНДАЦИИ:")
        print("=" * 60)
        print(recommendations)
        print("=" * 60)

        state["recommendations"] = recommendations
        logger.info(
            f"✅ Рекомендации сгенерированы, длина: {len(recommendations)} символов"
        )
        return state

    except Exception as e:
        logger.error(f"❌ Ошибка в generate_recommendations: {e}")
        state["recommendations"] = (
            "Временные технические трудности при генерации рекомендаций."
        )
        return state


def create_graph():
    """Создает граф обработки параметров"""
    graph = StateGraph(State)

    # Добавляем ноды
    graph.add_node("analyze_parameters", analyze_parameters)
    graph.add_node("generate_recommendations", generate_recommendations)

    # Определяем порядок выполнения
    graph.set_entry_point("analyze_parameters")
    graph.add_edge("analyze_parameters", "generate_recommendations")
    graph.add_edge("generate_recommendations", END)

    return graph


async def analyze_from_json(json_data: dict):
    """Основная функция для анализа из JSON данных"""
    try:
        graph = create_graph()
        app = graph.compile()

        # Преобразуем JSON в State
        state_data = transform_json_to_state(json_data)

        logger.info("🚀 Запуск анализа из JSON...")
        result = await app.ainvoke(state_data)

        # Вывод структурированных результатов
        print("\n" + "=" * 80)
        print("📊 ИТОГОВЫЙ ОТЧЕТ")
        print("=" * 80)
        print(f"🆔 ID анализа: {result.get('analysis_id', 'N/A')}")
        print(f"⏰ Время анализа: {result.get('timestamp', 'N/A')}")
        print(f"📈 Уровень риска: {result.get('risk_level', 'N/A').upper()}")
        print(f"🔍 Наблюдений: {len(result.get('red_flags', []))}")
        print("=" * 80)

        return result

    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        return {
            "error": str(e),
            "red_flags": ["Системная ошибка"],
            "recommendations": "Пожалуйста, обратитесь в техническую поддержку.",
            "risk_level": "error",
        }


# Пример использования с вашим JSON
async def main():
    """Пример использования с вашим JSON"""

    # Ваш JSON данные
    json_data = {
        "status": "success",
        "filename": "5bffe367-827e-43cc-9630-56a72625541a.jpg",
        "analyze": {
            "age": 27,
            "region": {
                "x": 92,
                "y": 317,
                "w": 638,
                "h": 638,
                "left_eye": [515, 574],
                "right_eye": [283, 572],
            },
            "face_confidence": 0.93,
            "gender": {"Woman": 0.11057894444093108, "Man": 99.88942742347717},
            "dominant_gender": "Man",
            "race": {
                "asian": 39.49163556098938,
                "indian": 8.91764983534813,
                "black": 5.210122466087341,
                "white": 7.888878136873245,
                "middle eastern": 4.392336681485176,
                "latino hispanic": 34.0993732213974,
            },
            "dominant_race": "asian",
            "emotion": {
                "angry": 16.673298677599274,
                "disgust": 1.3068607755107955e-07,
                "fear": 0.37283170756181566,
                "happy": 1.3823053506544301e-05,
                "sad": 39.418076207408454,
                "surprise": 3.756743062607554e-05,
                "neutral": 43.535747353523654,
            },
            "dominant_emotion": "neutral",
            "time": 3.9077908992767334,
            "acne": 0.53654,
            "redness": 3e-05,
            "eyebags": 0.46343,
            "facial_symmetry": 0.39427789912783345,
            "symmetry_confidence": True,
            "left_eye_openness": 0.984048781730791,
            "right_eye_openness": 1.0,
            "average_eye_openness": 0.9920243908653955,
            "eye_symmetry": 0.984048781730791,
            "muscle_tension": 0.4665816494928699,
            "brow_tension": 1.0,
            "mouth_tension": 0.02770259055700948,
            "forehead_tension": 0.37204235792160034,
            "stress_level": 0.532345329898574,
            "sleep_quality": 0.8297603198076949,
            "skin_health_index": 0.715375,
            "vitality_score": 0.0,
        },
    }

    result = await analyze_from_json(json_data)
    return result


# if __name__ == "__main__":
#     asyncio.run(main())
