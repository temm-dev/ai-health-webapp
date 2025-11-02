<h1 align="center"> AI-HEALTH-APP </h1>

<p align="center">
  <img src="assets/images/health.jpg" alt="Project Demo" width="800">
</p>

<div align="center">

[![Python 3.13.5](https://img.shields.io/badge/python-3.13.7-blue.svg)](https://www.python.org/downloads/)

</div>

A web application that uses computer vision and artificial intelligence technologies to analyze faces from photographs and assess users' health status.

## 💡 Project idea
The user visits the website / application / telegram bot, uploads a photo or video, the service processes the data, and then issues a report on the user's health.

## ⚙️ Technology stack

### **Backend**
- Python 3.13.7
- FastAPI
- Uvicorn
- Scikit-Learn
- TensorFLow
- Aiogram 3 (asynchronous framework for Telegram bots)
- Asyncio/Aiohttp/Aiofiles (asynchronous operations)

### **Frontend**
- HTML, CSS, JS


## 🚀 Quick start

### 💻 Local installation
```bash
# Clone a repository
git clone https://github.com/temm-dev/ai-health-webapp.git
cd ai-health-webapp

# Install dependencies
pip install -r requirements.txt

# Launch the app
cd src
uvicorn app.main:app --host 0.0.0.0 --port 8000
```


## 🌐 API Usage
```bash
curl -X POST "http://0.0.0.0/api/v1/upload" -F "file=@/path_to_image.jpg"
```

### Example of a successful response
```json
{
  "age": 24,
  "region": {
    "x": 194,
    "y": 380,
    "w": 552,
    "h": 552,
    "left_eye": [
      561,
      598
    ],
    "right_eye": [
      353,
      606
    ]
  },
  "face_confidence": 0.9,
  "gender": {
    "Woman": 0.06263645482249558,
    "Man": 99.93736147880554
  },
  "dominant_gender": "Man",
  "race": {
    "asian": 0.017265962378587574,
    "indian": 0.03209102142136544,
    "black": 0.0018770331735140644,
    "white": 86.25044822692871,
    "middle eastern": 4.870230332016945,
    "latino hispanic": 8.828084915876389
  },
  "dominant_race": "white",
  "emotion": {
    "angry": 8.769406378269196,
    "disgust": 0.0003436009365032078,
    "fear": 13.045157492160797,
    "happy": 0.0036395420465851203,
    "sad": 44.0920889377594,
    "surprise": 0.00023784293716744287,
    "neutral": 34.08912718296051
  },
  "dominant_emotion": "sad",
  "time": 4.566195011138916,
  "antispoof_score": 0.9,
  "is_real": true,
  "acne": 0.15842,
  "redness": 0.79574,
  "eyebags": 0.04584,
  "facial_symmetry": 0.5230155638766237,
  "symmetry_confidence": 0,
  "left_eye_openness": 0.4007472406624948,
  "right_eye_openness": 0.3973293645249586,
  "average_eye_openness": 0.3990383025937267,
  "eye_symmetry": 0.9965821238624638,
  "muscle_tension": 0.364219668941149,
  "brow_tension": 1,
  "mouth_tension": 0.006995794912265244,
  "forehead_tension": 0.2708175729745405,
  "red_flags": [
    "red_flag1",
    "red_flag2",
    "red_flag3"
  ],
  "recommendations": "recommendations",
  "risk_level": "medium",
  "analysis_id": "analysis_id",
  "timestamp": "2020-1-03T00:6:50.398361",
  "stress_level": 0.44514008857013865,
  "sleep_quality": 0.6302220190110182,
  "skin_health_index": 0.7079099999999999,
  "vitality_score": 0.6116718668505411,
  "user_message": ""
}
```


<br>
<br>
<br>

> **Have a nice day!** 🍀