# 🏠 Prop-O-Lex: India Property Valuation System

Prop-O-Lex is a state-of-the-art property valuation system for the Indian real estate market. It leverages a hybrid machine learning approach (Linear Regression + XGBoost) to provide accurate estimations based on location, property type, area, and historical trends.

## 🚀 Live Links
- **Web App**: [https://prop-o-lex-edav.vercel.app](https://prop-o-lex-edav.vercel.app)
- **API Documentation**: [https://web-production-665c.up.railway.app/docs](https://web-production-665c.up.railway.app/docs)

## 🏗️ Architecture
The project follows a **Two-Tier Split Architecture**:
1.  **Backend (Railway)**: Python/FastAPI service hosting the ML models and geographic proxy endpoints.
2.  **Frontend (Vercel)**: Static HTML/JS/CSS application with a modern, glassmorphic UI.

### Key Components
- `app.py`: FastAPI entry point.
- `hybrid_inference.py`: Core logic for combined model predictions and confidence scoring.
- `feature_engineering.py`: Processes raw input into ML-ready features (e.g., city growth scores).
- `models/`: Contains trained `.pkl` pipeline files and metadata.

## 🛠️ Developer Setup

### 1. Installation
```bash
git clone https://github.com/innosifytechnologies/Prop_O_Lex.git
cd Prop_O_Lex
pip install -r requirements.txt
```

### 2. Configuration
Create a `config.py` file with your LocationIQ API key:
```python
LOCATIONIQ_API_KEY = "your_key_here"
```

### 3. Local Development
```bash
python app.py
```
Open `index.html` in your browser. By default, it points to `localhost:8000`.

## 🧠 Machine Learning
- **Dataset**: ~70,000 deduplicated records across major Indian cities.
- **Features**: Includes historical price trends (2000-2025) and city-tier growth scores.
- **Training**: Use `train_models.py` to retrain the models.

## 📍 API Reference
The API is self-documenting via Swagger. Visit `/docs` on the live backend for full details.

#### `POST /predict`
Estimates property value.
- **Input**: JSON with lat/lon, city, area, bhk, property type.
- **Output**: JSON with estimated price, confidence level, and price band.

#### `GET /geocode`
Proxy to LocationIQ for address auto-suggestions.

## 🚢 Deployment
- **Backend**: Railway (Auto-deploy on Git push via `Procfile`).
- **Frontend**: Vercel (Deployed as a static site). **Note**: Remember to update `BACKEND_URL` in `index.html` before deploying.
