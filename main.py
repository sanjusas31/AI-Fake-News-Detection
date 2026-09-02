from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import joblib
import sqlite3

app = FastAPI(
    title="AI Fake News Detection System"
)


# Static files
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# Templates
templates = Jinja2Templates(
    directory="templates"
)


# Load trained model
model = joblib.load(
    "model/fake_news_model.pkl"
)


# Load vectorizer
vectorizer = joblib.load(
    "model/vectorizer.pkl"
)
# Create database connection
conn = sqlite3.connect("news_history.db", check_same_thread=False)

cursor = conn.cursor()

# Create prediction history table
cursor.execute("""
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    news TEXT,
    prediction TEXT,
    confidence REAL
)
""")

conn.commit()

# Home page
@app.get("/", response_class=HTMLResponse)
def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# Prediction route
@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    news: str = Form(...)
):
    if len(news.strip()) < 20:
        return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "error": "Please enter a news article with at least 20 characters.",
            "news": news
        }
    )

    # Convert text into numbers
    news_vector = vectorizer.transform([news])


    # Prediction
    prediction = model.predict(news_vector)[0]


    # Confidence
    probability = model.predict_proba(news_vector)[0]

    confidence = max(probability) * 100


    # Result
    if prediction == 0:
        result = "FAKE NEWS ❌"
        result_class = "fake"
    else:
        result = "REAL NEWS ✅"
    result_class = "real"
    cursor.execute(
        "INSERT INTO history (news, prediction, confidence) VALUES (?, ?, ?)",
        (news, result, round(confidence, 2)))
    conn.commit()
    return templates.TemplateResponse(
    request=request,
    name="index.html",
    context={
        "result": result,
        "result_class": result_class,
        "confidence": round(confidence, 2),
        "news": news
    }
)
class NewsInput(BaseModel):
    news: str


@app.post("/api/predict")
def api_predict(data: NewsInput):

    news_vector = vectorizer.transform([data.news])

    prediction = model.predict(news_vector)[0]

    probability = model.predict_proba(news_vector)[0]

    confidence = max(probability) * 100

    if prediction == 0:
        result = "FAKE NEWS"
    else:
        result = "REAL NEWS"

    return {
        "prediction": result,
        "confidence": round(confidence, 2)
    }
class NewsInput(BaseModel):
    news: str


@app.post("/api/predict")
def api_predict(data: NewsInput):

    news_vector = vectorizer.transform([data.news])

    prediction = model.predict(news_vector)[0]

    probability = model.predict_proba(news_vector)[0]

    confidence = max(probability) * 100

    if prediction == 0:
        result = "FAKE NEWS"
    else:
        result = "REAL NEWS"

    return {
        "prediction": result,
        "confidence": round(confidence, 2)
    }
@app.get("/history", response_class=HTMLResponse)
def history(request: Request):

    cursor.execute(
        "SELECT id, news, prediction, confidence FROM history ORDER BY id DESC"
    )

    records = cursor.fetchall()

    return templates.TemplateResponse(
        request=request,
        name="history.html",
        context={
            "records": records
        }
    )
@app.post("/clear-history")
def clear_history():

    cursor.execute("DELETE FROM history")

    conn.commit()

    return RedirectResponse(
        url="/history",
        status_code=303
    )