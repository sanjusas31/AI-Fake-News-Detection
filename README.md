# Fake and Real News Classification System

An AI-powered Fake News Detection web application that analyzes news articles and predicts whether the news is Fake or Real.

## Features

- Detect Fake and Real News
- Machine Learning based prediction
- Confidence score
- Input validation
- Prediction history
- SQLite database
- Clear prediction history
- REST API
- Swagger API documentation
- User-friendly web interface

## Technologies Used

- Python
- FastAPI
- Scikit-learn
- Pandas
- HTML
- CSS
- SQLite
- Jinja2

## Machine Learning Model

The project uses:

- TF-IDF Vectorizer
- Logistic Regression

The model was trained using Fake and Real News datasets.

Model Accuracy: 98.45%

## Project Structure

```text
AI-Fake-News-Detection/
│
├── dataset/
│   ├── Fake.csv
│   └── True.csv
│
├── model/
│   ├── fake_news_model.pkl
│   └── vectorizer.pkl
│
├── static/
│   └── style.css
│
├── templates/
│   ├── index.html
│   └── history.html
│
├── main.py
├── train_model.py
├── requirements.txt
├── news_history.db
└── README.md