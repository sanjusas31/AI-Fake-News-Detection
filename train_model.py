import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


print("Loading datasets...")


# Load Fake News dataset
fake = pd.read_csv("dataset/Fake.csv")

# Load Real News dataset
real = pd.read_csv("dataset/True.csv")


# Add labels
fake["label"] = 0
real["label"] = 1


print("Combining datasets...")


# Combine both datasets
data = pd.concat([fake, real])


# Combine title and text
data["content"] = (
    data["title"].fillna("") + " " +
    data["text"].fillna("")
)


# Keep only content and label
data = data[["content", "label"]]


# Shuffle data
data = data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# X = input news text
# y = output label
X = data["content"]
y = data["label"]


print("Splitting dataset...")


# 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("Converting text into numbers...")


# Convert text to numerical features
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)


X_train_vectorized = vectorizer.fit_transform(X_train)

X_test_vectorized = vectorizer.transform(X_test)


print("Training AI model...")


# Create Machine Learning model
model = LogisticRegression(
    max_iter=1000
)


# Train the model
model.fit(
    X_train_vectorized,
    y_train
)


print("Testing model...")


# Make predictions
prediction = model.predict(
    X_test_vectorized
)


# Calculate accuracy
accuracy = accuracy_score(
    y_test,
    prediction
)


print("\nModel Accuracy:", round(accuracy * 100, 2), "%")


print("Saving model...")


# Save trained model
joblib.dump(
    model,
    "model/fake_news_model.pkl"
)


# Save TF-IDF vectorizer
joblib.dump(
    vectorizer,
    "model/vectorizer.pkl"
)


print("\nSUCCESS!")
print("AI Model and Vectorizer saved successfully!")