import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import os

# Load dataset
df = pd.read_csv("cases.csv")

# Train model if not already trained
MODEL_FILE = "model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"

if not os.path.exists(MODEL_FILE):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["case"])
    y = df["section"]

    model = MultinomialNB()
    model.fit(X, y)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
else:
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)

# Streamlit UI
st.title("AI Lawyer Assistant")
st.write("Enter case details and get the relevant IPC Section with explanation.")

user_input = st.text_area("Enter case description:")

if st.button("Predict Section"):
    if user_input.strip() != "":
        X_test = vectorizer.transform([user_input])
        predicted_section = model.predict(X_test)[0]

        # Fetch details
        details = df[df["section"] == predicted_section]["details"].values[0]

        st.subheader(f"Predicted Section: {predicted_section}")
        st.write(f"**Details:** {details}")

        st.success(f"This case falls under **{predicted_section}** because of similarity with training cases.")
    else:
        st.error("Please enter case details.")
