import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Title
st.title("AI Sentiment Analyzer")

# Input
user_input = st.text_area("Enter a review or tweet")

# Button
if st.button("Analyze Sentiment"):

    if user_input.strip() == "":
        st.warning("Please enter some text")

    else:

        # Convert text
        input_vector = vectorizer.transform([user_input])

        # Prediction probabilities
        probabilities = model.predict_proba(input_vector)

        negative_score = probabilities[0][0] * 100
        positive_score = probabilities[0][1] * 100

        # Neutral logic
        difference = abs(positive_score - negative_score)

        # Show sentiment
        if difference < 20:
            st.warning("Neutral 😐")

        elif positive_score > negative_score:
            st.success("Positive 😊")

        else:
            st.error("Negative 😠")

        # Show scores
        st.write(f"Positive: {positive_score:.2f}%")
        st.write(f"Negative: {negative_score:.2f}%")