import streamlit as st
import pandas as pd
import joblib
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords once (safe for Streamlit Cloud)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# =========================
# LOAD SAVED MODEL + VECTORIZER
# =========================
model = joblib.load('emotion_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# IMPORTANT:
# Replace these labels if your mapping is different in training.
label_map = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}

stop_words = set(stopwords.words('english'))

# =========================
# TEXT CLEANING (same logic as notebook)
# =========================
def remove_punc(txt):
    return txt.translate(str.maketrans('', '', string.punctuation))


def remove_digits(txt):
    return ''.join([ch for ch in txt if not ch.isdigit()])


def remove_non_ascii(txt):
    return ''.join([ch for ch in txt if ch.isascii()])


def remove_stopwords(txt):
    words = txt.split()
    cleaned = [w for w in words if w not in stop_words]
    return ' '.join(cleaned)


def clean_text(text):
    text = text.lower()
    text = remove_punc(text)
    text = remove_digits(text)
    text = remove_non_ascii(text)
    text = remove_stopwords(text)
    return text


def predict_emotion(text):
    cleaned = clean_text(text)
    transformed = vectorizer.transform([cleaned])
    pred = model.predict(transformed)[0]
    return label_map.get(pred, f'Class {pred}'), cleaned

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title='Emotion Detection App', page_icon='🧠', layout='centered')

st.title('🧠 Emotion Detection using NLP')
st.markdown('Enter a sentence and the model will predict the **emotion**.')

user_input = st.text_area('Enter your text here', height=150, placeholder='Example: I am feeling really happy today!')

if st.button('Predict Emotion'):
    if user_input.strip() == '':
        st.warning('Please enter some text first.')
    else:
        emotion, cleaned_text = predict_emotion(user_input)
        st.success(f'Predicted Emotion: **{emotion.upper()}**')

        with st.expander('See cleaned text used by model'):
            st.write(cleaned_text)

st.markdown('---')
st.caption('Built with Streamlit + NLP + Logistic Regression')
