import streamlit as st
import pandas as pd
import os
import datetime
from textblob import TextBlob
from transformers import pipeline
import openai
import speech_recognition as sr

# Set OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

# Load emotion classifier
emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", framework="pt")


LOG_FILE = "full_training_dataset.csv"


# Sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity
#defining the emotions
def detect_emotion(text):
    # your logic here
    return emotion, score

# Emotion classification

emotion_analyzer = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=False,
    framework="pt"  # Force PyTorch
)
# OpenAI response
def get_chat_response(prompt):
    res = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a compassionate mental health assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return res['choices'][0]['message']['content']

# Transcribe audio
def transcribe_audio(audio_file):
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)
    return r.recognize_google(audio)

# Log mood
def log_mood_entry(text, emotion, score):
    df = pd.DataFrame([[datetime.datetime.now(), text, emotion, score]], columns=["timestamp", "entry", "emotion", "score"])
    if os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(LOG_FILE, index=False)

# Show mood summary
def show_summary():
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        st.line_chart(df["score"])
        st.dataframe(df.tail(5))

# Streamlit UI
st.title("MindCare: AI Mental Health Companion")

st.markdown("""
This chatbot helps you reflect on your mood using text or voice.
All interactions are private and anonymous.
""")

# Voice input
audio = st.file_uploader("Upload a voice note", type=["wav", "mp3"])
user_input = ""
if audio:
    try:
        user_input = transcribe_audio(audio)
        st.text_area("Transcribed Text", value=user_input, height=100)
    except Exception as e:
        st.error(f"Could not transcribe: {e}")
else:
    user_input = st.text_area("How are you feeling today?")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter or upload something you'd like to share.")
    else:
        polarity = analyze_sentiment(user_input)
        emotion, score = detect_emotion(user_input)
        ai_reply = get_chat_response(user_input)

        # Log and show
        log_mood_entry(user_input, emotion, score)

        st.success(f"Detected Emotion: {emotion} ({score:.2f})")
        st.info(f"Sentiment Polarity: {polarity:.2f}")
        st.markdown("---")
        st.markdown("### AI Suggestion")
        st.write(ai_reply)
        st.markdown("---")
        st.markdown("### Wellness Tip")
        st.markdown("Try a 4-7-8 breathing exercise: Breathe in for 4 seconds, hold for 7, and out for 8.")

st.markdown("---")
st.subheader("Mood Tracker Summary")
show_summary()
