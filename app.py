import streamlit as st
from textblob import TextBlob
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import datetime
from fpdf import FPDF
import pandas as pd
from io import BytesIO
import os
import speech_recognition as sr


nltk.download('punkt')

st.set_page_config(page_title="Mental Health AI Chatbot", layout="centered")
st.title("🧠 AI Mental Health Chatbot")


# --- Train Local Model ---
@st.cache_resource
def train_local_emotion_model():
    texts = [
        "I feel so happy and excited today",
        "Life is beautiful and I'm grateful",
        "I'm okay, just going with the flow",
        "It's been a boring day",
        "I'm feeling really sad and down",
        "I can't take this anymore, it's too much"
    ]
    labels = ["positive", "positive", "neutral", "neutral", "negative", "negative"]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    model = MultinomialNB()
    model.fit(X, labels)
    return model, vectorizer

model, vectorizer = train_local_emotion_model()

def predict_emotion_local(text):
    X_test = vectorizer.transform([text])
    return model.predict(X_test)[0]

# Coping strategies
coping_strategies = {
    "positive": ["Keep up the positive thoughts!", "Practice gratitude daily.", "Stay connected with loved ones."],
    "neutral": ["Try mindfulness meditation.", "Keep a journal.", "Take short walks to clear your mind."],
    "negative": ["Take deep breaths.", "Reach out to a friend or therapist.", "Write down your thoughts."]
}
therapists = [
    "BetterHelp: https://www.betterhelp.com",
    "Talkspace: https://www.talkspace.com",
    "7 Cups: https://www.7cups.com"
]

# History state
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar input choice
st.sidebar.header("Choose Input Method")
input_mode = st.sidebar.radio("Input type", ["Text", "Voice"])
user_input = ""

# --- Text Input ---
if input_mode == "Text":
    user_input = st.text_area("How are you feeling today?", key="text_input")


elif input_mode == "Voice":
    uploaded_file = st.file_uploader("Upload your voice (mp3 format)", type=['mp3'])
    if uploaded_file:
        # Save the uploaded file as temp.mp3
        with open("temp.mp3", "wb") as f:
            f.write(uploaded_file.read())
        
        st.audio("temp.mp3")  # This will allow playing the uploaded MP3
        
        # Now you can process the MP3 file
        recognizer = sr.Recognizer()
        audio_file = sr.AudioFile("temp.mp3")
        
        # Transcribing the MP3 file
        with audio_file as source:
            audio = recognizer.record(source)  # Read the entire file
            try:
                text = recognizer.recognize_google(audio)  # You can use any speech-to-text API here
                st.write("Transcription: ", text)  # Display the transcribed text
            except sr.UnknownValueError:
                st.write("Sorry, could not understand the audio.")
            except sr.RequestError as e:
                st.write(f"Error with the speech-to-text service: {e}")
        try:
            # Attempt whisper first
            import whisper
            model_whisper = whisper.load_model("base")
            result = model_whisper.transcribe("temp.wav")
            user_input = result["text"]
            st.success("Transcribed text:")
            st.write(user_input)
        except:
            st.warning("Whisper failed or unavailable. Falling back to SpeechRecognition...")
            recognizer = sr.Recognizer()
            with sr.AudioFile("temp.wav") as source:
                audio = recognizer.record(source)
                try:
                    user_input = recognizer.recognize_google(audio)
                    st.success("Transcribed text:")
                    st.write(user_input)
                except:
                    st.error("Could not transcribe audio.")

def analyze_emotion(text):
    # Dummy logic for testing — replace with real model later
    text = text.lower()
    if "sad" in text or "down" in text:
        return "negative", "sad"
    elif "angry" in text or "mad" in text:
        return "negative", "angry"
    elif "anxious" in text or "nervous" in text:
        return "negative", "anxious"
    elif "happy" in text or "joy" in text:
        return "positive", "happy"
    else:
        return "neutral", "neutral"

# Default emotion
emotion = "neutral"  # Make sure it's initialized
detected_emotion = "neutral"  # Initialize this too in case it's used

if user_input:
    # Suppose this function returns something like: ("negative", "sad")
    emotion, detected_emotion = analyze_emotion(user_input)


# --- Analyze and Respond ---
if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter or say how you're feeling.")
    else:
        emotion = predict_emotion_local(user_input)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        st.session_state.history.append({
            "time": timestamp,
            "input": user_input,
            "emotion": emotion
        })

        st.subheader("🔍 Detected Emotion:")
        st.success(emotion.capitalize())

        st.subheader("🛠 Coping Strategies:")
        for tip in coping_strategies[emotion]:
            st.markdown(f"- {tip}")

        st.subheader("👩‍⚕️ Therapist Resources:")
        for link in therapists:
            st.markdown(f"- [{link.split(':')[0]}]({link.split(': ')[1]})")

        st.subheader("🌿 Wellness Suggestions:")
        if emotion == "negative":
            st.markdown("- Try 4-7-8 breathing technique")
            st.markdown("- Listen to calming music")
        if detected_emotion == "sad":
             st.markdown("- Try the 4-7-8 breathing technique")
             st.markdown("- Go for a short walk outside")
             st.markdown("- Write down 3 things you're grateful for")
        elif detected_emotion == "angry":
             st.markdown("- Practice deep breathing or progressive muscle relaxation")
             st.markdown("- Take a break and do something physical like jogging or push-ups")
             st.markdown("- Try journaling to release emotions")
        elif detected_emotion == "anxious":
             st.markdown("- Listen to calming music or nature sounds")
             st.markdown("- Use grounding techniques (like 5-4-3-2-1 sensory awareness)")
             st.markdown("- Avoid caffeine and try herbal tea")
        else:
             st.markdown("- Try general mindfulness meditation")
             st.markdown("- Talk to a trusted friend or use a journal to vent")
        if emotion == "neutral":
             st.markdown("- Practice yoga or light stretching")
             st.markdown("- Maintain a routine for consistency")
             st.markdown("- Reflect on positive moments from the day")
        elif emotion == "positive":
            if detected_emotion == "happy":
                st.markdown("- Share your happiness with someone you care about")
                st.markdown("- Capture the moment by journaling or taking a photo")
                st.markdown("- Do more of what brought you joy today")
            elif detected_emotion == "grateful":
                st.markdown("- Express your gratitude to someone")
                st.markdown("- Reflect on how your day unfolded positively")
                st.markdown("- Consider writing a thank-you note or message")
            elif detected_emotion == "excited":
                st.markdown("- Channel your energy into a creative project")
                st.markdown("- Plan something fun or productive with your momentum")
                st.markdown("- Share your excitement with friends or family")
            else:
                st.markdown("- Enjoy and savor the moment")
                st.markdown("- Reinforce positive habits that led to this feeling")
                st.markdown("- Encourage someone else today with a kind gesture")

# --- Chat History Display ---
if st.session_state.history:
    st.markdown("---")
    st.subheader("🕒 Chat History")
    df = pd.DataFrame(st.session_state.history)
    for entry in reversed(st.session_state.history):
        st.markdown(f"**{entry['time']}** - *{entry['emotion'].capitalize()}*: {entry['input']}")

    # --- Export Options ---
    st.markdown("### 📁 Export Chat History")
    col1, col2, col3 = st.columns(3)

    with col1:
        txt = "\n".join([f"{x['time']} - {x['emotion'].capitalize()}: {x['input']}" for x in st.session_state.history])
        st.download_button("Download TXT", txt, file_name="chat_history.txt")

    with col2:
        csv = df.to_csv(index=False)
        st.download_button("Download CSV", csv, file_name="chat_history.csv")

    # PDF Creation
    with open("file.txt", "w") as f:
        pdf = FPDF()  # ✅ Proper indentation
        pdf.add_page()

        pdf.set_font("Arial", size=12)
        for row in st.session_state.history:
            line = f"{row['time']} - {row['emotion'].capitalize()}: {row['input']}"
            pdf.cell(200, 10, txt=line, ln=True)

    # Get PDF as bytes
    pdf_bytes = pdf.output(dest='S').encode('latin1')  # Returns a str, so encode it

    # Convert to BytesIO
    pdf_buffer = BytesIO(pdf_bytes)
    st.download_button("Download PDF", data=pdf_buffer, file_name="chat_history.pdf", mime="application/pdf")

st.markdown("---")
st.caption("This tool is not a substitute for professional mental health care.")
