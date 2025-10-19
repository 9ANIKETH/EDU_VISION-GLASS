# ---------------- IMPORTS ----------------
import google.generativeai as genai
import os
import threading
import speech_recognition as sr
import pyttsx3
import queue
import time

# ---------------- TTS ENGINE ----------------
engine = pyttsx3.init(driverName='espeak')  # Use 'espeak' for Linux
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

# ---------------- TTS QUEUE ----------------
tts_queue = queue.Queue()

def tts_loop():
    """Continuously speak text from the TTS queue."""
    while True:
        text = tts_queue.get()
        if text is None:  # None is signal to exit
            break
        engine.say(text)
        engine.runAndWait()

# Start TTS thread
tts_thread = threading.Thread(target=tts_loop, daemon=True)
tts_thread.start()

# ---------------- GEMINI CONFIG ----------------
API_KEY = "AIzaSyC5IbIs34jvKMMf7yHMCk_5ToW3ijwL_P8"  # Replace with your API key
genai.configure(api_key=API_KEY)

generation_config = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 1,
    "max_output_tokens": 100,
}

try:
    model = genai.GenerativeModel('gemini-2.5-flash-lite', generation_config=generation_config)
    convo = model.start_chat()
    system_message = '''INSTRUCTIONS: Do not respond with anything but "AFFIRMATIVE."
    SYSTEM MESSAGE: You are being used to power a voice assistant and should respond as such.
    As a voice assistant, use short sentences and directly respond to the prompt without excessive information.'''
    convo.send_message(system_message.replace('\n',''))
except Exception as e:
    print(f"Error initializing Generative AI model: {e}")
    exit(1)

stop_word = "exit"

# ---------------- SPEECH RECOGNITION ----------------
recognizer = sr.Recognizer()

def listen_microphone(timeout=5):
    """Listen to USB microphone and return lowercase text."""
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        try:
            print("Listening...")
            audio = recognizer.listen(source, timeout=timeout)
            text = recognizer.recognize_google(audio)
            print(f"User said: {text}")
            return text.lower()
        except sr.WaitTimeoutError:
            return ""
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            return ""

# ---------------- CONVERSATION THREAD ----------------
def conversation_loop():
    tts_queue.put("Conversation mode on. Say something.")

    while True:
        tts_queue.put("Listening...")
        user_input = listen_microphone()
        
        if not user_input:
            continue

        if stop_word in user_input:
            tts_queue.put("Exiting conversation mode")
            break

        convo.send_message(user_input)
        response = convo.last.text

        if response:
            tts_queue.put(response)

# Start conversation thread
conversation_thread = threading.Thread(target=conversation_loop, daemon=True)
conversation_thread.start()

# ---------------- MAIN LOOP ----------------
try:
    while True:
        time.sleep(1)  # Keep main thread alive without busy-waiting
except KeyboardInterrupt:
    print("Stopping main program...")
    tts_queue.put(None)  # Signal TTS thread to exit
    tts_thread.join()
