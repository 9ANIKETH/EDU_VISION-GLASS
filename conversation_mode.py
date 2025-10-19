import speech_recognition as sr
import google.generativeai as genai
import multiprocessing as mp

API_KEY = "AIzaSyApKrl1R6prVOvVHenCiPYFYnaR2lSUULU"
genai.configure(api_key=API_KEY)
model_ai = genai.GenerativeModel('gemini-1.5-flash')
convo = model_ai.start_chat()
convo.send_message("You are a short and clear voice assistant.")

recognizer = sr.Recognizer()
stop_word = "exit"

def tts_worker(queue):
    import pyttsx3
    engine = pyttsx3.init(driverName='espeak')
    engine.setProperty('rate', 160)
    engine.setProperty('volume', 1.0)
    while True:
        text = queue.get()
        if text == "STOP":
            break
        engine.say(text)
        engine.runAndWait()

tts_queue = mp.Queue()
tts_process = mp.Process(target=tts_worker, args=(tts_queue,), daemon=True)
tts_process.start()

def listen_microphone(timeout=5):
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=timeout)
            return recognizer.recognize_google(audio).lower()
        except:
            return ""

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

if __name__ == "__main__":
    conversation_loop()
    tts_queue.put("STOP")
