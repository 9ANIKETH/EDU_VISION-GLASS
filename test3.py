import speech_recognition as sr

r = sr.Recognizer()
with sr.Microphone() as source:
    print("🎤 Say something...")
    audio = r.listen(source)

try:
    text = r.recognize_google(audio, language="eng-BD")
    print("✅ You said:", text)
except Exception as e:
    print("❌ Error:", e)
