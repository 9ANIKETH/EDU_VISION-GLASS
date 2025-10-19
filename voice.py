# voice.py
from banglatts import BanglaTTS
from pydub import AudioSegment
from pydub.playback import play
import os

# Initialize Bangla TTS
tts = BanglaTTS(save_location=".")

def speak(text, voice="male"):
    try:
        # Create speech file
        path = tts(text, voice=voice, filename="response.wav")

        # Load and play audio
        audio = AudioSegment.from_wav(path)
        play(audio)

        # Optionally delete file after play
        if os.path.exists(path):
            os.remove(path)

    except Exception as e:
        print(f"❌ Error in speak(): {e}")
