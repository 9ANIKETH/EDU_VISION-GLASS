import os
from banglatts import BanglaTTS

tts = BanglaTTS(save_location="save_model_location")
path = tts("আমি বাংলায় কথা বলতে পারি।", voice='female', filename='1.wav')

os.system(f"aplay {path}")   # ALSA ব্যবহার করে play হবে
