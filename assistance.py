import google.generativeai as genai
import sys
from voice import speak   # ✅ अब ये BanglaTTS वाली voice.py use करेगा

def aiProcess():
    API_KEY = "AIzaSyApKrl1R6prVOvVHenCiPYFYnaR2lSUULU"  # ⚠️ अपना API Key डालो
    genai.configure(api_key=API_KEY)

    generation_config = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 1,
        "max_output_tokens": 1024,
    }

    try:
        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            generation_config=generation_config
        )
        convo = model.start_chat()

        system_message = (
            "SYSTEM MESSAGE: You are a helpful Bangla voice assistant. "
            "Always respond in Bangla with short, clear sentences. "
        )
        convo.send_message(system_message)

        print("🤖 Bangla Voice Assistant Initialized. Type 'exit' to quit.\n")

    except Exception as e:
        print(f"❌ Error initializing Generative AI model: {e}")
        return

    stop_word = "exit"

    try:
        while True:
            user_input = input("📝 You: ").strip()

            if stop_word in user_input.lower():
                exit_msg = "ম্যাট্রিক্স বন্ধ করা হচ্ছে এবং বের হওয়া হচ্ছে..."
                print(f"🤖 AI: {exit_msg}")
                speak(exit_msg)
                break

            convo.send_message(user_input)
            response = convo.last.text.strip()

            print(f"🤖 AI: {response}")
            speak(response, voice="male")  # ✅ BanglaTTS से बोलेगा

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Exiting...")

    finally:
        sys.exit(0)


if __name__ == "__main__":
    aiProcess()
