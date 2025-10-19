import os
import subprocess

while True:
    print("\n===== MAIN MENU =====")
    print("1. Guidance Detection Mode")
    print("2. AI Conversation Mode")
    print("3. OCR Text Read Mode")
    print("4. Exit")
    choice = input("Select an option: ")

    if choice == "1":
        subprocess.run(["python3", "guidance_mode.py"])
    elif choice == "2":
        subprocess.run(["python3", "conversation_mode.py"])
    elif choice == "3":
        subprocess.run(["python3", "ocr_mode.py"])
    elif choice == "4":
        print("Exiting...")
        break
    else:
        print("Invalid choice! Please select 1-4.")
