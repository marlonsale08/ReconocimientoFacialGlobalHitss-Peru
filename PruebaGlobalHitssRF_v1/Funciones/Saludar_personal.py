from gtts import gTTS
import pygame
import pyttsx3
from datetime import datetime

def voiceSpeech(nameText):
    start_time=datetime.now()
    message = greetingTime()
    tts = gTTS(message + ' ' + nameText + '...','es-es')
    tts.save('BD.mp3')
    pygame.mixer.init()
    pygame.mixer.music.load("BD.mp3")
    pygame.mixer.music.play()
    end_time=datetime.now()
    print("TIEMPO")
    print(end_time-start_time)

def voiceSpeechL(nameText):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[20].id + '+f3')
    engine.say("Buenos días, "+ nameText+"...")
    engine.setProperty('rate',170)
    engine.setProperty('volume', 0.9)
    engine.runAndWait()

def greetingTime():
    current_hour = datetime.now().hour
    if current_hour < 12:
        return "Buenos días"
    elif 12 <= current_hour < 18:
        return "Buenas tardes"
    else:
        return "Buenas noches"