from gtts import gTTS
import pygame
import pyttsx3
from datetime import datetime
import boto3
import time

def voiceSpeech(nameText):
    origin="ClasificadorKNN/dni_colaborador/"+nameText+".text"
    f=open(origin,"r")
    nameText=f.read()
    print(nameText)
    f.close()
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

def voiceSpeechA(nameText=None):
    message=greetingTime()
    polly_client = boto3.Session(
                    aws_access_key_id="AKIAI5RAYLJOXPW2BFGQ",                     
        aws_secret_access_key="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",
        region_name='sa-east-1').client('polly')

    response = polly_client.synthesize_speech(VoiceId='Enrique',
                    OutputFormat='mp3', 
                    Text =message + ' ' + nameText )
    file = open('BD.mp3', 'wb')
    file.write(response['AudioStream'].read())
    file.close()
    pygame.mixer.init()
    pygame.mixer.music.load("BD.mp3")
    pygame.mixer.music.play()

def greetingTime():

    current_hour = datetime.now().hour
    if current_hour < 12:
        return "Buenos días"
    elif 12 <= current_hour < 18:
        return "Buenas tardes"
    else:
        return "Buenas noches"