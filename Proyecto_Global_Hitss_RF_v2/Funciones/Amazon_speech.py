import boto3
import time
import pygame

def speech_amazon():

	polly_client = boto3.Session(
					aws_access_key_id="AKIAI5RAYLJOXPW2BFGQ",                     
		aws_secret_access_key="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",
		region_name='sa-east-1').client('polly')

	response = polly_client.synthesize_speech(VoiceId='Enrique',TextType='SSML',
					OutputFormat='mp3', 
					Text = '<speak> <prosody rate="110%"> Buenos Días a llamado a claro empresas ,en que puedo ayudarlo?</prosody> </speak>')
	file = open('speech.mp3', 'wb')
	file.write(response['AudioStream'].read())
	file.close()


