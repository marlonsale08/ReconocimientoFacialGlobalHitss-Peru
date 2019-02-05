from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import smtplib
import xlwt
import wx
from sys import stdout
from datetime import datetime
import math
from sklearn import neighbors
import os
from time import sleep
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
from tempfile import TemporaryFile
import time
import cv2
import threading
import pyttsx3
from gtts import gTTS
import pygame
import shutil
import concurrent.futures
import pygame, sys
from Clases import *
from Funciones import *
#from tk-tools import *
pygame.init()

global frame_captured, is_exit, video_capture, current_hour



def EnviarMensajeTexto(mensaje=None,correoFrom=None,correoTo=None,subject=None):
    # crear el objeto del mensaje
    msg = MIMEMultipart()

    # configurar mensaje
    password = "vergaramarlon12"
    msg['From'] = correoFrom
    msg['To'] = correoTo
    msg['Subject'] = subject

    #Cuerpo del mensaje
    msg.attach(MIMEText(mensaje, 'plain'))

    #Crear server
    server = smtplib.SMTP('smtp.gmail.com: 587')

    server.starttls()

    #login server
    server.login(msg['From'], password)

    #enviar mensaje
    server.sendmail(msg['From'], msg['To'], msg.as_string())

    server.quit()

    print ("Envio de correo exitoso %s:" % (msg['To']))

def EnviarMensajeFoto(msgImage=None,correoFrom=None,correoTo=None,subject=None):

    msg = MIMEMultipart()
    password = "vergaramarlon12"
    msg['From'] = correoFrom
    msg['To'] = correoTo
    msg['Subject'] = subject

    # add in the message body
    #fp = open('test.jpeg', 'rb')
    #msgImage = MIMEImage(fp.read())
    #fp.close()
    #msg.attach(msgImage)
    #fp.close()  
    #msgImage=MIMEImage(img)
    msg.attach(msgImage)
    server = smtplib.SMTP('smtp.gmail.com: 587')
    server.starttls()
    server.login(msg['From'], password)
    server.sendmail(msg['From'], msg['To'], msg.as_string())
    server.quit()
    print ("Envio de correo exitoso %s:" % (msg['To']))

def zoom(frame=None,mirror=False,cam=None,scale=None):
    
    #scale=10
    #cam = cv2.VideoCapture(0)

    ret_val, frame = cam.read()
    if mirror: 
        frame = cv2.flip(frame, 1)
        #get the webcam size
    height, width, channels = frame.shape
        #prepare the crop
    centerX,centerY=int(height/2),int(width/2)
    radiusX,radiusY= int(scale*height/100),int(scale*width/100)

    minX,maxX=centerX-radiusX,centerX+radiusX
    minY,maxY=centerY-radiusY,centerY+radiusY

    cropped = frame[minX:maxX, minY:maxY]
    resized_cropped = cv2.resize(cropped, (width, height))
    small_frame = cv2.resize(resized_cropped, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    return rgb_small_frame

def greetingTime():
    current_hour = datetime.now().hour
    if current_hour < 12:
        return "Buenos días"
    elif 12 <= current_hour < 18:
        return "Buenas tardes"
    else:
        return "Buenas noches"

def getFiles(origin):
    files = os.listdir(origin)
    for name in files:
        full_path = os.path.join(origin, name)
        if os.path.isdir(full_path):
            Known_face_personalHitss.append(name)
            Known_face_bandera.append(0)

def voiceSpeech(nameText):
    message = greetingTime()
    tts = gTTS(message + ' ' + nameText + '...','es-es')
    tts.save('BD.mp3')
    pygame.mixer.init()
    pygame.mixer.music.load("BD.mp3")
    pygame.mixer.music.play()

def voiceSpeechL(nameText):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[20].id + '+f3')
    engine.say("Buenos días, "+ nameText+"...")
    engine.setProperty('rate',170)
    engine.setProperty('volume', 0.9)
    engine.runAndWait()

def removeGuess(origin):
    now = time.time()
    files = os.listdir(origin)
    for name in files:
        full_path = os.path.join(origin, name)
        if os.path.isfile(full_path):
            stat = os.stat(full_path)
            if now - stat.st_ctime > old:
                os.remove(full_path)

def predictionsGuess(X_img, origin):
    final_result = False
    if os.path.isdir(origin):
        for index, img_path in enumerate(image_files_in_folder(origin), start = 0):
            #print ("Index: ", index)
            image = face_recognition.load_image_file(img_path)
            results = False
            if len(face_recognition.face_encodings(image)) > 0:
                saved_image = face_recognition.face_encodings(image)[0]
                captured_image = face_recognition.face_encodings(X_img)[0]
                results = face_recognition.compare_faces([saved_image], captured_image, tolerance=0.6)
                #print("Entrando al len FR: ", results)
                if results[0] == True:
                    #print("Nuevo resultado: ", re0sults)
                    final_result = results[0]
                    break
    #print("resultados: ", final_result)
    return final_result

def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    
    X = []
    y = []

    # Recorre a través de cada persona en el conjunto de entrenamiento
    for class_dir in os.listdir(train_dir):
                                                                                                                                    
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Recorre cada imagen de entrenamiento para la persona actual
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)
                     
            if len(face_bounding_boxes) != 1:
                
                # Filtro de muchas caras
                if verbose:
                    print("Imagen {} no es recomendable para el entrenamiento: {}".format(img_path, "No se encontraron caras" if len(face_bounding_boxes) < 1 else "Muchas caras encontradas"))
            else:
                # Codifica imagen actual del conjunto de entrenamiento
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)
                
    # Calcula numero de vecinos
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Numero de vecinos encontrados:", n_neighbors)

    # Crea y entrena el KNN
    
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')   
    knn_clf.fit(X, y)


    # Guarda el kernel generado
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf

def predict(X_img, knn_clf=None, model_path=None, distance_threshold=0.6):
       
    #Validacion de extension 
    #if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
       # raise Exception("Formato de imagen no admitido: {}".format(X_img_path))
    #validacion de ruta de clasificador ya entrenado
    if knn_clf is None and model_path is None:
        raise Exception("Debe proporcionar el archivo del clasificador o la ruta ")

    # carga un modelo entrenado
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # carga archivo de imagen 
    #X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # Devuelve cero si no hay caras
    if len(X_face_locations) == 0:
        return []
    # Encontrar codificaciones para caras en el image.
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Utilizamos el knn en nuestro conjunto definido para la cara de prueba
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    #print(closest_distances)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predecir clases y eliminar clasificaciones que no están dentro del umbral
    return [(pred, loc) if rec else ("Desconocido", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

def show_prediction_labels_on_image(img_path, predictions):
       
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # Dibuja un cuadro
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
       
        name = name.encode("UTF-8")
        # Dibuja una etiqueta con un nombre debajo de la cara
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Elimina la biblioteca de dibujos de memoria RAM
    del draw
    # Muestra la imagen resultante
    pil_image.show()

def ReemplazaColor(imagen, color, recolor, distancia = 0):
    #y asi de simple se remplaza un color :)
    #la funcion resive como argumento la imagen
    #el color que queremos remplazar
    #y con que color lo queremos remplazar
    #distancia es para los colores que,
    #sin ser en RGB iguales, al ojo lucen exactamente igual.
    pixel_array = pygame.PixelArray(imagen) #la traformamos en  arreglo
    #por suerte, pygame inclulle una funcion que lo hace XD
    pixel_array.replace(color, recolor, distancia)
    pixel_array.make_surface() #lo convertimos en imagen