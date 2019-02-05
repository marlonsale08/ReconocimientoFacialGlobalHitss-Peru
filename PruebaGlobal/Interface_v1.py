'''
    SISTEMA DE RECONOCIMIENTO FACIAL HITSS-LIMA-PERU
        Borrar cada 5 minutos
        Tiempo : old = 5 * 60 s
'''
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import smtplib
import xlwt
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
pygame.init()

global frame_captured, is_exit, video_capture, current_hour

old = 5 * 60
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
#global video_capture 
#video_capture = cv2.VideoCapture(1)

'''def capture():
    global frame_captured, is_exit
    while not is_exit:
        # Grab a single frame of video
        ret, frame_captured = video_capture.read()
'''
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

class Powerup(pygame.sprite.Sprite):
    
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image =pygame.image.load('./GlobalHitss.jpg')
        #self.image=pygame.transform.scale(self.image,(1320,800))
        self.rect = self.image.get_rect()
 
    def update(self, window):
        # AQUÍ INCLUIREMOS EL CÓDIGO PARA ANIMAR NUESTRA HOJA DE SPRITES
        pass
     
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

if __name__ == "__main__":

    window = pygame.display.set_mode([1320,800])
    pygame.display.set_caption('SISTEMA DE RECONOCIMIENTO FACIAL GLOBAL HITSS PERU')
    clock = pygame.time.Clock()
    
    powerup = Powerup()
    powerup_group = pygame.sprite.GroupSingle()
    powerup_group.add(powerup)
    #imagen=pygame.image.load('./test.jpeg')
    #imagen=pygame.transform.scale(imagen,(150,10))
    logo2=pygame.image.load('logoGlobal.png')
    imagen=pygame.image.load('./botonBlancoHD.png')
    dashboardEntrada=pygame.image.load('./fondo2.jpg')
    dashboardEntrada=pygame.transform.scale(dashboardEntrada,(400,585))
    dashboardEntrada2=pygame.image.load('./fondo2.jpg')
    dashboardEntrada2=pygame.transform.scale(dashboardEntrada,(250,680))
    fondoPrincipal=pygame.image.load('./fondoPrincipal.jpg')
    dashboard1=pygame.transform.rotate(dashboardEntrada,90)
    dashboard1=pygame.transform.scale(dashboard1,(350,690))
    #dashboard1=pygame.transform.rotozoom(dashboard1,180,1)
    dashboard2=pygame.transform.rotate(imagen,270)
    dashboard2=pygame.transform.scale(dashboard2,(250,690))
    #dashboard2=pygame.transform.rotozoom(dashboard2,180,1)
    dashboard3=pygame.transform.scale(dashboard2,(250,690))
    dashboardPrincipal=pygame.transform.rotate(imagen,90)
    dashboardPrincipal=pygame.transform.scale(dashboardPrincipal,(400,300))
    imagen2=pygame.image.load('./logo.png')
    logoPrincipal=pygame.image.load('./logoGlobal.png')
    #logoPrincipal=pygame.image.scale(logoPrincipal,)
    imagen3=pygame.image.load('./waiting.png')
    imagen3=pygame.transform.scale(imagen3,(80,80))
    icono=pygame.image.load('./calendario.png')
    #ReemplazaColor(icono,(0,0,0),(0,0,0),0.7)
    icono=pygame.transform.scale(icono,(30,30))
    icono3=pygame.image.load('./buscador.png')
    icono3=pygame.transform.scale(icono3,(30,30))
    icono4=pygame.image.load('./enviar_correo.png')
    icono4=pygame.transform.scale(icono4,(30,30))
    icono5=pygame.image.load('./rompecabezas.png')
    #ReemplazaColor(icono5,(0,0,0),(0,0,0),0.7)
    icono5=pygame.transform.scale(icono5,(30,30))
    icono6=pygame.image.load('./icono6.png')
    icono6=pygame.transform.scale(icono6,(30,30))
    icono7=pygame.image.load('./icono7.png')
    icono7=pygame.transform.scale(icono7,(30,30))
    icono8=pygame.image.load('./icono8.png')
    icono8=pygame.transform.scale(icono8,(30,30))
    icono9=pygame.image.load('./icono9.png')
    icono9=pygame.transform.scale(icono9,(30,30))
    boton=pygame.image.load('./botonBlancoHD.png')
    boton=pygame.transform.scale(boton,(250,100))
    #ReemplazaColor(boton,(0,0,0),(0,0,0),0.7)
    timer=pygame.image.load('./timer.png')
    timer=pygame.transform.scale(timer,(30,30))
    print("CONTROL1")
    print(type(timer))
    #ReemplazaColor(timer,(0,0,0),(255,255,255))
    #window.blit(imagen,(25,25))
    #window.blit(imagen,(225,225))
    #window.blit(imagen,(425,425))
    #152,142,140
    pygame.display.update()
    fuente = pygame.font.Font(None,18)
    texto1 = fuente.render("SISTEMA DE RECONOCIMIENTO FACIAL", 0, (255,255,255))
    fuente2 = pygame.font.Font(None,10)
    texto2 = fuente.render("Team Hitss Lab", 0, (255,255,255)) 
    now = time.time()
    origin = "ClasificadorKNN/train"
    dest = "ClasificadorKNN/invitados/"

    video_capture = cv2.VideoCapture(0)
    video_capture.set(3,1920)
    video_capture.set(4,1080)
    is_exit = False

    '''capture_thread = threading.Thread(target=capture, name='captureThread', daemon=True)
    capture_thread.start()
    capture_thread.join() 
    #frame_captured = None
    if threading.Thread(target=capture, name='captureThread', daemon=True):
        print("matando hilos...")
        threadError = threading.Thread(target=capture, name='captureThread', daemon=True)
        threadError._stop()''' 
    Known_face_personalHitss = []

    Known_face_bandera = []
    
    face_locations = []
    face_encodings = []
    face_names = []
    
    process_this_frame = True

    guessID = 10
        
    getFiles(origin)

    style1 = xlwt.easyxf(num_format_str='D-MMM-YY-HH')
    wb = xlwt.Workbook()
    ws = wb.add_sheet('A Test Sheet')
        
    #print("Entrenando Clasificador KNN")
    #classifier = train("ClasificadorKNN/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
    #print("Entrenamiento Completo!")
    i=0    
    while True:
        pygame.display.update()
        powerup.update(window)
        powerup_group.draw(window)
        #pygame.display.flip()
        '''window.blit(dashboard2,(10,10))
        window.blit(dashboard2,(250,10))
        window.blit(dashboard3,(1090,60))
        window.blit(texto1, (15,20))
        window.blit(texto2, (1150,680))
        window.blit(logo2,(1150,5))'''
        #window.blit(fondoPrincipal,)
        window.blit(logoPrincipal,(153,20))
        #window.blit(dashboardPrincipal,(87,209))
        window.blit(dashboardEntrada,(27,95))
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x_mouse,y_mouse = pygame.mouse.get_pos()
                print(x_mouse)
                print(y_mouse)
                if (x_mouse<=100 or x_mouse>=10) and (y_mouse<=100 or y_mouse>=10):
                    
                    print("BOTON MACHUCADO")
                    while True:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                pygame.quit()
                                sys.exit() 
                        #window.kill()
                        pygame.display.update()
                        powerup.update(window)
                        powerup_group.draw(window)
                        window.blit(dashboardEntrada2,(10,10))
                        window.blit(dashboard2,(280,10))
                        window.blit(dashboard2,(1090,10))
                        window.blit(texto1, (15,20))
                        window.blit(texto2, (1150,680))
                        window.blit(logo2,(1150,5))
                        if old - ((time.time() - now) % old) > old - 1:
                            removeGuess(dest)

                        ret, frame_captured = video_capture.read()  

                        frame = frame_captured
                        frame2 =frame
                        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                        
                        rgb_small_frame = small_frame[:, :, ::-1]

                        X_img = rgb_small_frame
                        
                        if process_this_frame:
                            face_locations = face_recognition.face_locations(X_img)#,number_of_times_upsample=2)
                            face_encodings = face_recognition.face_encodings(X_img)

                            predictions = []

                            face_names = []    

                            #for face_encoding in face_encodings:
                            for index, face_encoding in enumerate(face_encodings, start=0):
                                #name = "Desconocido"
                                predictions = predict(X_img, model_path="trained_knn_model.clf",distance_threshold=0.44)
                                print(predictions)
                                print("     "+str(datetime.now()))
                                face_names.append(str(predictions[index][0]))
                                
                                if predictions:
                                    faceID = Known_face_personalHitss.index(str(predictions[index][0])) if (predictions[index][0]!='Desconocido') else -1
                                    #print(index,"   -------predictions",str(predictions[index][0]))
                                    #name=str(predictions[i][0])
                                    if faceID != -1:
                                        
                                        if Known_face_bandera[faceID] == 0:
                                            Known_face_bandera[faceID] = 1
                                            voiceSpeech(str(predictions[index][0]))
                                            #voiceSpeech(str(predictions[index][0]))
                                            ws.write(i,0,datetime.now(),style1)
                                            ws.write(i,1,str(predictions[index][0]))
                                            print("      SALUDO")
                                            print("      "+str(datetime.now()))
                                            wb.save('Registro de Personal.xls')    
                                            #print("CONTROL CONOCIDO")
                                            i+=1
                                    else:
                                        #X_img=X_img-18
                                        '''ImagenEnviar=Image.fromarray(X_img)
                                        ImagenEnviar.save("ImagenEnviar.jpg")
                                        fp=open('ImagenEnviar.jpg','rb')
                                        msgImage=MIMEImage(fp.read())
                                        EnviarMensajeFoto(msgImage,"marlonsale08@gmail.com","marlonsale08@gmail.com","Posible persona desconocida en Hitss")'''
                                        '''compare_guess = predictionsGuess(X_img, dest)
                                        #print("compare guess: ", compare_guess)
                                        start_time=time.time()
                                        #with concurrent.futures.ThreadPoolExecutor() as executor:
                                            #compare_guess=executor.submit(predictionsGuess,X_img,dest)
                                            #print(compare_guess.running())
                                        if compare_guess.result() == False:
                                            Known_face_bandera.append(-1)
                                            Known_face_personalHitss.append('Invitado ' + str(guessID))
                                            cv2.imwrite(dest + "invitado%d.jpg" % guessID, frame) 
                                            voiceSpeech('Por favor identifiquese con la recepcionista')                   
                                               # guessID = guessID + 1
                                        end_time=time.time()
                                        #compare_guess.cancel()
                                        #print("Llamada"+str(compare_guess.cancel()))
                                        print(end_time-start_time)
                                        #full_file_path = os.path.join(imgFile)
                                        #print("RUTA: ", full_file_path)
                                        #src = full_file_path'''
                                        #shutil.move(src, dest)

                        process_this_frame= not process_this_frame
                            
                            #for name, (top, right, bottom, left) in predictions:
                                #print("- Encontrado  {} en ({}, {})".format(name, left, top))

                            #show_prediction_labels_on_image(os.path.join("ClasificadorKNN/test", X_img), predictions)
                        #video = pygame.pixelcopy.make_surface(frame)
                        
                        for (top, right, bottom, left), name in zip(face_locations, face_names):
                            if name=="Desconocido":
                                
                                color= (255,0,0)
                            else:
                                color=(0,0,255)
                            
                            top *= 4
                            right *= 4
                            bottom *= 4
                            left *= 4

                            #print("FRAMES...")
                            #print(frame)
                            #print("--------------------------------------------------------------")

                            cv2.rectangle(frame2, (left, top), (right, bottom), color, 2)
                            cv2.rectangle(frame2, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                            font = cv2.FONT_HERSHEY_TRIPLEX
                            cv2.putText(frame2, name, (left +20, bottom -6), font, 0.75, (255, 255, 255), 1)

                        video = pygame.pixelcopy.make_surface(frame2)
                        video = pygame.transform.flip(video,False,True)
                        #video=pygame.tranform.scale2x(video)
                        video=pygame.transform.rotate(video,-90)
                        video=pygame.transform.scale(video,(450,330))
                        window.blit(video,(600,10))
                        pygame.display.update()

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            is_exit = True
                            break
                    video_capture.release()
                    cv2.destroyAllWindows()
            elif event.type == pygame.KEYDOWN:
                print("SE PRESIONO UNA TECLA")
            #if event.type==pygame.
        #window.blit(icono,(10,10))
        #window.blit(icono2,(10,10))
        #window.blit(icono3,(20,50))
        #window.blit(icono4,(20,150))
        #window.blit(icono5,(20,250))
        #window.blit(boton,(20,50))
        #time.sleep(old - ((time.time() - now) % 300.0))
        '''if old - ((time.time() - now) % old) > old - 1:
            removeGuess(dest)

        ret, frame_captured = video_capture.read()  

        frame = frame_captured
        
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        rgb_small_frame = small_frame[:, :, ::-1]

        X_img = rgb_small_frame
        
        if process_this_frame:
            face_locations = face_recognition.face_locations(X_img)#,number_of_times_upsample=2)
            face_encodings = face_recognition.face_encodings(X_img)

            predictions = []

            face_names = []    

            #for face_encoding in face_encodings:
            for index, face_encoding in enumerate(face_encodings, start=0):
                #name = "Desconocido"
                predictions = predict(X_img, model_path="trained_knn_model.clf",distance_threshold=0.44)
                print(predictions)
                print("     "+str(datetime.now()))
                face_names.append(str(predictions[index][0]))
                
                if predictions:
                    faceID = Known_face_personalHitss.index(str(predictions[index][0])) if (predictions[index][0]!='Desconocido') else -1
                    #print(index,"   -------predictions",str(predictions[index][0]))
                    #name=str(predictions[i][0])
                    if faceID != -1:
                        
                        if Known_face_bandera[faceID] == 0:
                            Known_face_bandera[faceID] = 1
                            voiceSpeech(str(predictions[index][0]))
                            #voiceSpeech(str(predictions[index][0]))
                            ws.write(i,0,datetime.now(),style1)
                            ws.write(i,1,str(predictions[index][0]))
                            print("      SALUDO")
                            print("      "+str(datetime.now()))
                            wb.save('Registro de Personal.xls')    
                            #print("CONTROL CONOCIDO")
                            i+=1
                    else:
                        
                        compare_guess = predictionsGuess(X_img, dest)
                        #print("compare guess: ", compare_guess)
                        start_time=time.time()
                        #with concurrent.futures.ThreadPoolExecutor() as executor:
                            #compare_guess=executor.submit(predictionsGuess,X_img,dest)
                            #print(compare_guess.running())
                        if compare_guess.result() == False:
                            Known_face_bandera.append(-1)
                            Known_face_personalHitss.append('Invitado ' + str(guessID))
                            cv2.imwrite(dest + "invitado%d.jpg" % guessID, frame) 
                            voiceSpeech('Por favor identifiquese con la recepcionista')                   
                               # guessID = guessID + 1
                        end_time=time.time()
                        #compare_guess.cancel()
                        #print("Llamada"+str(compare_guess.cancel()))
                        print(end_time-start_time)
                        #full_file_path = os.path.join(imgFile)
                        #print("RUTA: ", full_file_path)
                        #src = full_file_path
                        #shutil.move(src, dest)

        process_this_frame= not process_this_frame
            
            #for name, (top, right, bottom, left) in predictions:
                #print("- Encontrado  {} en ({}, {})".format(name, left, top))

            #show_prediction_labels_on_image(os.path.join("ClasificadorKNN/test", X_img), predictions)
        
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            #print("FRAMES...")
            #print(frame)
            #print("--------------------------------------------------------------")

            cv2.rectangle(frame, (left, top), (right, bottom), (255,0,0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255,0,0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame)
        video = pygame.pixelcopy.make_surface(X_img)
        #video=pygame.tranform.scale2x(video)
        video=pygame.transform.rotozoom(video,-90,1)
        window.blit(video,(630,100))
        pygame.display.update()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            is_exit = True
            break
    video_capture.release()
    cv2.destroyAllWindows()'''
    