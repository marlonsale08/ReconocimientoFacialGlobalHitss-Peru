import math
from sklearn import neighbors
import os
from time import sleep
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
#import pygame
from tempfile import TemporaryFile
import time
import cv2
import threading
from gtts import gTTS
import pygame
import shutil

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
#global video_capture 
#video_capture = cv2.VideoCapture(1)
global frame, is_exit, video_capture

def voiceSpeech(nameText):
    tts = gTTS('Buenos días, '+ nameText+'...','es-es')
    tts.save('BD.mp3')
    pygame.mixer.init()
    pygame.mixer.music.load("BD.mp3")
    pygame.mixer.music.play()

def capture():
    while not is_exit:
        # Grab a single frame of video
        ret, frame = video_capture.read()

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

if __name__ == "__main__":
    origin = "ClasificadorKNN/train"
    dest = "ClasificadorKNN/invitados/"

    video_capture = cv2.VideoCapture(0)

    Known_face_personalHitss=[]

    Known_face_bandera=[]
    
    face_locations = []
    face_encodings = []
    face_names = []
    
    process_this_frame = True

    guessID = 1

    files = os.listdir(origin)
    for name in files:
        full_path = os.path.join(origin, name)
        if os.path.isdir(full_path):
            Known_face_personalHitss.append(name)
            Known_face_bandera.append(0)


    #print("Entrenando Clasificador KNN")
    #classifier = train("ClasificadorKNN/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
    #print("Entrenamiento Completo!")
    
    while True:
        
        ret, frame = video_capture.read()
        
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        rgb_small_frame = small_frame[:, :, ::-1]

        X_img = rgb_small_frame
        face_names = []
        if process_this_frame:
            face_locations = face_recognition.face_locations(X_img)#,number_of_times_upsample=2)
            face_encodings = face_recognition.face_encodings(X_img)
            #print("Buscando caras en {}".format(image_file))
            #print(face_locations)
            predictions = []
            #predictions = predict(X_img, model_path="trained_knn_model.clf",distance_threshold=0.44)
                
            #for face_encoding in face_encodings:
            for index, face_encoding in enumerate(face_encodings, start=0):
                name = "Desconocido"
                predictions = predict(X_img, model_path="trained_knn_model.clf",distance_threshold=0.44)
                face_names.append(str(predictions[index][0]))
                
                if predictions:
                    faceID = Known_face_personalHitss.index(str(predictions[index][0])) if (predictions[index][0]!='Desconocido') else -1
                    print(index,"   -------predictions",str(predictions[index][0]))
                    #name=str(predictions[i][0])
                    if faceID != -1:
                        
                        if Known_face_bandera[faceID] == 0:
                            Known_face_bandera[faceID] = 1
           
                            #print(Known_face_bandera[faceID])

                            voiceSpeech(str(predictions[index][0]))
                            
                            #print("CONTROL CONOCIDO")
                    else:
                        Known_face_bandera.append(-1);
                        Known_face_personalHitss.append('Invitado ' + str(guessID))

  
                        #voiceSpeech('Por favor identifiquese con la recepcionista')
                        #time.sleep(0.01)
                        
                        #cv2.imwrite(dest + "invitado%d.jpg" % guessID, frame) 
                        
                        guessID = guessID + 1

                        #full_file_path = os.path.join(imgFile)
                        #print("RUTA: ", full_file_path)
                        #src = full_file_path
                        #shutil.move(src, dest)

        process_this_frame= not process_this_frame
            
            #for name, (top, right, bottom, left) in predictions:
                #print("- Encontrado  {} en ({}, {})".format(name, left, top))

            #show_prediction_labels_on_image(os.path.join("ClasificadorKNN/test", X_img), predictions)

        for name in face_names:
            print("Name in faceNAME: ", name)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()