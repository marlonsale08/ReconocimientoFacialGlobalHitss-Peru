import os.path
import os
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

def predictionsGuess(X_img, origin):
    global Known_face_bandera,Known_face_personalHitss,guessID,desconocidos
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
    if final_result==False:
        Known_face_bandera.append(-1)
        Known_face_personalHitss.append('Invitado ' + str(guessID))
        cv2.imwrite(dest + "invitado%d.jpg" % guessID, frame) 
                                            #voiceSpeech('Por favor identifiquese con la recepcionista') 
        desconocidos.append(frame)   
        ImagenEnviar=Image.fromarray(frame)
        ImagenEnviar.save("ImagenEnviar.jpg")
        fp=open('ImagenEnviar.jpg','rb')
        msgImage=MIMEImage(fp.read())
        EnviarMensajeFoto(msgImage,"marlonsale08@gmail.com","marlonsale08@gmail.com","PERSONA DESCONOCIDA EN HITSS")               
        guessID = guessID + 1    