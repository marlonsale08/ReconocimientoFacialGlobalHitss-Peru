import os.path
import os
import face_recognition

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