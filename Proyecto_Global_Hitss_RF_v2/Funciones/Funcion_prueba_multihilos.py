import cv2
import time
import threading
import face_recognition

def deteccion(img):
    time.sleep(2)
    print("HOLA MUNDO")
    
if __name__=="__main__":  
    
    video_captured=cv2.VideoCapture(0)
    is_exit=True
    while is_exit:

        ret, img = video_captured.read()
        cv2.imshow("Video",img)
        #video_captured.waitkey()
        small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        img_small = small_frame[:, :, ::-1]
       
        face_locations=face_recognition.face_locations(img_small)

        for (top, right, bottom, left) in face_locations:
            
            color=(0,255,0)           
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(img, (left, top), (right, bottom), color, 2)

        cv2.imshow("Video",img)
        hilo=threading.Thread(target=deteccion,args=(img,))
        hilo.start()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            is_exit = False
            break