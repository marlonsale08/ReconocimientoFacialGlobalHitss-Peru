'''Funcion guarda 40 fotografias de un nuevo ingreso en Hitss'''
import cv2
import time
import os
from PIL import Image,ImageDraw

def tomador_fotos_cerca(cam=None,Id=None):
    i=1
    dest="ClasificadorKNN/train/"+Id+"/"
    switch=True
    cam.set(3,1920)
    cam.set(4,1080)
    while switch:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            switch = False
            #.release()
            break

        ret, foto=cam.read() 
        foto_g=foto
        if i==20:
        	i=0

        cv2.imwrite(dest + "foto%i.jpg" % i, foto_g) 

        color=(255,0,0)
        parametro=60

        ancho=cam.get(4)
        ancho=int(ancho)
        largo=cam.get(3)
        largo=int(largo)

        top = (ancho)//2-parametro
        right =(largo)//2-parametro
        bottom = (ancho)//2+parametro
        left = (largo)//2+parametro
        #draw=ImageDraw.Draw(foto)
        #draw.rectangle(((left,top),(rigth,bottom)),outline=(0,0,255))
        cv2.rectangle(foto, (left, top), (right, bottom), color, 3)

        cv2.imshow("Video",foto_g)
        i=i+1

'''cam=cv2.VideoCapture(0)
cam.set(10,100)
nombre="Marlon"
tomador_fotos(cam,nombre)
cv2.destroyAllWindows()'''

def tomador_fotos_lejos(cam=None,Id=None):
    i=20
    dest="ClasificadorKNN/train/"+Id+"/"
    print(type(Id))
    print(dest)
    switch=True
    cam.set(3,1920)
    cam.set(4,1080)
    while switch:

        if cv2.waitKey(1) & 0xFF == ord('q'):
            switch = False
            #.release()
            break
        if i==40:
        	i=20
        ret, foto=cam.read() 
        foto_g=foto

        cv2.imwrite(dest + "foto%i.jpg" % i, foto_g) 
        color=(255,0,0)
        parametro=120
        
        ancho=cam.get(4)
        ancho=int(ancho)
        largo=cam.get(3)
        largo=int(largo)

        top = (ancho)//2-parametro
        right =(largo)//2-parametro
        bottom = (ancho)//2+parametro
        left = (largo)//2+parametro
        #draw=ImageDraw.Draw(foto)
        #draw.rectangle(((left,top),(rigth,bottom)),outline=(0,0,255))
        cv2.rectangle(foto, (left, top), (right, bottom), color, 3)

        cv2.imshow("Video",foto_g)
        i=i+1
        #print(ancho+largo)
'''cam=cv2.VideoCapture(0)
cam.set(10,100)
nombre="Marlon"
tomador_fotos(cam,nombre)
cv2.destroyAllWindows()'''
