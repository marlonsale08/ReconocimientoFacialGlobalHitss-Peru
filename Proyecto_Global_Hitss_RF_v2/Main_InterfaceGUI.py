import os
#os.environ["SDL_VIDEODRIVER"] = "dummy"
'''
    SISTEMA DE RECONOCIMIENTO-FACIAL-GLOBAL-HITSS
''' 
#Dependencias de librerias python:
from face_recognition.face_recognition_cli import image_files_in_folder
from multiprocessing import Process
from sys import stdout
from datetime import datetime
import math
from sklearn import neighbors
import xlwt
import wx
import os
from time import sleep
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from tempfile import TemporaryFile
import time
import cv2
from PIL import Image
import threading
import pyttsx3
from gtts import gTTS
import pygame
import shutil
import concurrent.futures
import pygame, sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import smtplib
import shutil
from collections import Counter
###################################################
#importando dependencias del paquete __Funciones__:
###################################################
from Funciones.Entrenamiento import train
from Funciones.Enviar_mensaje_foto import EnviarMensajeFoto
from Funciones.Enviar_mensaje_texto import EnviarMensajeTexto
from Funciones.Obtener_hora_actual import greetingTime
from Funciones.Obtener_nombres_personal import getFiles
from Funciones.Predecir_personal import predict 
from Funciones.Prediccion_invitado import predictionsGuess
from Funciones.Remover_invitado import removeGuess
from Funciones.Saludar_personal import voiceSpeech
from Funciones.Zoom_camara import zoom
from Funciones.Obtener_hora_actual import greetingTime
from Funciones.Enviar_reporte_asistencia import EnviarMensajeExcel 
from Funciones.Ingresar_colaborador import tomador_fotos_cerca,tomador_fotos_lejos
from Funciones.Amazon_speech import speech_amazon

pygame.init()

global frame_captured,is_exit, video_capture, current_hour,Video_muestra,Known_face_bandera,Known_face_personalHitss,guessID,desconocidos

old = 5 * 60
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def predictionsGuess2(X_img, origin):
    global Known_face_bandera,Known_face_personalHitss,guessID,desconocidos,frame_captured
    final_result = False
    #time.sleep(7)
    #if os.path.isdir(origin):
        #for index, img_path in enumerate(image_files_in_folder(origin), start = 0):
            #print ("Index: ", index)
            #image = face_recognition.load_image_file(img_path)
            #results = False
            #if len(face_recognition.face_encodings(image)) > 0:
                #saved_image = face_recognition.face_encodings(image)[0]
                #captured_image = face_recognition.face_encodings(X_img)[0]
                #results = face_recognition.compare_faces([saved_image], captured_image, tolerance=0.6)
                #print("Entrando al len FR: ", results)
                #if results[0] == True:
                    #print("Nuevo resultado: ", re0sults)
                    #f#inal_result = results[0]
                    #break
    #print("resultados: ", final_result)
    #if final_result==False:
    Known_face_bandera.append(-1)
    Known_face_personalHitss.append('Invitado ' + str(guessID))
    cv2.imwrite(dest + "invitado%d.jpg" % guessID, frame)                                    #voiceSpeech('Por favor identifiquese con la recepcionista') 
     #desconocidos.append(frame)   
    ImagenEnviar=Image.fromarray(frame)
    ImagenEnviar.save("ImagenEnviar.jpg")
    fp=open('ImagenEnviar.jpg','rb')
    msgImage=MIMEImage(fp.read())
    f=open('Correo_enviar.text')
    correo=f.read()
    print(correo)
    f.close()
    EnviarMensajeFoto(msgImage,"marlonsale08@gmail.com",correo,"PERSONA DESCONOCIDA EN HITSS")               
    guessID = guessID + 1 
    print("PASO LA PRUEBA")   
###################################################################################
#Ventana para ingresar nuevos usuarios
###################################################################################3
class ingresar_ventana(wx.Frame): 
    
    def __init__(self, parent, title): 
        global Id,video_capture
        wx.Frame.__init__(self, parent, -1, title)
        self.panel = wx.Panel(self)  
        #self.clave = wx.StaticText(self.panel, label="HOLA MUNDO1") 
        #self.result = wx.StaticText(self.panel, label="HOLA MUNDO2")
        #self.result.SetForegroundColour(wx.RED)
        self.button = wx.Button(self.panel, label="Guardar") 
        self.button2 = wx.Button(self.panel, label="Toma1") 
        self.button3 = wx.Button(self.panel, label="Toma2") 
        self.lblusuario = wx.StaticText(self.panel, label="Ingrese nombre de colaborador:") 
        self.editname = wx.TextCtrl(self.panel, size=(140, -1))
        self.editDNI = wx.TextCtrl(self.panel, size=(140, -1)) 
        self.lbldni = wx.StaticText(self.panel, label="Ingrese DNI de colaborador:")         

        # Set sizer for the frame, so we can change frame size to match widgets 
        self.windowSizer = wx.BoxSizer() 
        self.windowSizer.Add(self.panel, 1, wx.ALL | wx.EXPAND)   

        # Set sizer for the panel content 
        self.sizer = wx.GridBagSizer(5, 5) 
        #self.sizer.Add(self.clave, (1, 0)) 
        #self.sizer.Add(self.result, (1, 0)) 
        self.sizer.Add(self.lblusuario, (0, 0)) 
        self.sizer.Add(self.editname, (0, 1))  
        
        self.sizer.Add(self.lbldni, (1, 0), flag=wx.EXPAND)
        self.sizer.Add(self.editDNI, (1, 1), flag=wx.EXPAND)
        self.sizer.Add(self.button, (2, 1), flag=wx.EXPAND) 
        self.sizer.Add(self.button2, (3, 0), flag=wx.EXPAND)
        self.sizer.Add(self.button3, (3, 1), flag=wx.EXPAND)
        #self.sizer.Add(self.button3, (2, 1), flag=wx.EXPAND)
        #self.sizer.Add(self.button3, (2, 1), flag=wx.EXPAND)
        #self.sizer.Add(self.button3, (2, 2), (3, 2), flag=wx.EXPAND)

        # Set simple sizer for a nice border 
        self.border = wx.BoxSizer() 
        self.border.Add(self.sizer, 1, wx.ALL | wx.EXPAND, 5) 

        # Use the sizers 
        self.panel.SetSizerAndFit(self.border) 
        self.SetSizerAndFit(self.windowSizer) 

        # Set event handlers 
        self.button.Bind(wx.EVT_BUTTON, self.OnButton)
        self.button2.Bind(wx.EVT_BUTTON, self.OnButton2) 
        self.button3.Bind(wx.EVT_BUTTON, self.OnButton3)   

    def OnButton(self, e):
        #if len(self.editname.GetValue())==8:
        print("COLABORADOR GUARDADO")
        dest="ClasificadorKNN/train/"+self.editDNI.GetValue()
        try:
            os.mkdir(dest)
        except:
            print("LA CARPETA YA EXISTE")
            pass
        dest2="ClasificadorKNN/dni_colaborador/"+self.editDNI.GetValue()+".text"
        f=open(dest2,'w')
        f.write(self.editname.GetValue())
        f.close()
        #cv2.destroyAllWindows()
        #else:
            #print("DNI INVALIDO")
        #self.sizer.Add(self.result, (1, 0))
        #self.result.SetLabel(self.editname.GetValue()) 
        #tomador_fotos_cerca(video_capture,self.editname.GetValue())
        #cv2.destroyAllWindows()
        #print (self.editname.GetValue())
    def OnButton2(self,e):
        dest="ClasificadorKNN/train/"+self.editDNI.GetValue()
        try:
            os.mkdir(dest)
        except:
            print("LA CARPETA YA EXISTE")
            pass
        tomador_fotos_cerca(video_capture,self.editDNI.GetValue())
        cv2.destroyAllWindows()
    def OnButton3(self,e):
        dest="ClasificadorKNN/train/"+self.editDNI.GetValue()
        try:
            os.mkdir(dest)
        except:
            print("LA CARPETA YA EXISTE")
            pass
        tomador_fotos_lejos(video_capture,self.editDNI.GetValue())  
        cv2.destroyAllWindows()

class MyApp1(wx.App):
    def OnInit(self):
        app = wx.App(False) 
        frame = ingresar_ventana(None, "Ingresar Nuevo Colaborador")
        frame.Show(True)
        frame.Centre()
        app.MainLoop()
        return True
    
##########################################################################
#Ventana para poder modificar correo
###########################################################################3
class eliminar_ventana(wx.Frame): 
    
    def __init__(self, parent, title): 
        global Id,video_capture,Known_face_personalHitss,Known_face_bandera
        wx.Frame.__init__(self, parent, -1, title)
        self.panel = wx.Panel(self)  
        #self.clave = wx.StaticText(self.panel, label="HOLA MUNDO1") 
        #self.result = wx.StaticText(self.panel, label="HOLA MUNDO2")
        #self.result.SetForegroundColour(wx.RED)
        self.button = wx.Button(self.panel, label="Eliminar") 
        self.lblusuario = wx.StaticText(self.panel, label="Ingrese DNI de colaborador:") 
        self.editDNI = wx.TextCtrl(self.panel, size=(140, -1)) 

        # Set sizer for the frame, so we can change frame size to match widgets 
        self.windowSizer = wx.BoxSizer() 
        self.windowSizer.Add(self.panel, 1, wx.ALL | wx.EXPAND)   

        # Set sizer for the panel content 
        self.sizer = wx.GridBagSizer(5, 5) 
        #self.sizer.Add(self.clave, (1, 0)) 
        #self.sizer.Add(self.result, (1, 1)) 
        self.sizer.Add(self.lblusuario, (0, 0)) 
        self.sizer.Add(self.editDNI, (0, 1)) 
        self.sizer.Add(self.button, (1, 0), (1, 2), flag=wx.EXPAND) 

        # Set simple sizer for a nice border 
        self.border = wx.BoxSizer() 
        self.border.Add(self.sizer, 1, wx.ALL | wx.EXPAND, 5) 

        # Use the sizers 
        self.panel.SetSizerAndFit(self.border) 
        self.SetSizerAndFit(self.windowSizer) 

        # Set event handlers 
        self.button.Bind(wx.EVT_BUTTON, self.OnButton) 

    def OnButton(self, e): 
        dest="ClasificadorKNN/train/"+self.editDNI.GetValue()
        shutil.rmtree(dest)
        origin = "ClasificadorKNN/train"
        #Known_face_personalHitss,Known_face_bandera=getFiles(origin,Known_face_personalHitss,Known_face_bandera)
        #self.result.SetLabel(self.editname.GetValue()) 
        try :
            shutil.rmtree(dest)
        except:
            print("ERROR AL ELIMINAR USUSARIO")
            pass
        #tomador_fotos(video_capture,self.editname.GetValue())
        #cv2.destroyAllWindows()
        #print (self.editname.GetValue())
def eliminar_usuario():
    global Known_face_personalHitss,Known_face_bandera
    train("ClasificadorKNN/train",model_save_path="trained_knn_model.clf",n_neighbors=2)
    Known_face_personalHitss,Known_face_bandera=getFiles(origin,Known_face_personalHitss,Known_face_bandera)
class MyApp2(wx.App):
    def OnInit(self):
        app = wx.App(False) 
        frame = eliminar_ventana(None, "Eliminar colaborador del sistema")
        frame.Show(True)
        frame.Centre()
        app.MainLoop()
        return True

###################################################################################
#Ventana para configurar usuarios
###################################################################################
class configurar_ventana(wx.Frame): 

    def __init__(self, parent, title): 
        global Id,video_capture
        wx.Frame.__init__(self, parent, -1, title)
        self.panel = wx.Panel(self)  
        #self.clave = wx.StaticText(self.panel, label="HOLA MUNDO1") 
        #self.result = wx.StaticText(self.panel, label="HOLA MUNDO2")
        #self.result.SetForegroundColour(wx.RED)
        
        self.button = wx.Button(self.panel, label="Guardar") 
        self.lblusuario = wx.StaticText(self.panel, label="Ingrese correo a reportar:") 
        self.editname = wx.TextCtrl(self.panel, size=(200, -1)) 

        # Set sizer for the frame, so we can change frame size to match widgets 
        self.windowSizer = wx.BoxSizer() 
        self.windowSizer.Add(self.panel, 1, wx.ALL | wx.EXPAND)   

        # Set sizer for the panel content 
        self.sizer = wx.GridBagSizer(5, 5) 
        #self.sizer.Add(self.clave, (1, 0)) 
        #self.sizer.Add(self.result, (1, 1)) 
        self.sizer.Add(self.lblusuario, (0, 0)) 
        self.sizer.Add(self.editname, (0, 1)) 
        self.sizer.Add(self.button, (1, 0), (1, 2), flag=wx.EXPAND) 

        # Set simple sizer for a nice border 
        self.border = wx.BoxSizer() 
        self.border.Add(self.sizer, 1, wx.ALL | wx.EXPAND, 5) 

        # Use the sizers 
        self.panel.SetSizerAndFit(self.border) 
        self.SetSizerAndFit(self.windowSizer) 

        # Set event handlers 
        self.button.Bind(wx.EVT_BUTTON, self.OnButton) 

    def OnButton(self, e): 
        #self.result.SetLabel(self.editname.GetValue()) 
        f=open("Correo_enviar.text","w")
        f.write(self.editname.GetValue())
        f.close()
        #print (self.editname.GetValue())

class MyApp3(wx.App):
    def OnInit(self):
        app = wx.App(False) 
        frame = configurar_ventana(None, "Configuracion de correo")
        frame.Show(True)
        frame.Centre()
        app.MainLoop()
        return True
#############################################################################

class Powerup(pygame.sprite.Sprite):
    
    def __init__(self,imagen):
        pygame.sprite.Sprite.__init__(self)
        self.image =pygame.image.load(imagen)
        #self.image=pygame.transform.scale(self.image,(1320,800))
        self.rect = self.image.get_rect()
 
    def update(self, window):
    
        pass
#Clase que denota el cursor del mouse y obtiene la posicion del mismo
class Cursor(pygame.Rect):
    def __init__(self):
        pygame.Rect.__init__(self,0,0,1,1)
    def update(self):
        self.left,self.top=pygame.mouse.get_pos()

#Clase boton crea un boton alternando dos imagenes imagen1 e imagen2 cunaod esta sobre una de ellas

class Boton(pygame.sprite.Sprite):
    def __init__(self,imagen1,imagen2,x=200,y=200):
        self.imagen_normal=imagen1
        self.imagen_seleccion=imagen2
        self.imagen_actual=self.imagen_normal
        self.rect=self.imagen_actual.get_rect()
        self.rect.left,self.rect.top=(x,y)
    def update(self,pantalla,cursor):
        if cursor.colliderect(self.rect):
            self.imagen_actual=self.imagen_seleccion
        else: self.imagen_actual=self.imagen_normal

        pantalla.blit(self.imagen_actual,self.rect)

if __name__ == "__main__":
#Cargar todas las imagenes que se usaran tanto iconos como fondos:
    window = pygame.display.set_mode([1320,800])
    pygame.display.set_caption('Sistema de Reconocimiento Facial Global Hitss Peru')
    clock = pygame.time.Clock()
    powerup = Powerup('/home/marlon/Proyecto_Global_Hitss_RF_v1/Interface/Imagenes/GlobalHitss.jpg')
    powerup_group = pygame.sprite.GroupSingle()
    powerup_group.add(powerup)
    logo2=pygame.image.load('/home/marlon/Proyecto_Global_Hitss_RF_v1/Interface/Imagenes/logoGlobal.png')
    imagen=pygame.image.load('/home/marlon/Proyecto_Global_Hitss_RF_v1/Interface/Imagenes/botonBlancoHD.png')
    dashboardEntrada=pygame.image.load('/home/marlon/Proyecto_Global_Hitss_RF_v1/Interface/Imagenes/fondo2.jpg')
    dashboardEntrada=pygame.transform.scale(dashboardEntrada,(400,585))
    dashboardEntrada2=pygame.image.load('/home/marlon/Proyecto_Global_Hitss_RF_v1/Interface/Imagenes/fondo2.jpg')
    dashboardEntrada2=pygame.transform.scale(dashboardEntrada,(250,800))
    fondoPrincipal=pygame.image.load('/home/marlon/Proyecto_Global_Hitss_RF_v1/Interface/Imagenes/fondoPrincipal.jpg')
    dashboard1=pygame.transform.rotate(dashboardEntrada,90)
    dashboard1=pygame.transform.scale(dashboard1,(350,690))
    dashboard1=pygame.transform.rotozoom(dashboard1,180,1)
    dashboard2=pygame.transform.scale(imagen,(1200,150))
    dashboard2=pygame.transform.rotate(dashboard2,360)
    dashboard2arriba=pygame.transform.rotate(dashboard2,180)
    dashboard3=pygame.transform.scale(imagen,(1200,600))
    dashboard3=pygame.transform.rotate(dashboard3,360)
    prendido=pygame.image.load('/home/marlon/Proyecto_Global_Hitss_RF_v1/Interface/Imagenes/iniciar-128.png')
    prendido2=pygame.image.load('/home/marlon/Proyecto_Global_Hitss_RF_v1/Interface/Imagenes/iniciar-128-2.png')
    boton1=Boton(prendido,prendido2,150,520)
    cursor1=Cursor()
    camaraweb1=pygame.image.load('/home/marlon/Proyecto_Global_Hitss_RF_v1/Interface/Imagenes/camara-web-128.png')
    camaraweb2=pygame.image.load('/home/marlon/Proyecto_Global_Hitss_RF_v1/Interface/Imagenes/camara-web-64.png')
    botonCamara=Boton(camaraweb1,camaraweb2,150,100)
    entrenar=pygame.image.load('/home/marlon/Proyecto_Global_Hitss_RF_v1/Interface/Imagenes/entrenar-128.png')
    busqueda1=pygame.image.load('/home/marlon/Proyecto_Global_Hitss_RF_v1/Interface/Imagenes/Buscar-128.png')
    correo=pygame.image.load('/home/marlon/Proyecto_Global_Hitss_RF_v1/Interface/Imagenes/correo-128.png')
    busqueda2=pygame.image.load('/home/marlon/Proyecto_Global_Hitss_RF_v1/Interface/Imagenes/busqueda-256-2.png')
    botonBusqueda=Boton(busqueda1,busqueda2,150,300)
    dashboardPrincipal=pygame.transform.rotate(imagen,90)
    dashboardPrincipal=pygame.transform.scale(dashboardPrincipal,(400,300))
    logoPrincipal=pygame.image.load('/home/marlon/Proyecto_Global_Hitss_RF_v1/Interface/Imagenes/logoGlobal.png')
    boton=pygame.image.load('/home/marlon/Proyecto_Global_Hitss_RF_v1/Interface/Imagenes/botonBlancoHD.png')
    boton=pygame.transform.scale(boton,(250,100))
    cursor2=Cursor()
    fuente = pygame.font.Font(None,18)
    texto1 = fuente.render("SISTEMA DE RECONOCIMIENTO FACIAL", 0, (255,255,255))
    fuente2 = pygame.font.Font(None,16)
    texto2 = fuente.render("Ingresar Staff", 0, (255,255,255)) 
    texto3=fuente.render("Eliminar Staff", 0, (255,255,255))
    texto4 = fuente.render("Entrenar Sistema", 0, (255,255,255))
    texto5 = fuente.render("Configurar correo", 0, (255,255,255))
    now = time.time()
    origin = "ClasificadorKNN/train"
    dest = "ClasificadorKNN/invitados/"
####################################################################################
    #Inicializa la camara en video_capture
    
    video_capture = cv2.VideoCapture(0)
    video_capture.set(3,1920)
    video_capture.set(4,1080)

    #Inicializando todas la variables a usar
    is_exit = False
    Known_face_personalHitss = []
    Known_face_bandera = []
    known_id_personal=[]
    face_locations = []
    face_encodings = []
    face_names = []
    conocidos=[]
    desconocidos=[]
    process_this_frame = True
    guessID = 1     
    Known_face_personalHitss,Known_face_bandera=getFiles(origin,Known_face_personalHitss,Known_face_bandera)
    style1 = xlwt.easyxf(num_format_str='HH-DD-MM-YY')
    wb = xlwt.Workbook()
    ws = wb.add_sheet('A Test Sheet')
    i=0
    bandera_mañana=0
    bandera_tarde=0
    bandera_noche=0
    bandera_enviado=0
    old=5*60
    #hilo_train=threading.Thread(target=train,args=("ClasificadorKNN/train","trained_knn_model.clf", 1,None,None,))
    #hilo_train.start()
    train("ClasificadorKNN/train",model_save_path="trained_knn_model.clf",n_neighbors=2)
    while True:
        pygame.display.update()
        powerup.update(window)
        powerup_group.draw(window)
        window.blit(camaraweb2,(80,20))
        window.blit(logoPrincipal,(153,25))
        window.blit(dashboardEntrada,(27,95))
        cursor1.update()
        boton1.update(window,cursor1)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x_mouse,y_mouse = pygame.mouse.get_pos()

                if (150<=x_mouse<=520 ) and (300<=y_mouse<=648):
                    while True:

                        timeConsulta=datetime.now().hour

                        if timeConsulta==11 and bandera_mañana==0:
                           
                            Known_face_personalHitss,Known_face_bandera=getFiles(origin,Known_face_personalHitss,Known_face_bandera)
                            bandera_mañana=1

                        elif timeConsulta==18 and bandera_tarde==0:
                            
                            Known_face_personalHitss,Known_face_bandera=getFiles(origin,Known_face_personalHitss,Known_face_bandera)
                            bandera_tarde=1

                        elif timeConsulta==6 and bandera_noche==0:
                            
                            Known_face_personalHitss,Known_face_bandera=getFiles(origin,Known_face_personalHitss,Known_face_bandera)
                            bandera_noche=1

                        elif timeConsulta==18 and bandera_enviado==0:
                            hilo_enviar_correo_excel=threading.Thread(target=EnviarMensajeExcel,args=("GlobalHitssPeruRF@gmail.com","GlobalHitssPeruRF@gmail.com","ASISTENCIA DEL DIA "))
                            #EnviarMensajeExcel(correoFrom="GlobalHitssPeruRF@gmail.com",correoTo="GlobalHitssPeruRF@gmail.com",subject="ASISTENCIA DIA")
                            hilo_enviar_correo_excel.start()
                            bandera_enviado=1
                            #train("ClasificadorKNN/train",model_save_path="trained_knn_model.clf",n_neighbors=2)
                            

                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                pygame.quit()
                                sys.exit() 
                            elif event.type == pygame.MOUSEBUTTONDOWN:
                                x_mouse,y_mouse = pygame.mouse.get_pos()

                                if (68<=x_mouse<=172 ) and (20<=y_mouse<=148):
                                    MyApp1().MainLoop()
                                elif (68<=x_mouse<=172) and (200<=y_mouse<=328):
                                    MyApp2().MainLoop()
                                elif (68<=x_mouse<=172 ) and (380<=y_mouse<=508):
                                    #Known_face_personalHitss,Known_face_bandera=getFiles(origin,Known_face_personalHitss,Known_face_bandera)
                                    #hilo_train=threading.Thread(target=train,args=("ClasificadorKNN/train","trained_knn_model.clf", 1,None,None,))
                                    #hilo_train.start()
                                    train("ClasificadorKNN/train",model_save_path="trained_knn_model.clf",n_neighbors=2)
                                    Known_face_personalHitss,Known_face_bandera=getFiles(origin,Known_face_personalHitss,Known_face_bandera)
                                    print("HOLA MUNDO")
                                    #train("ClasificadorKNN/train", model_save_path="trained_knn_model.clf", n_neighbors=1)
                                elif (68<=x_mouse<=172 ) and (560<=y_mouse<=688):
                                    #hilo_enviar_correo_excel=threading.Thread(target=EnviarMensajeExcel,args=(None,"GlobalHitssPeruRF@gmail.com","GlobalHitssPeruRF@gmail.com","ASISTENCIA DEL DIA",))
                                    #hilo_enviar_correo_excel.start()  
                                    #EnviarMensajeFoto(msgImage,"marlonsale08@gmail.com","marlonsale08@gmail.com","Posible persona desconocida en Hitss")
                                    MyApp3().MainLoop
                        pygame.display.update()
                        powerup.update(window)
                        powerup_group.draw(window)
                        window.blit(dashboard2arriba,(180,10))
                        window.blit(dashboard2,(180,550))
                        window.blit(dashboardEntrada2,(10,-10))
                        window.blit(entrenar,(68,380))
                        window.blit(busqueda1,(68,200))
                        window.blit(correo,(68,560))
                        window.blit(camaraweb1,(68,20))
                        window.blit(texto2, (90,162))
                        window.blit(texto3, (90,346))
                        window.blit(texto4, (80,533))
                        window.blit(texto5, (80,690))
                        window.blit(logo2,(1120,50))

                        #if old - ((time.time() - now) % old) > old - 1:
                            #removeGuess(dest)

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

                            if len(desconocidos)>200:
                                desconocidos=[]
                            face_names = []    

                            for index, face_encoding in enumerate(face_encodings, start=0):
                                predictions = predict(X_img, model_path="trained_knn_model.clf",distance_threshold=0.44)
                                print(predictions)
                                face_names.append(str(predictions[index][0]))
                                
                                if predictions:
                                    faceID = Known_face_personalHitss.index(str(predictions[index][0])) if (predictions[index][0]!='Desconocido') else -1
                                    
                                    if faceID != -1:
                                        
                                        if Known_face_bandera[faceID] == 0:
                                            Known_face_bandera[faceID] = 1
                                            voiceSpeech(str(predictions[index][0]))
                                            hora=time.strftime("%H:%M:%S") #Formato de 24 horas
                                            ws.write(i,0,hora,style1)
                                            ws.write(i,1,str(predictions[index][0]))
                                            wb.save('Reporte_Asistencia/ASISTENCIA.xls')
                                            wb.save('Funciones/ASISTENCIA.xls') 
                                            conocidos.append(frame)
                                            i+=1

                                    else:

                                        print("DESCONOCIDO")

                                        for (top, right, bottom, left), name in zip(face_locations, face_names):
         
                                                color= (0,180,0)
                                                top *= 4
                                                right *= 4
                                                bottom *= 4
                                                left *= 4
                                                cv2.rectangle(frame_captured, (left, top), (right, bottom), color, 4)
                                                #cv2.rectangle(frame_captured, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                                                #font = cv2.FONT_HERSHEY_TRIPLEX
                                                #cv2.putText(frame_captured, name, (left +20, bottom -6), font, 0.75, (255, 255, 255), 1)
                                                img = Image.fromarray(frame_captured)
                                                crop_imgd = img.crop((left-30,top-50,right+30,bottom+20))
                                                crop_imgd.save('desconocido.jpg')
                                                desconocido=pygame.image.load('desconocido.jpg')
                                                desconocidos.append(desconocido)
                                        #X_img=X_img-18
                                        '''ImagenEnviar=Image.fromarray(X_img)
                                        ImagenEnviar.save("ImagenEnviar.jpg")
                                        fp=open('ImagenEnviar.jpg','rb')
                                        msgImage=MIMEImage(fp.read())'''
                        #EnviarMensajeFoto(msgImage,"marlonsale08@gmail.com","marlonsale08@gmail.com","Posible persona desconocida en Hitss")
                                        
                                        
                                        hilo=threading.Thread(target=predictionsGuess2,args=(X_img, dest,))
                                        #hilo.daemon()
                                        hilo.start()
                                        #print("compare guess: ", compare_guess)
                                        #start_time=time.time()
                                        #with concurrent.futures.ThreadPoolExecutor() as executor:
                                        #compare_guess=executor.submit(predictionsGuess,X_img,dest)
                                            #print(compare_guess.running())
                                        '''if compare_guess == False:
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
                                            guessID = guessID + 1'''
        
                                        #desconocidos.append(frame)
                                        #end_time=time.time()
                                        #compare_guess.cancel()
                                        #print("Llamada"+str(compare_guess.cancel()))
                                        #print(end_time-start_time)
                                        #full_file_path = os.path.join(imgFile)
                                        #print("RUTA: ", full_file_path)
                                        #src = full_file_path'''
                                        #shutil.move(src, dest)

                        process_this_frame= not process_this_frame
          
                        for (top, right, bottom, left), name in zip(face_locations, face_names):
                        
                            if name=="Desconocido":
                                color= (255,0,0)
                                
                            else:
                                color=(0,0,255)
                                  
                            top *= 4
                            right *= 4
                            bottom *= 4
                            left *= 4
                            cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
                            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                            font = cv2.FONT_HERSHEY_TRIPLEX
                            cv2.putText(frame, name, (left +20, bottom -6), font, 1, (255, 255, 255), 1)
                            
                        x=len(desconocidos)-1
                        y=0
                   
                        for m in range(len(desconocidos)):
                      
                            video3=desconocidos[x]
                            video3 = pygame.transform.flip(video3,False,True)
                            
                            video3=pygame.transform.rotate(video3,180)
                            video3=pygame.transform.scale(video3,(100,100))
                            window.blit(video3,(277*(y+1)-145*y,574))
                            x=x-1
                            y=y+1
                   
                        z=len(conocidos)-1 
                        a=0
                        for m in range(len(conocidos)):
                            video3 = pygame.pixelcopy.make_surface(conocidos[z])
                            video3 = pygame.transform.flip(video3,False,True)
                            video3=pygame.transform.rotate(video3,-90)
                            video3=pygame.transform.scale(video3,(100,100))
                            window.blit(video3,(277*(a+1)-150*a,30))
                            z=z-1
                            a=a+1
                        video = pygame.pixelcopy.make_surface(frame)
                        video = pygame.transform.flip(video,False,True)
                        #video=pygame.tranform.scale2x(video)
                        video=pygame.transform.rotate(video,-90)
                        video=pygame.transform.scale(video,(450,330))
                        window.blit(video,(800,200))
                        pygame.display.update()
                        end_time=datetime.now()
                        #print(end_time-start_time)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            is_exit = True
                            break
                    video_capture.release()
                    cv2.destroyAllWindows()
            elif event.type == pygame.KEYDOWN:
                print("SE PRESIONO UNA TECLA")