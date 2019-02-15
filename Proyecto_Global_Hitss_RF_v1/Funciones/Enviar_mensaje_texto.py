from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import smtplib

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

#EnviarMensajeTexto("HOLA","marlonsale08@gmail.com","marlonsale08@gmail.com")