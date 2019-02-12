from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import smtplib

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