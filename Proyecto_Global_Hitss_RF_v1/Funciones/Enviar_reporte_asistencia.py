from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
import smtplib
from email import encoders
from datetime import datetime

def EnviarMensajeExcel(msgImage=None,correoFrom=None,correoTo=None,subject=None):

    msg = MIMEMultipart()
    password = "globalhitssperu"
    msg['From'] = correoFrom
    msg['To'] = correoTo
    msg['Subject'] = subject

    adjunto_MIME=MIMEBase('application','octet-stream')

    archivo_adjunto=open('Reporte_Asistencia/ASISTENCIA.xls','rb')
    
    adjunto_MIME.set_payload((archivo_adjunto).read())
    encoders.encode_base64(adjunto_MIME)
    dia=str(datetime.now())
    adjunto_MIME.add_header('Content-Disposition', "attachment; filename= Reporte_Asistencia_%s.xls" %dia)
    msg.attach(adjunto_MIME)

    server = smtplib.SMTP('smtp.gmail.com: 587')

    server.starttls()

    server.login(msg['From'], password)

    server.sendmail(msg['From'], msg['To'], msg.as_string())

    server.quit()

    print ("Envio de correo exitoso %s:" % (msg['To']))