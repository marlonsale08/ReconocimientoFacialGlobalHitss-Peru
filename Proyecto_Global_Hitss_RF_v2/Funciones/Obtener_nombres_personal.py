import os
import os.path

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def getFiles(origin=None,Known_face_personalHitss=None ,Known_face_bandera=None):

    files = os.listdir(origin)

    for name in files:
        full_path = os.path.join(origin, name)
        if os.path.isdir(full_path):
            Known_face_personalHitss.append(name)
            Known_face_bandera.append(0)
            
    return Known_face_personalHitss,Known_face_bandera