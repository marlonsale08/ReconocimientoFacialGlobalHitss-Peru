import math
import pickle
import face_recognition
from sklearn import neighbors
import os
import os.path

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

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
    print(closest_distances)
 
    #si closest_distances<distance_threshold = True en todo caso False ,para todas las caras en la imagen
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predecir clases y eliminar clasificaciones que no están dentro del umbral
    return [(pred, loc) if rec else ("Desconocido", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
