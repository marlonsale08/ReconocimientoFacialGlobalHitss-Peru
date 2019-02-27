from sys import stdout
import math
from sklearn import neighbors
import os
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import os.path
import pickle

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    print("ENTRENANDO SISTEMA")
    X = []
    y = []
    # Recorre a través de cada persona en el conjunto de entrenamiento
    for class_dir in os.listdir(train_dir):
                                                                                                                                    
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue
        # Recorre cada imagen de entrenamiento para la persona actual
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)
                     
            if len(face_bounding_boxes) != 1:
                
                # Filtro de muchas caras
                if verbose:
                    print("Imagen {} no es recomendable para el entrenamiento: {}".format(img_path, "No se encontraron caras" if len(face_bounding_boxes) < 1 else "Muchas caras encontradas"))
            else:
                # Codifica imagen actual del conjunto de entrenamiento
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)            
    # Calcula numero de vecinos
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Numero de vecinos encontrados:", n_neighbors)

    # Crea y entrena el KNN
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')   
    knn_clf.fit(X, y)

    # Guarda el kernel generado
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)
    print("SISTEMA ENTRENADO CORRECTAMEMTE")

    return knn_clf