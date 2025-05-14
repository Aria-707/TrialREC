from flask import Flask, render_template, request, jsonify,redirect,Response
import cv2
import os
import numpy as np
import time
import base64
import re
import requests
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore
cred = credentials.Certificate('asistenciaconreconocimiento-firebase-adminsdk-fbsvc-793e372c66.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

app = Flask(__name__)
CORS(app)




# Rutas
dataPath = os.path.join(os.path.dirname(__file__), 'Data')
model_path = os.path.join('backend', 'modeloLBPHReconocimientoOpencv.xml')

# Clasificadores
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Carga del modelo
if os.path.exists(model_path):
    face_recognizer.read(model_path)
    imagePaths = os.listdir(dataPath)
else:
    imagePaths = []

# Para evitar registros duplicados
cap = None
duracion_reconocimiento = 3
estudiantes_reconocidos = set()
tiempos_reconocimiento = {}

def entrenar_modelo():
    global face_recognizer, imagePaths
    print("Entrenando modelo...")
    
    peopleList = os.listdir(dataPath)
    print(f"Personas encontradas: {peopleList}")
    
    labels = []
    facesData = []
    label = 0
    
    for nameDir in peopleList:
        personPath = os.path.join(dataPath, nameDir)
        print(f"Procesando carpeta: {nameDir}")
        
        if not os.path.isdir(personPath):
            print(f"Saltando {nameDir} (no es directorio)")
            continue
        
        images_count = 0
        for fileName in os.listdir(personPath):
            image_path = os.path.join(personPath, fileName)
            image = cv2.imread(image_path, 0)
            if image is not None:
                labels.append(label)
                facesData.append(image)
                images_count += 1
        
        print(f"  - {images_count} imágenes procesadas para {nameDir}")
        label += 1
    
    print(f"Total de imágenes: {len(facesData)}")
    print(f"Total de etiquetas: {len(labels)}")
    
    if facesData:
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.train(facesData, np.array(labels))
        imagePaths = peopleList
        print("=== MODELO ENTRENADO CON ÉXITO ===")
        print(f"ImagePaths actualizado: {imagePaths}")
    else:
        print("ERROR: No se encontraron imágenes para entrenar")

def registrar_asistencia(nombre):
    url = 'https://registro-asistencia-pgc.netlify.app/.netlify/functions/regAsistencia'
    headers = {'Content-Type': 'application/json'}
    payload = {"estudiante": nombre, "estadoAsistencia": "Presente"}
    try:
        r = requests.post(url, json=payload, headers=headers)
        r.raise_for_status()
        print(f"[✔] Asistencia registrada para {nombre}")
    except Exception as e:
        print(f"[✖] Error registrando asistencia: {e}")

def decode_image(data_url):
    """Decodifica base64 data:image/... y devuelve BGR numpy array."""
    b64 = re.sub(r'^data:image/.+;base64,', '', data_url)
    b = base64.b64decode(b64)
    arr = np.frombuffer(b, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/registrar')
def registrar():
    return render_template('registrar.html')

@app.route('/registrer', methods=['POST'])
def registrer():
    global cap
    print("=== INICIANDO REGISTRO ===")

    estudiante = request.form['estudiante']
    print(f"Estudiante: {estudiante}")

    personPath = os.path.join(dataPath, estudiante)
    print(f"Ruta de la persona: {personPath}")

    if not os.path.exists(personPath):
        os.makedirs(personPath)
        print(f"Directorio creado: {personPath}")
    else:
        print(f"Directorio ya existe: {personPath}")

    # Verificar que la cámara esté funcionando
    if cap is None or not cap.isOpened():
        print("Reiniciando cámara...")
        cap = cv2.VideoCapture(0)

    count = 0
    print("Iniciando captura de fotos...")
    fotos_subidas = []  # Lista para almacenar las rutas de las fotos subidas a Supabase

    while count < 100:
        ret, frame = cap.read()
        if not ret:
            print(f"Error al leer frame {count}")
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceClassif.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (150, 150), interpolation=cv2.INTER_CUBIC)
                # photo_path = os.path.join(personPath, f'rostro_{count}.jpg')
                # cv2.imwrite(photo_path, face)

                # Convertir la imagen a bytes JPEG para subirla a Supabase
                _, buffer = cv2.imencode('.jpg', face)
                imagen_bytes = buffer.tobytes()
                nombre_archivo = f'{estudiante}/rostro_{count}.jpg' # Ruta en Supabase

                count += 1
                break  # Solo tomar el primer rostro detectado

        # Mostrar el frame con el rectángulo del rostro (opcional)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Capturando rostros', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print(f"Captura completada. Total: {count} fotos subidas a Supabase")

    # Entrenar el modelo
    print("=== INICIANDO ENTRENAMIENTO ===")
    entrenar_modelo()

    return redirect('/')



# Función para verificar el estado inicial
def verificar_estado_inicial():
    print("\n=== VERIFICANDO ESTADO INICIAL ===")
    print(f"dataPath: {dataPath}")
    print(f"Carpeta Data existe: {os.path.exists(dataPath)}")
    
    if os.path.exists(dataPath):
        personas = os.listdir(dataPath)
        print(f"Personas existentes: {personas}")
    
    print(f"Cámara inicializada: {cap is not None and cap.isOpened()}")
    print(f"faceClassif cargado: {faceClassif is not None}")
    print("================================\n")

# Llamar esta función al inicio de tu aplicación
# verificar_estado_inicial()

@app.route('/registro', methods=['POST'])
def registro():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"estado": "error", "mensaje": "No se recibió imagen"}), 400

    # Decodifica la imagen base64
    image_data = re.sub(r'^data:image/.+;base64,', '', data['image'])
    image_bytes = base64.b64decode(image_data)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return jsonify({"estado": "sin_rostro"})

    # Procesa solo el primer rostro
    x, y, w, h = faces[0]
    rostro = auxFrame[y:y+h, x:x+w]
    rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
    label, confianza = face_recognizer.predict(rostro)

    box = [int(x), int(y), int(w), int(h)]
    if confianza < 70 and label < len(imagePaths):
        nombre = imagePaths[label]
        # Lógica de registro único
        if nombre not in tiempos_reconocimiento:
            tiempos_reconocimiento[nombre] = time.time()
        elif time.time() - tiempos_reconocimiento[nombre] >= duracion_reconocimiento:
            if nombre not in estudiantes_reconocidos:
                estudiantes_reconocidos.add(nombre)
                registrar_asistencia(nombre)
        return jsonify({
            "estado": "reconocido",
            "estudiante": nombre,
            "confianza": float(confianza),
            "box": box
        })
    else:
        return jsonify({
            "estado": "desconocido",
            "confianza": float(confianza),
            "box": box
        })
    
@app.route('/guardar_foto', methods=['POST'])
def guardar_foto():
    """
    Recibe JSON { estudiante: str, foto: <base64> }
    Detecta y recorta la cara antes de guardar en Data/<estudiante>/rostro_<timestamp>.jpg
    """
    data = request.get_json()
    nombre = data.get('estudiante','').strip()
    foto_b64 = data.get('foto','')
    if not nombre or not foto_b64:
        return jsonify({"error":"Faltan datos"}), 400

    # Sanea el nombre y crea carpeta
    nombre = os.path.splitext(os.path.basename(nombre))[0]
    personPath = os.path.join(dataPath, nombre)
    os.makedirs(personPath, exist_ok=True)

    # Decodifica base64 -> bytes -> numpy array -> imagen BGR
    header, encoded = foto_b64.split(',', 1)
    img_bytes = base64.b64decode(encoded)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convertir a gris y detectar caras
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        # No se detectó cara, saltamos esta foto
        return jsonify({"ok": False, "msg": "no face"}), 200

    # Recorta la primera cara
    x, y, w, h = faces[0]
    rostro = gray[y:y+h, x:x+w]
    rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)

    # Guarda el recorte
    timestamp = int(time.time() * 1000)
    ruta = os.path.join(personPath, f'rostro_{timestamp}.jpg')
    cv2.imwrite(ruta, rostro)

    return jsonify({"ok": True}), 200

def registrar_asistencia(nombre):
    try:
        db.collection('asistenciaReconocimiento').add({
            'estudiante': nombre,
            'estadoAsistencia': 'Presente',
            'fechaYhora': firestore.SERVER_TIMESTAMP,
            'asignatura': 'Física'
        })
        print(f"[✔] Asistencia registrada para {nombre}")
    except Exception as e:
        print(f"[✖] Error registrando asistencia: {e}")
if __name__ == '__main__':
    app.run(debug=True)
