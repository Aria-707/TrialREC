from flask import Flask, render_template, Response, request, redirect
import cv2
import os
import numpy as np
import time
import requests

app = Flask(__name__)

# Rutas
dataPath = os.path.join(os.path.dirname(__file__), 'Data')
model_path = os.path.join('backend', 'modeloLBPHReconocimientoOpencv.xml')

# Reconocimiento facial
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Carga previa del modelo si existe
if os.path.exists(model_path):
    face_recognizer.read(model_path)
    imagePaths = os.listdir(dataPath)
else:
    imagePaths = []

# Variables globales
cap = None
duracion_reconocimiento = 3
estudiantes_reconocidos = set()
tiempos_reconocimiento = {}

def entrenar_modelo():
    global face_recognizer, imagePaths

    peopleList = os.listdir(dataPath)
    labels = []
    facesData = []
    label = 0

    for nameDir in peopleList:
        personPath = os.path.join(dataPath, nameDir)
        for fileName in os.listdir(personPath):
            image = cv2.imread(os.path.join(personPath, fileName), 0)
            if image is not None:
                labels.append(label)
                facesData.append(image)
        label += 1

    if facesData:
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.train(facesData, np.array(labels))
        face_recognizer.write(model_path)
        imagePaths = peopleList
        print("Modelo entrenado con éxito.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/videoReC')
def videoRec():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    global cap, imagePaths

    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = gray.copy()
        faces = faceClassif.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            rostro = auxFrame[y:y+h, x:x+w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            result = face_recognizer.predict(rostro)

            if result[1] < 70 and result[0] < len(imagePaths):
                nombre = imagePaths[result[0]]
                if nombre not in tiempos_reconocimiento:
                    tiempos_reconocimiento[nombre] = time.time()
                elif time.time() - tiempos_reconocimiento[nombre] >= duracion_reconocimiento:
                    if nombre not in estudiantes_reconocidos:
                        estudiantes_reconocidos.add(nombre)
                        print(f"[✔] Reconocido: {nombre}")
                        registrar_asistencia(nombre)
                cv2.putText(frame, nombre, (x, y-25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Desconocido', (x, y-20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/registro')
def registro():
    return render_template('registro.html')

@app.route('/registrar', methods=['POST'])
def registrar():
    global cap
    estudiante = request.form['estudiante']
    personPath = os.path.join(dataPath, estudiante)

    if not os.path.exists(personPath):
        os.makedirs(personPath)

    count = 0
    while count < 100:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceClassif.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (150, 150), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(personPath, f'rostro_{count}.jpg'), face)
            count += 1

    cap.release()
    cv2.destroyAllWindows()

    # Entrena una vez que se han capturado los datos
    entrenar_modelo()
    return redirect('/')

def generate_frames_registro():
    global cap
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceClassif.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_registro')
def video_registro():
    return Response(generate_frames_registro(), mimetype='multipart/x-mixed-replace; boundary=frame')

def registrar_asistencia(nombre):
    url = 'https://registro-asistencia-pgc.netlify.app/.netlify/functions/regAsistencia'  # CAMBIA esto
    headers = {'Content-Type': 'application/json'}
    payload = {
        "estudiante": nombre,
        "estadoAsistencia": "Presente"
    }
    try:
        r = requests.post(url, json=payload, headers=headers)
        r.raise_for_status()
        print(f"[✔] Asistencia registrada para {nombre}")
    except Exception as e:
        print(f"[✖] Error registrando asistencia: {e}")

if __name__ == '__main__':
    app.run(debug=True)