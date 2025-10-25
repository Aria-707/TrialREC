from flask import Flask, render_template, request, jsonify,redirect,Response
import cv2
import os
import numpy as np
import time
import base64
import re
import requests
import json
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

cred = credentials.Certificate('asistenciaconreconocimiento-firebase-adminsdk.json')
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
    # Mantener un índice de labels por persona
    label_dict = {name: idx for idx, name in enumerate(imagePaths)}
    next_label = len(imagePaths)

else:
    imagePaths = []
    label_dict = {}
    next_label = 0

print("Model loaded. Persons:", imagePaths)
print("Label dict:", label_dict, " Next label:", next_label)

# Para evitar registros duplicados
cap = None
duracion_reconocimiento = 3
estudiantes_reconocidos = set()
tiempos_reconocimiento = {}

def obtener_curso_activo(profesor_id=None):
    try:
        ahora = datetime.now()
        dia_semana = ahora.strftime('%A')
        hora_actual = ahora.strftime('%H:%M')
        
        print(f"\n=== BUSCANDO CURSO ACTIVO ===")
        print(f"Día: {dia_semana}, Hora: {hora_actual}")
        
        cursos_ref = db.collection('courses')
        if profesor_id:
            cursos_ref = cursos_ref.where('profesorID', '==', profesor_id)
        
        cursos = cursos_ref.get()
        
        for curso_doc in cursos:
            curso_data = curso_doc.to_dict()
            schedule = curso_data.get('schedule', {})
            
            if dia_semana in schedule:
                horario_dia = schedule[dia_semana]
                hora_inicio = horario_dia.get('start', '00:00')
                hora_fin = horario_dia.get('end', '23:59')
                
                if hora_inicio <= hora_actual <= hora_fin:
                    print(f"[✔] Curso activo: {curso_doc.id} - {curso_data.get('nameCourse')}")
                    print(f"=== FIN BÚSQUEDA ===\n")
                    return curso_doc.id
        
        print(f"[!] No se encontró curso activo. Usando 'default_course'")
        print(f"=== FIN BÚSQUEDA ===\n")
        return 'default_course'
        
    except Exception as e:
        print(f"[✖] ERROR obteniendo curso activo: {e}")
        return 'default_course'

def entrenar_incremental(nuevos_registros):
    """
    recibe nuevos_registros: dict { persona: [rutas_img1, rutas_img2, ...], ... }
    """
    global face_recognizer, label_dict, next_label, imagePaths

    # Preparar listas
    facesData, labels = [], []

    for persona, rutas in nuevos_registros.items():
        # Asigna etiqueta nueva si no existe
        if persona not in label_dict:
            label_dict[persona] = next_label
            imagePaths.append(persona)
            next_label += 1

        lbl = label_dict[persona]
        for img_path in rutas:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                facesData.append(img)
                labels.append(lbl)

    if facesData:
        # Continúa entrenamiento (update) sobre el modelo existente
        face_recognizer.update(facesData, np.array(labels))
        face_recognizer.write(model_path)
        print(f"Entrenamiento incremental: añadido {len(facesData)} imágenes.")
    else:
        print("No hay imágenes nuevas para entrenar.")

def entrenar_modelo():
    global face_recognizer, imagePaths
    print("Entrenando modelo...")

    # Listado de personas en Data/
    peopleList = [d for d in os.listdir(dataPath) if os.path.isdir(os.path.join(dataPath, d))]
    print(f"Personas encontradas: {peopleList}")

    labels, facesData = [], []
    label = 0

    for nameDir in peopleList:
        personPath = os.path.join(dataPath, nameDir)
        print(f" Procesando carpeta: {nameDir}")

        for fileName in os.listdir(personPath):
            image_path = os.path.join(personPath, fileName)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                facesData.append(img)
                labels.append(label)

        label += 1

    print(f" Total de imágenes: {len(facesData)}")
    if not facesData:
        print(" ERROR: No se encontraron imágenes para entrenar")
        return

    # 1) Entrenar el recognizer
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(facesData, np.array(labels))
    imagePaths = peopleList
    print(" === MODELO ENTRENADO CON ÉXITO ===")
    print(f" ImagePaths actualizado: {imagePaths}")

    # 2) Guardar modelo en disco
    face_recognizer.write(model_path)
    print(f" Modelo guardado en: {model_path}")

    # 3) Borrar todas las imágenes originales
    for nameDir in peopleList:
        personPath = os.path.join(dataPath, nameDir)
        for fileName in os.listdir(personPath):
            img_file = os.path.join(personPath, fileName)
            os.remove(img_file)
        # (Opcional) eliminar carpeta vacía:
        os.rmdir(personPath)
        print(f" Carpeta eliminada: {personPath}")

    print(" Todas las imágenes originales han sido borradas.")

def entrenar_incremental(nuevos_registros):
    """
    recibe nuevos_registros: dict { persona: [rutas_img1, rutas_img2, ...], ... }
    """
    global face_recognizer, label_dict, next_label, imagePaths

    # Preparar listas
    facesData, labels = [], []

    for persona, rutas in nuevos_registros.items():
        # Asigna etiqueta nueva si no existe
        if persona not in label_dict:
            label_dict[persona] = next_label
            imagePaths.append(persona)
            next_label += 1

        lbl = label_dict[persona]
        for img_path in rutas:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                facesData.append(img)
                labels.append(lbl)

    if facesData:
        # Continúa entrenamiento (update) sobre el modelo existente
        face_recognizer.update(facesData, np.array(labels))
        face_recognizer.write(model_path)
        print(f"Entrenamiento incremental: añadido {len(facesData)} imágenes.")
    else:
        print("No hay imágenes nuevas para entrenar.")


def registrar_asistencia(nombre_estudiante, courseID='default_course'):
    """
    Registra la asistencia de un estudiante en la estructura jerárquica:
    courses/{courseID}/assistances/{YYYY-MM-DD}/{estudianteID}
    
    Args:
        nombre_estudiante: Nombre del estudiante reconocido
        courseID: ID del curso activo (por defecto 'default_course')
    """
    try:
        # 1. Obtener la fecha actual en formato YYYY-MM-DD
        from datetime import datetime
        fecha_hoy = datetime.now().strftime('%Y-%m-%d')
        hora_actual = datetime.now().strftime('%H:%M')
        
        print(f"\n=== REGISTRANDO ASISTENCIA ===")
        print(f"Estudiante: {nombre_estudiante}")
        print(f"Fecha: {fecha_hoy}")
        print(f"Hora: {hora_actual}")
        print(f"Curso: {courseID}")
        
        # 2. Buscar el ID del estudiante por su nombre en la colección 'person'
        personas_ref = db.collection('person')
        query = personas_ref.where('namePerson', '==', nombre_estudiante).where('type', '==', 'Estudiante').limit(1)
        resultados = query.get()
        
        if not resultados:
            print(f"[✖] ERROR: No se encontró estudiante con nombre '{nombre_estudiante}' en la colección.")
            return False
        
        estudiante_doc = resultados[0]
        estudianteID = estudiante_doc.id
        print(f"EstudianteID encontrado: {estudianteID}")
        
        # 3. Verificar que el estudiante esté inscrito en el curso
        estudiante_data = estudiante_doc.to_dict()
        cursos_estudiante = estudiante_data.get('courses', [])
        
        if courseID not in cursos_estudiante:
            print(f"[✖] ADVERTENCIA: El estudiante {nombre_estudiante} no está inscrito en el curso {courseID}")
            # Puedes decidir si continuar o no. Por ahora, continuamos pero mostramos advertencia
        
        # 4. Referencia al documento de asistencia del día
        asistencia_ref = db.collection('courses').document(courseID).collection('assistances').document(fecha_hoy)
        
        # 5. Verificar si ya existe el documento de la fecha
        asistencia_doc = asistencia_ref.get()
        
        # 6. Crear o actualizar el documento con la asistencia del estudiante
        datos_asistencia = {
            estudianteID: {
                'estadoAsistencia': 'Presente',
                'horaRegistro': hora_actual
            }
        }
        
        if asistencia_doc.exists:
            # El documento ya existe, verificar si el estudiante ya registró asistencia
            datos_existentes = asistencia_doc.to_dict() or {}
            
            if estudianteID in datos_existentes:
                print(f"[!] El estudiante {nombre_estudiante} ya tiene asistencia registrada hoy")
                return True
            
            # Agregar el nuevo estudiante al documento existente
            asistencia_ref.update(datos_asistencia)
            print(f"[✔] Asistencia ACTUALIZADA para {nombre_estudiante} en {courseID}/{fecha_hoy}")
        else:
            # Crear nuevo documento para esta fecha
            asistencia_ref.set(datos_asistencia)
            print(f"[✔] Asistencia CREADA para {nombre_estudiante} en {courseID}/{fecha_hoy}")
        
        print(f"=== REGISTRO EXITOSO ===\n")
        return True
        
    except Exception as e:
        print(f"[✖] ERROR registrando asistencia: {e}")
        import traceback
        traceback.print_exc()
        return False

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
    nuevas_rutas = []
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

        #cv2.imshow('Capturando rostros', frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #cv2.destroyAllWindows()
    print(f"Captura completada. Total: {count} fotos subidas a Supabase")

    # Entrenar el modelo
    print("=== INICIANDO ENTRENAMIENTO INCREMENTAL ===")
    entrenar_incremental({estudiante: nuevas_rutas})

# Borra las imágenes procesadas
    for r in nuevas_rutas:
        os.remove(r)
        os.rmdir(personPath)
        print(" Imágenes temporales eliminadas.")

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

if __name__ == '__main__':
    app.run(debug=True)
