"""
scheduler_asistencia.py
Sistema de inicialización automática de documentos de asistencia.
Verifica cada minuto si debe crear documentos de asistencia para cursos activos.
"""

import time
import threading
from datetime import datetime, timedelta
from firebase_config import db


# Mapeo de días
DIAS_ESPANOL_A_INGLES = {
    'Lunes': 'Monday',
    'Martes': 'Tuesday',
    'Miércoles': 'Wednesday',
    'Jueves': 'Thursday',
    'Viernes': 'Friday',
    'Sábado': 'Saturday',
    'Domingo': 'Sunday'
}

DIAS_INGLES_A_ESPANOL = {v: k for k, v in DIAS_ESPANOL_A_INGLES.items()}

# Diccionario para rastrear qué documentos ya fueron inicializados
documentos_inicializados = set()  # Formato: "courseID_fecha"

def obtener_cursos_proximos_a_iniciar(salon_configurado):
    """
    Obtiene los cursos que iniciarán en los próximos 5 minutos.
    Esta función se ejecuta en el minuto :54, por lo que busca clases que empiecen en :00
    
    Args:
        salon_configurado: Salón actualmente configurado
        
    Returns:
        list: Lista de tuplas (courseID, fecha, hora_inicio)
    """
    try:
        ahora = datetime.now()
        dia_ingles = ahora.strftime('%A')
        dia_espanol = DIAS_INGLES_A_ESPANOL.get(dia_ingles, dia_ingles)
        hora_actual = ahora.strftime('%H:%M')
        fecha_actual = ahora.strftime('%Y-%m-%d')
        
        # En el minuto :54, buscamos clases que empiecen en :00 (5 minutos después)
        hora_proxima = (ahora + timedelta(minutes=6)).strftime('%H:%M')
        
        cursos_a_inicializar = []
        
        print(f"   📅 Día: {dia_espanol} ({dia_ingles})")
        print(f"   🕐 Hora actual: {hora_actual}")
        print(f"   🎯 Buscando clases que inicien a las: {hora_proxima}")
        print(f"   🏫 Salón configurado: {salon_configurado}")
        
        if not salon_configurado:
            print("   [!] No hay salón configurado - saltando verificación")
            return []
        
        # Obtener todos los cursos
        cursos_ref = db.collection('courses')
        cursos = cursos_ref.get()
        
        for curso_doc in cursos:
            curso_id = curso_doc.id
            curso_data = curso_doc.to_dict()
            
            # CASO 1: Verificar grupos (subcolección)
            try:
                groups_ref = db.collection('courses').document(curso_id).collection('groups')
                groups = groups_ref.get()
                
                if groups:
                    for group_doc in groups:
                        group_data = group_doc.to_dict()
                        schedule = group_data.get('schedule', [])
                        
                        for horario in schedule:
                            resultado = verificar_horario_exacto(
                                horario, dia_espanol, dia_ingles, 
                                hora_proxima, salon_configurado
                            )
                            
                            if resultado:
                                cursos_a_inicializar.append((curso_id, fecha_actual, resultado))
                                nombre_curso = curso_data.get('nameCourse', 'Sin nombre')
                                print(f"   ✅ Encontrado: {nombre_curso} ({curso_id}) - Inicia a las {resultado}")
                    
                    continue
            except Exception as e:
                pass
            
            # CASO 2: Schedule directo
            schedule = curso_data.get('schedule', [])
            
            for horario in schedule:
                resultado = verificar_horario_exacto(
                    horario, dia_espanol, dia_ingles,
                    hora_proxima, salon_configurado
                )
                
                if resultado:
                    cursos_a_inicializar.append((curso_id, fecha_actual, resultado))
                    nombre_curso = curso_data.get('nameCourse', 'Sin nombre')
                    print(f"   ✅ Encontrado: {nombre_curso} ({curso_id}) - Inicia a las {resultado}")
        
        if not cursos_a_inicializar:
            print(f"   ℹ️ No hay cursos iniciando a las {hora_proxima} en {salon_configurado}")
        
        return cursos_a_inicializar
        
    except Exception as e:
        print(f"   ❌ ERROR obteniendo cursos: {e}")
        return []

def verificar_horario_exacto(horario, dia_espanol, dia_ingles, hora_buscada, salon_requerido):
    """
    Verifica si un horario coincide EXACTAMENTE con la hora buscada, día y salón.
    
    Args:
        horario: Diccionario con day, iniTime, classroom
        dia_espanol: Día en español (ej: "Lunes")
        dia_ingles: Día en inglés (ej: "Monday")
        hora_buscada: Hora exacta a buscar (formato "HH:MM")
        salon_requerido: Salón configurado
        
    Returns:
        str: Hora de inicio si coincide exactamente, None si no
    """
    try:
        dia_horario = horario.get('day', '')
        hora_inicio_str = horario.get('iniTime', '00:00')
        classroom = horario.get('classroom', '').strip()
        
        # 1. Verificar salón
        if classroom != salon_requerido:
            return None
        
        # 2. Verificar día
        if dia_horario != dia_espanol and dia_horario != dia_ingles:
            return None
        
        # 3. Verificar hora EXACTA
        if hora_inicio_str == hora_buscada:
            return hora_inicio_str
        
        return None
        
    except Exception as e:
        return None

def inicializar_documento_asistencia(course_id, fecha):
    """
    Crea el documento de asistencia con todos los estudiantes del curso en estado 'Ausente'.
    
    Args:
        course_id: ID del curso
        fecha: Fecha en formato YYYY-MM-DD
        
    Returns:
        bool: True si se inicializó correctamente
    """
    try:
        print(f"\n{'='*70}")
        print(f"📋 INICIALIZANDO DOCUMENTO DE ASISTENCIA")
        print(f"{'='*70}")
        print(f"   Curso: {course_id}")
        print(f"   Fecha: {fecha}")
        
        # Verificar si ya fue inicializado
        clave_documento = f"{course_id}_{fecha}"
        if clave_documento in documentos_inicializados:
            print(f"   [!] Ya fue inicializado previamente en esta sesión")
            return True
        
        # Referencia al documento de asistencia
        asistencia_ref = db.collection('courses').document(course_id).collection('assistances').document(fecha)
        
        # Verificar si ya existe en Firebase
        asistencia_doc = asistencia_ref.get()
        if asistencia_doc.exists:
            print(f"   [!] El documento ya existe en Firebase")
            documentos_inicializados.add(clave_documento)
            return True
        
        # Obtener información del curso
        curso_ref = db.collection('courses').document(course_id)
        curso_doc = curso_ref.get()
        
        if not curso_doc.exists:
            print(f"   ❌ ERROR: Curso {course_id} no existe")
            return False
        
        curso_data = curso_doc.to_dict()
        estudiantes_ids = curso_data.get('estudianteID', [])
        
        if not estudiantes_ids or len(estudiantes_ids) == 0:
            print(f"   ⚠️ ADVERTENCIA: No hay estudiantes inscritos en el curso")
            print(f"   ℹ️ Creando documento vacío")
            asistencia_ref.set({})
            documentos_inicializados.add(clave_documento)
            return True
        
        print(f"   📚 Estudiantes inscritos: {len(estudiantes_ids)}")
        
        # Crear diccionario con todos los estudiantes en estado "Ausente"
        datos_asistencia = {}
        
        for estudiante_id in estudiantes_ids:
            datos_asistencia[estudiante_id] = {
                'estadoAsistencia': 'Ausente',
                'horaRegistro': None,
                'late': False
            }
        
        # Crear el documento en Firebase
        asistencia_ref.set(datos_asistencia)
        
        # Marcar como inicializado
        documentos_inicializados.add(clave_documento)
        
        print(f"   ✅ Documento creado exitosamente")
        print(f"   📊 Total de estudiantes: {len(estudiantes_ids)}")
        print(f"   📝 Estado inicial: Ausente")
        print(f"{'='*70}\n")
        
        return True
        
    except Exception as e:
        print(f"   ❌ ERROR inicializando documento: {e}")
        import traceback
        traceback.print_exc()
        return False


def limpiar_registros_antiguos():
    """
    Limpia el registro de documentos inicializados de días anteriores.
    Se ejecuta una vez al día a la medianoche.
    """
    global documentos_inicializados
    fecha_actual = datetime.now().strftime('%Y-%m-%d')
    
    # Filtrar solo los documentos del día actual
    documentos_inicializados = {
        doc for doc in documentos_inicializados 
        if doc.endswith(fecha_actual)
    }
    
    print(f"\n🧹 Limpieza de registros antiguos completada")
    print(f"   Documentos en memoria: {len(documentos_inicializados)}")


def tarea_programada(salon_configurado_callback):
    """
    Tarea que se ejecuta cada minuto para verificar cursos a inicializar.
    OPTIMIZACIÓN: Solo procesa cuando los minutos terminan en :54
    
    Args:
        salon_configurado_callback: Función que retorna el salón actual configurado
    """
    print("\n" + "="*70)
    print("🚀 SCHEDULER DE ASISTENCIA INICIADO")
    print("="*70)
    print("   ⏰ Verificación: Solo cuando los minutos terminen en :54")
    print("   📋 Inicialización: Automática en horas en punto (:00)")
    print("   🧹 Limpieza: Diariamente a la medianoche")
    print("   💡 Optimizado: Procesamiento mínimo de CPU")
    print("="*70 + "\n")
    
    ultima_limpieza = datetime.now().date()
    ultimo_minuto_procesado = -1  # Para evitar procesar dos veces el mismo :54
    
    while True:
        try:
            ahora = datetime.now()
            minuto_actual = ahora.minute
            
            # Verificar si es un nuevo día para limpiar registros
            hoy = ahora.date()
            if hoy > ultima_limpieza:
                limpiar_registros_antiguos()
                ultima_limpieza = hoy
            
            # ⭐ OPTIMIZACIÓN: Solo procesar cuando los minutos terminen en :54
            if minuto_actual == 54 and ultimo_minuto_procesado != 54:
                print(f"\n🔔 [{ahora.strftime('%H:%M:%S')}] ¡Minuto :54 detectado! Iniciando verificación...")
                
                # Obtener salón configurado
                salon_actual = salon_configurado_callback()
                
                if salon_actual:
                    # Buscar cursos próximos a iniciar
                    cursos = obtener_cursos_proximos_a_iniciar(salon_actual)
                    
                    # Inicializar documentos
                    for curso_id, fecha, hora_inicio in cursos:
                        inicializar_documento_asistencia(curso_id, fecha)
                    
                    if not cursos:
                        print(f"   ℹ️ No hay cursos a inicializar en este momento")
                else:
                    print(f"   ⚠️ No hay salón configurado - saltando verificación")
                
                # Marcar que ya procesamos este :54
                ultimo_minuto_procesado = 54
                print(f"✅ Verificación completada. Próxima verificación en ~60 minutos.\n")
            
            # Resetear cuando pasamos del minuto :54
            elif minuto_actual != 54:
                ultimo_minuto_procesado = -1
            
        except Exception as e:
            print(f"\n❌ ERROR en tarea programada: {e}")
            import traceback
            traceback.print_exc()
        
        # Esperar 60 segundos antes de la siguiente verificación
        time.sleep(60)

def iniciar_scheduler(salon_configurado_callback):
    """
    Inicia el scheduler en un hilo separado.
    
    Args:
        salon_configurado_callback: Función que retorna el salón actual
    """
    thread = threading.Thread(
        target=tarea_programada, 
        args=(salon_configurado_callback,),
        daemon=True  # Se cierra cuando termina el programa principal
    )
    thread.start()
    return thread