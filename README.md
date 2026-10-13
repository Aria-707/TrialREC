# TrialREC

Kiosco de reconocimiento facial para registrar la asistencia académica de estudiantes. El navegador captura el fotograma, pero la identificación y la escritura de la asistencia ocurren en el servidor.

## Arquitectura

```text
PC del salón
    │ navegador y cámara
    │ WireGuard / HTTP 5000
    ▼
BFF de Flask
    │ pedido firmado
    │ localhost 5001
    ▼
Reconocimiento con YuNet y SFace
    │
    └── Firestore
```

`kiosk.py` inicia dos procesos en el mismo computador:

- El BFF sirve las pantallas en `127.0.0.1:5000` y, cuando el adaptador existe, en la dirección privada de WireGuard.
- El reconocimiento escucha únicamente en `127.0.0.1:5001`. Ese puerto no se publica a la red.
- Cada navegador conserva su salón en una cookie firmada. El correo de administrador solo se usa para elegir el salón o cambiar el modo evento.

## Requisitos

- Windows 10 u 11 de 64 bits en el servidor del piloto.
- Python 3.10 con `pip`.
- Un proyecto de Firebase con Authentication y Firestore.
- Una cámara disponible en cada PC que use el kiosco.
- WireGuard en el servidor y en cada PC autorizado cuando se accede desde otro equipo.

Las versiones de las dependencias de Python están fijadas en `requirements.txt`.

## Instalación

Desde la raíz del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Crear el archivo local de configuración solo si todavía no existe:

```powershell
Copy-Item .env.example .env
```

Si `.env` ya existe, no se reemplaza.

## Archivos privados

Estos archivos permanecen en la raíz y no se suben al repositorio:

- `.env`: configuración de ejecución y secretos.
- `asistenciaconreconocimiento-firebase-adminsdk.json`: cuenta de servicio, cuando `FIREBASE_SERVICE_ACCOUNT` está vacía.
- `firebase_config_public.json`: configuración pública del cliente de Firebase.
- `encryption.key` y `encryption_salt.key`: cifrado de las fotos.
- `classroom_config.txt`: salón adicional que el scheduler vigila al arrancar, si existe.
- `audit_log.json`: auditoría local.

No se debe rotar `encryption.key` mientras existan fotos cifradas con esa clave.

## Configuración

`.env.example` contiene todas las variables disponibles y valores de referencia. Antes de ejecutar:

1. Cambiar `KIOSK_HMAC_SECRET` en `.env`.
2. Mantener `KIOSK_DEBUG=false` fuera del desarrollo.
3. Confirmar que los modelos estén en `app/recognition/models/`.
4. Dejar `FERNET_KEY` vacía para continuar usando `encryption.key`, o proporcionar la misma clave que cifró las fotos existentes.
5. Proporcionar la cuenta de servicio mediante el JSON local o `FIREBASE_SERVICE_ACCOUNT` en base64.

`DEMO_MODE=false` deja el modo evento apagado al arrancar. Cuando es `true`, las variables del evento son obligatorias y sus horas usan el formato de 24 horas `HH:MM`.

## Ejecución

Con el túnel encendido, si se va a usar:

```powershell
.\.venv\Scripts\python.exe kiosk.py
```

Direcciones:

- Servidor local: `http://127.0.0.1:5000`
- PCs autorizados por WireGuard: `http://10.0.0.1:5000`
- Reconocimiento interno: `http://127.0.0.1:5001`

La primera visita desde un navegador pide una cuenta administradora y el salón. Después, ese navegador abre el kiosco del salón guardado. El modo evento es un solo interruptor para todo el servidor.

Al arrancar se eliminan las personas, vectores y aceptaciones temporales de eventos anteriores. Los cursos y la asistencia ya guardada permanecen.

## Pruebas

Las pruebas no abren la cámara ni escriben en Firestore:

```powershell
$env:PYTHONIOENCODING = "utf-8"
.\.venv\Scripts\python.exe -m pytest tests/test_kiosk_gate.py tests/test_schedule.py tests/test_match.py -q
```

## Estructura

```text
app/
  http/          BFF, firma y rutas HTTP
  recognition/   arranque, cámara, YuNet, SFace y memoria de la sesión
  attendance/    asistencia, horario y scheduler
  classroom/     salones vigilados por el servidor
  events/        modo evento
  consent/       aceptación legal y revocación biométrica
  firebase/      cliente administrativo y vectores cifrados
  security/      cifrado y auditoría
config/          lectura y validación de .env
templates/       pantallas del kiosco
static/          JavaScript, estilos e imágenes
tests/           pruebas de acceso, horario y emparejamiento
kiosk.py         punto de entrada
```

## Seguridad operativa

- No publicar el puerto `5001`.
- No abrir el puerto `5000` a la red Wi-Fi general; los PCs del salón entran por WireGuard.
- No subir `.env`, cuentas de servicio, claves, fotos, auditorías ni archivos `.conf`.
- Cada PC de salón debe tener su propio archivo de acceso de WireGuard.
- El firewall del servidor permite UDP `51820` para WireGuard y TCP `5000` desde la red privada del túnel.
- Los cambios de código se cargan al reiniciar. El reinicio debe hacerse fuera de una toma en curso.
