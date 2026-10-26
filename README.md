# 📷 Proyectos con ESP32-CAM y Visión por Computadora

Conjunto de proyectos que usan una **ESP32-CAM** como cámara de video por Wi-Fi y **Python** para procesar la imagen en tiempo real con visión por computadora e inteligencia artificial.

| # | Proyecto | Descripción |
|---|----------|-------------|
| 1 | **Reconocimiento de emociones** | Detecta el rostro y clasifica la emoción: feliz, enojado, triste o normal. |
| 2 | **Detección de objetos + juego "Encuentra el objeto"** | Detección en tiempo real con YOLOv8 y un juego interactivo. |
| 3 | **Alerta intruso** | Si el rostro está cubierto (máscara, barbijo u otro objeto), activa una alerta visual y sonora. |

---

## 🧰 Materiales

- 1 × **ESP32-CAM** (módulo AI-Thinker con cámara OV2640)
- 1 × **Placa expansora / base programadora** para ESP32-CAM con **salida USB tipo C** (se usa en lugar del adaptador FTDI, ya que incluye el puerto USB y los botones de RESET/BOOT)
- 1 × Cable **USB tipo C** (de datos, no solo de carga)
- Red **Wi-Fi de 2.4 GHz** (la ESP32 no funciona en 5 GHz)
- 1 × Computadora con Windows, macOS o Linux
- *(Opcional, Proyecto 3)* Buzzer o LED externo

---

## 🔌 Conexión del hardware

1. Coloca la **ESP32-CAM** sobre la **placa expansora** respetando la orientación de los pines (la antena/cámara hacia el lado indicado en la placa).
2. Conecta la placa expansora a la computadora con el cable **USB-C**.
3. Verifica que el sistema detecte un nuevo puerto COM (Windows) o `/dev/ttyUSB0` (Linux). Si no aparece, instala el driver del chip de la placa (**CH340** o **CP2102**, según el modelo de tu expansor).

---

## ⚙️ Parte 1: Configurar Arduino IDE y cargar el firmware

### 1.1 Instalar Arduino IDE
Descarga e instala la última versión desde: https://www.arduino.cc/en/software

### 1.2 Agregar el soporte para placas ESP32
1. Abre Arduino IDE → **Archivo → Preferencias**.
2. En **"Gestor de URLs Adicionales de Tarjetas"** pega:
   ```
   https://espressif.github.io/arduino-esp32/package_esp32_index.json
   ```
3. Ve a **Herramientas → Placa → Gestor de tarjetas**.
4. Busca **`esp32`** (de *Espressif Systems*) e instala el paquete.

### 1.3 Descargar la plantilla (ejemplo) de la cámara
1. Ve a **Archivo → Ejemplos → ESP32 → Camera → CameraWebServer**.
2. Se abrirá el ejemplo con varios archivos (`CameraWebServer.ino`, `camera_pins.h`, etc.). Esa es la plantilla que usamos.

### 1.4 Configurar la plantilla
En `CameraWebServer.ino`:

1. Selecciona el modelo de cámara. Comenta todos y deja **solo** este activo:
   ```cpp
   #define CAMERA_MODEL_AI_THINKER // Has PSRAM
   ```
2. Escribe los datos de tu Wi-Fi:
   ```cpp
   const char* ssid     = "NOMBRE_DE_TU_WIFI";
   const char* password = "CONTRASEÑA_DE_TU_WIFI";
   ```

### 1.5 Configurar la placa y subir el código
En el menú **Herramientas**:

| Opción | Valor |
|--------|-------|
| Placa | **AI Thinker ESP32-CAM** |
| Puerto | El puerto COM de tu expansor |
| Upload Speed | 115200 |
| Partition Scheme | Huge APP (3MB No OTA/1MB SPIFFS) |

Luego pulsa **Subir (→)**.

> 💡 Si aparece el error `Failed to connect to ESP32`, mantén presionado el botón **BOOT/IO0** de la placa expansora mientras inicia la carga, y suéltalo cuando empiece a subir. Después presiona **RESET**.

### 1.5 Obtener la IP de la cámara
1. Abre el **Monitor Serie** (115200 baudios).
2. Presiona **RESET** en la placa.
3. Aparecerá un mensaje como:
   ```
   Camera Ready! Use 'http://192.168.1.12' to connect
   ```
4. Abre esa IP en el navegador para comprobar que la cámara funciona. En este proyecto la IP es `http://192.168.1.12`.

El **stream de video** (el que usa Python) está en:
```
http://192.168.1.12:81/stream
```

> ⚠️ La IP puede cambiar si reinicias el router. Anota la tuya y cámbiala en los scripts de Python (variable `URL`), o reserva una IP fija para la ESP32 en tu router.

---

## 🐍 Parte 2: Entorno virtual de Python

### 2.1 Requisitos
- **Python 3.9 – 3.11** (recomendado)
- `pip`

### 2.2 Crear y activar el entorno virtual

**Windows (PowerShell / CMD):**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2.3 Instalar las dependencias
```bash
pip install --upgrade pip
pip install opencv-python numpy requests ultralytics deepface pyttsx3
```

| Librería | Uso |
|----------|-----|
| `opencv-python` | Captura del stream y detección de rostros |
| `numpy` | Manejo de imágenes |
| `requests` | Comunicación con la cámara |
| `ultralytics` | YOLOv8 (detección de objetos) |
| `deepface` | Reconocimiento de emociones |
| `pyttsx3` | Voz de la alarma ("alerta, intruso") |

> Guarda las dependencias con `pip freeze > requirements.txt` para instalarlas luego con `pip install -r requirements.txt`.

---

## ▶️ Parte 3: Ejecutar los proyectos

Antes de ejecutar cualquier script:
1. Asegúrate de que la ESP32-CAM esté **encendida y conectada al mismo Wi-Fi** que tu computadora.
2. Verifica que la IP en el código coincida con la de tu cámara.
3. Activa el entorno virtual.

### Proyecto 1: Reconocimiento de emociones 😀😠😢😐
```bash
python emociones.py
```
Detecta el rostro en el video y muestra la emoción: **feliz, enojado, triste o normal**.

### Proyecto 2: Detección de objetos y juego "Encuentra el objeto" 🎯
```bash
python deteccion_objetos.py
```
- Usa **YOLOv8** (`yolov8n.pt`, modelo ligero para reducir el lag). El modelo se descarga automáticamente la primera vez.
- Interfaz con ventana de tamaño ajustable.
- **Juego "Encuentra el objeto":** el programa pide mostrar un objeto a la cámara. Objetos del juego:

  🐦 pájaro · ⌚ reloj · 📖 libro · 🖱️ mouse · 🎮 control · 📱 teléfono · ⚽ balón · 🪙 moneda · 🍼 botella

### Proyecto 3: Alerta intruso 🚨
```bash
python alerta_intruso.py
```
- Si el **rostro es visible** → lo reconoce normalmente.
- Si el rostro está **cubierto** (máscara, barbijo u otro objeto) → se activa una **alerta visual** en pantalla y suena la voz: *"Alerta, intruso"*.

> Para cerrar cualquier ventana de video presiona la tecla **`q`**.

---

## 🗂️ Estructura del proyecto

```
📦 esp32-cam-proyectos
 ┣ 📂 firmware
 ┃ ┗ 📜 CameraWebServer.ino
 ┣ 📜 emociones.py
 ┣ 📜 deteccion_objetos.py
 ┣ 📜 alerta_intruso.py
 ┣ 📜 requirements.txt
 ┗ 📜 README.md
```
*(Ajusta los nombres si tus archivos se llaman distinto.)*

---

## 🛠️ Solución de problemas

| Problema | Posible solución |
|----------|------------------|
| No aparece el puerto COM | Instala el driver CH340/CP2102 y usa un cable USB-C **de datos** |
| `Failed to connect to ESP32` | Mantén presionado **BOOT** al subir el código y luego pulsa **RESET** |
| No conecta al Wi-Fi | Usa una red de **2.4 GHz** y revisa SSID y contraseña |
| No abre el video en Python | Revisa la IP y usa `:81/stream`; confirma que PC y ESP32 estén en la misma red |
| Mucho lag / imagen lenta | Baja la resolución en el firmware (ej. `FRAMESIZE_QVGA`), usa `yolov8n.pt` y procesa un frame de cada 2 o 3 |
| La cámara se reinicia sola | Usa un puerto USB con suficiente corriente (la ESP32-CAM consume bastante) |

---

## 🧪 Tecnologías

![Arduino](https://img.shields.io/badge/Arduino_IDE-00979D?style=for-the-badge&logo=arduino&logoColor=white)
![ESP32](https://img.shields.io/badge/ESP32--CAM-E7352C?style=for-the-badge&logo=espressif&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)

---

## 👤 Autor

Desarrollado por **eydam**.

## 📄 Licencia

Proyecto de uso educativo. Puedes usarlo y modificarlo libremente citando al autor.
