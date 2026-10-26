"""esp32_comun.py
Utilidades compartidas por los proyectos con ESP32-CAM:
cámara en hilo, ventana con estilo visual, helpers de dibujo y audio (sonidos + voz).
"""

import queue
import threading
import time
from functools import lru_cache
from types import SimpleNamespace
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from PIL import ImageFont

try:
    import winsound
except ImportError:  # no es Windows
    winsound = None

URL_POR_DEFECTO = "IP AQUI "

# ===== Colores (RGB) =====
BG = (14, 16, 24)
HEADER_BG = (20, 23, 34)
CARD = (24, 27, 40)
TRACK = (44, 48, 66)
MUTED = (140, 150, 175)
BLANCO = (240, 244, 252)
ACENTO = (0, 208, 168)
VERDE = (46, 204, 113)
ROJO = (255, 70, 80)
AMBAR = (255, 190, 60)
DARK = (14, 16, 24)
PALETA = [
    (255, 99, 132), (54, 162, 235), (255, 206, 86), (75, 192, 192),
    (153, 102, 255), (255, 159, 64), (46, 204, 113), (231, 76, 60),
    (26, 188, 156), (241, 196, 15),
]


# ============================================================
#  CÁMARA
# ============================================================
class ESP32CamStream:
    """Lee el stream MJPEG en un hilo y conserva solo el último frame."""

    def __init__(self, url):
        self.url = url
        self.frame = None
        self.frame_id = 0
        self.stopped = False
        self.cond = threading.Condition()
        self.thread = threading.Thread(target=self._update, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _update(self):
        SOI, EOI = b"\xff\xd8", b"\xff\xd9"
        while not self.stopped:
            try:
                with requests.get(self.url, stream=True, timeout=5) as resp:
                    resp.raise_for_status()
                    buf = bytearray()
                    for chunk in resp.iter_content(chunk_size=2048):
                        if self.stopped:
                            return
                        buf += chunk
                        while True:
                            a = buf.find(SOI)
                            if a == -1:
                                del buf[:-1]
                                break
                            b = buf.find(EOI, a + 2)
                            if b == -1:
                                if a > 0:
                                    del buf[:a]
                                break
                            jpg = bytes(buf[a : b + 2])
                            del buf[: b + 2]
                            img = cv2.imdecode(
                                np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR
                            )
                            if img is None:
                                continue
                            with self.cond:
                                self.frame = img
                                self.frame_id += 1
                                self.cond.notify_all()
            except Exception:
                time.sleep(1)

    def wait_new(self, last_id, timeout=1.0):
        with self.cond:
            self.cond.wait_for(
                lambda: self.frame_id != last_id or self.stopped, timeout
            )
            if self.frame_id == last_id:
                return None, last_id
            return self.frame, self.frame_id

    def stop(self):
        self.stopped = True
        with self.cond:
            self.cond.notify_all()


def configurar_camara(url, framesize, quality):
    p = urlparse(url)
    base = f"{p.scheme}://{p.hostname}"
    for var, val in (("framesize", framesize), ("quality", quality)):
        if val is None:
            continue
        try:
            requests.get(f"{base}/control", params={"var": var, "val": val}, timeout=3)
            print(f"Cámara: {var} = {val}")
        except Exception as e:
            print(f"No se pudo configurar {var}: {e}")


def iniciar_camara(url, framesize, quality):
    configurar_camara(url, framesize, quality)
    print(f"Conectando al stream: {url}")
    cam = ESP32CamStream(url).start()
    frame, last_id, intentos = None, 0, 0
    while frame is None:
        frame, last_id = cam.wait_new(last_id, timeout=1.0)
        intentos += 1
        if frame is None and intentos % 5 == 0:
            print("Esperando imagen de la cámara... revisa la IP, el WiFi y que el stream esté activo.")
    return cam, frame, last_id


class HiloAnalisis(threading.Thread):
    """Analiza siempre el frame más reciente en un hilo aparte. Sobrescribe procesar()."""

    def __init__(self, cam):
        super().__init__(daemon=True)
        self.cam = cam
        self.lock = threading.Lock()
        self.resultado = None
        self.ms = 0.0
        self.detenido = False

    def procesar(self, frame):
        raise NotImplementedError

    def obtener(self):
        with self.lock:
            return self.resultado, self.ms

    def run(self):
        last_id = 0
        while not self.detenido:
            frame, last_id = self.cam.wait_new(last_id, timeout=1.0)
            if frame is None:
                continue
            try:
                t0 = time.time()
                res = self.procesar(frame)
                ms = (time.time() - t0) * 1000
                with self.lock:
                    self.resultado, self.ms = res, ms
            except Exception as e:
                print(f"Error en el análisis: {e}")
                time.sleep(0.5)


def cargar_yolo(nombre, imgsz):
    import torch
    from ultralytics import YOLO

    usa_cuda = torch.cuda.is_available()
    device = 0 if usa_cuda else "cpu"
    print(f"Cargando modelo {nombre} en {'GPU' if usa_cuda else 'CPU'}...")
    model = YOLO(nombre)
    model.predict(np.zeros((imgsz, imgsz, 3), np.uint8), imgsz=imgsz, device=device, verbose=False)
    return model, device


def agregar_args_comunes(parser, imgsz, conf):
    parser.add_argument("--url", default=URL_POR_DEFECTO, help="URL del stream de la ESP32-CAM")
    parser.add_argument("--model", default="yolov8n.pt", help="Modelo YOLO")
    parser.add_argument("--imgsz", type=int, default=imgsz, help="Tamaño de inferencia")
    parser.add_argument("--conf", type=float, default=conf, help="Confianza mínima")
    parser.add_argument("--tam", type=float, default=0.5,
                        help="Tamaño de la ventana (fracción del ancho de pantalla, 0.3 a 1.0)")
    parser.add_argument("--ancho", type=int, default=None,
                        help="Ancho de la ventana en píxeles (reemplaza a --tam)")
    parser.add_argument("--framesize", type=int, default=None,
                        help="Resolución de la cámara: 5=QVGA(320x240), 8=VGA(640x480)")
    parser.add_argument("--quality", type=int, default=None,
                        help="Calidad JPEG de la cámara (10-63, más alto = más liviano)")
    parser.add_argument("--silencio", action="store_true", help="Iniciar sin sonido")
    parser.add_argument("--sin-voz", action="store_true", help="Sin voz (solo sonidos)")


# ============================================================
#  AUDIO (sonidos + voz en un hilo aparte)
# ============================================================
class Audio:
    def __init__(self, voz=True, activo=True):
        self.activo = activo
        self.usar_voz = voz
        self._q = queue.Queue()
        self._sirena = threading.Event()
        self._frase = ""
        threading.Thread(target=self._run, daemon=True).start()

    # --- API pública ---
    def ok(self):
        self._enviar(("tonos", [(660, 110), (880, 110), (1175, 180)]))

    def fallo(self):
        self._enviar(("tonos", [(330, 200), (247, 350)]))

    def fin(self):
        self._enviar(("tonos", [(523, 120), (659, 120), (784, 120), (1047, 300)]))

    def decir(self, texto):
        self._enviar(("voz", texto))

    def sirena(self, encender, frase=""):
        self._frase = frase
        if encender:
            self._sirena.set()
        else:
            self._sirena.clear()

    # --- internos ---
    def _enviar(self, tarea):
        if self.activo:
            self._q.put(tarea)

    @staticmethod
    def _crear_motor():
        try:
            import pyttsx3
        except ImportError:
            print("Tip: para activar la voz instala pyttsx3  ->  pip install pyttsx3")
            return None
        try:
            try:
                import pythoncom
                pythoncom.CoInitialize()
            except Exception:
                pass
            motor = pyttsx3.init()
            motor.setProperty("rate", 165)
            for v in motor.getProperty("voices"):
                info = f"{v.name} {v.id} {getattr(v, 'languages', '')}".lower()
                if "spanish" in info or "español" in info or "es-" in info or "_es" in info:
                    motor.setProperty("voice", v.id)
                    break
            return motor
        except Exception as e:
            print(f"No se pudo iniciar la voz: {e}")
            return None

    def _tono(self, frecuencia, ms):
        if not self.activo:
            return
        if winsound:
            try:
                winsound.Beep(int(frecuencia), int(ms))
            except Exception:
                time.sleep(ms / 1000)
        else:
            print("\a", end="", flush=True)
            time.sleep(ms / 1000)

    def _hablar(self, motor, texto):
        if not self.activo:
            return
        try:
            motor.say(texto)
            motor.runAndWait()
        except Exception as e:
            print(f"Error de voz: {e}")

    def _run(self):
        motor = self._crear_motor() if self.usar_voz else None
        while True:
            try:
                tarea = self._q.get(timeout=0.1)
            except queue.Empty:
                tarea = None
            if tarea:
                tipo, dato = tarea
                if tipo == "tonos":
                    for f, ms in dato:
                        self._tono(f, ms)
                elif tipo == "voz" and motor:
                    self._hablar(motor, dato)
            if self._sirena.is_set() and self.activo:
                for f in (1400, 900, 1400, 900):
                    if not (self._sirena.is_set() and self.activo):
                        break
                    self._tono(f, 220)
                if motor and self._frase and self._sirena.is_set() and self.activo:
                    self._hablar(motor, self._frase)


# ============================================================
#  FUENTES Y TEXTO
# ============================================================
@lru_cache(maxsize=128)
def fuente(negrita, size):
    nombres = (["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"] if negrita
               else ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"])
    for nombre in nombres:
        try:
            return ImageFont.truetype(nombre, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def alto_fuente(f):
    return getattr(f, "size", 16)


def crear_fuentes(k):
    def s(v):
        return max(11, int(v * k))

    return {
        "titulo": fuente(True, s(24)), "sub": fuente(False, s(13)),
        "chip": fuente(True, s(14)), "panel_t": fuente(True, s(13)),
        "num": fuente(True, s(46)), "item": fuente(False, s(16)),
        "cont": fuente(True, s(16)), "small": fuente(False, s(12)),
        "grande": fuente(True, s(30)),
    }


def ajustar(d, texto, negrita, size, max_w, minimo=12):
    """Devuelve una fuente que hace caber el texto en max_w píxeles."""
    while size > minimo and d.textlength(texto, font=fuente(negrita, size)) > max_w:
        size -= 1
    return fuente(negrita, size)


def T(d, xy, texto, f, color, anchor="la"):
    try:
        d.text(xy, texto, font=f, fill=color, anchor=anchor)
    except Exception:
        d.text(xy, texto, font=f, fill=color)


def recortar(d, texto, f, max_w):
    if d.textlength(texto, font=f) <= max_w:
        return texto
    while len(texto) > 3 and d.textlength(texto + "…", font=f) > max_w:
        texto = texto[:-1]
    return texto + "…"


def envolver(d, texto, f, max_w):
    lineas, actual = [], ""
    for palabra in texto.split():
        prueba = (actual + " " + palabra).strip()
        if d.textlength(prueba, font=f) <= max_w:
            actual = prueba
        else:
            if actual:
                lineas.append(actual)
            actual = palabra
    if actual:
        lineas.append(actual)
    return lineas


def parrafo(d, texto, f, color, x, y, max_w, interlineado):
    for linea in envolver(d, texto, f, max_w):
        T(d, (x, y), linea, f, color)
        y += interlineado
    return y


def mezclar(c1, c2, t):
    """t=0 -> c1, t=1 -> c2."""
    return tuple(int(a * (1 - t) + b * t) for a, b in zip(c1, c2))


# ============================================================
#  VENTANA Y LAYOUT
# ============================================================
def tam_pantalla():
    try:
        import tkinter as tk
        r = tk.Tk()
        r.withdraw()
        w, h = r.winfo_screenwidth(), r.winfo_screenheight()
        r.destroy()
        return w, h
    except Exception:
        return 1366, 768


def calcular_layout(frac, fw, fh, sw, sh, full):
    total_w = int(sw * (1.0 if full else frac))
    max_h = sh if full else int(sh * 0.92)
    while True:
        k = min(max(total_w / 1000, 0.8), 1.6)
        pad = int(12 * k)
        header_h = int(58 * k)
        footer_h = int(34 * k)
        pw = int(total_w * 0.27)
        vw = total_w - pw - 3 * pad
        vh = int(vw * fh / fw)
        total_h = header_h + pad + vh + footer_h
        if total_h <= max_h or total_w <= 480:
            break
        total_w = int(total_w * 0.97)

    vx, vy = pad, header_h + pad
    px = vx + vw + pad
    pw = total_w - px - pad

    base = np.empty((total_h, total_w, 3), np.uint8)
    base[:] = BG
    base[:header_h] = HEADER_BG
    base[header_h - 3 : header_h] = ACENTO

    L = SimpleNamespace(
        k=k, pad=pad, header_h=header_h, footer_h=footer_h,
        total_w=total_w, total_h=total_h,
        vx=vx, vy=vy, vw=vw, vh=vh,
        px=px, py=vy, pw=pw, ph=vh, base=base,
    )
    if full:
        L.ox, L.oy = (sw - total_w) // 2, (sh - total_h) // 2
        L.fondo = np.zeros((sh, sw, 3), np.uint8)
    return L


class Ventana:
    """Maneja tamaño, pantalla completa, avisos y teclas comunes (+, -, F, Q)."""

    def __init__(self, nombre, tam=0.5, ancho=None):
        self.nombre = nombre
        self.sw, self.sh = tam_pantalla()
        frac = (ancho / self.sw) if ancho else tam
        self.frac = min(max(frac, 0.3), 1.0)
        self.full = False
        self.L = self.F = self._key = None
        self._aviso = None
        cv2.namedWindow(nombre, cv2.WINDOW_NORMAL)

    def preparar(self, frame):
        key = (round(self.frac, 3), self.full, frame.shape[1], frame.shape[0])
        if key != self._key:
            self.L = calcular_layout(self.frac, frame.shape[1], frame.shape[0],
                                     self.sw, self.sh, self.full)
            self.F = crear_fuentes(self.L.k)
            self._key = key
            if not self.full:
                cv2.resizeWindow(self.nombre, self.L.total_w, self.L.total_h)
                cv2.moveWindow(self.nombre,
                               max(0, (self.sw - self.L.total_w) // 2),
                               max(0, (self.sh - self.L.total_h) // 2 - 30))
        return self.L, self.F

    def toast(self, texto, seg=1.8):
        self._aviso = (texto, time.time() + seg)

    def dibujar_toast(self, d):
        if not self._aviso or time.time() >= self._aviso[1]:
            return
        L, F = self.L, self.F
        texto = self._aviso[0]
        h = int(30 * L.k)
        w = int(d.textlength(texto, font=F["chip"])) + int(28 * L.k)
        x = L.vx + (L.vw - w) // 2
        y = L.vy + L.vh - h - int(14 * L.k)
        d.rounded_rectangle((x, y, x + w, y + h), radius=h // 2, fill=ACENTO)
        T(d, (x + w // 2, y + h // 2), texto, F["chip"], DARK, "mm")

    def mostrar(self, img_pil):
        """Muestra la imagen y devuelve la versión BGR (para guardar fotos)."""
        bgr = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)
        if self.full:
            L = self.L
            L.fondo[L.oy : L.oy + L.total_h, L.ox : L.ox + L.total_w] = bgr
            cv2.imshow(self.nombre, L.fondo)
        else:
            cv2.imshow(self.nombre, bgr)
        return bgr

    def tecla(self):
        """Procesa +, -, F. Devuelve 'q' para salir, otra tecla en minúscula, o None."""
        k = cv2.waitKey(1) & 0xFF
        if k == 255:
            return None
        if k == 27:
            return "q"
        ch = chr(k).lower()
        if ch in ("+", "="):
            self.frac = min(1.0, self.frac + 0.05)
            self.toast(f"Tamaño: {int(self.frac * 100)}% de la pantalla")
            return None
        if ch == "-":
            self.frac = max(0.3, self.frac - 0.05)
            self.toast(f"Tamaño: {int(self.frac * 100)}% de la pantalla")
            return None
        if ch == "f":
            self.full = not self.full
            cv2.setWindowProperty(
                self.nombre, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if self.full else cv2.WINDOW_NORMAL,
            )
            return None
        return ch


# ============================================================
#  DIBUJO
# ============================================================
def video_en_canvas(frame, L):
    video = cv2.resize(frame, (L.vw, L.vh), interpolation=cv2.INTER_CUBIC)
    canvas = L.base.copy()
    canvas[L.vy : L.vy + L.vh, L.vx : L.vx + L.vw] = cv2.cvtColor(video, cv2.COLOR_BGR2RGB)
    return canvas


def dibujar_caja(img, x1, y1, x2, y2, color, grosor):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    largo = max(12, int(min(x2 - x1, y2 - y1) * 0.22))
    for px, py, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (px, py), (px + dx * largo, py), color, grosor + 1, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py + dy * largo), color, grosor + 1, cv2.LINE_AA)


def dibujar_chip(d, L, F, X1, Y1, col, label, fg=DARK):
    h = int(26 * L.k)
    padx = int(9 * L.k)
    w = int(d.textlength(label, font=F["chip"])) + 2 * padx
    x = max(L.vx, min(X1, L.vx + L.vw - w))
    y = Y1 - h - 3
    if y < L.vy:
        y = Y1 + 3
    d.rounded_rectangle((x, y, x + w, y + h), radius=h // 3, fill=col)
    T(d, (x + padx, y + h // 2), label, F["chip"], fg, "lm")


def dibujar_banner(d, L, texto, color, y, size=30, fg=DARK):
    """Banner centrado sobre el video. Devuelve su alto."""
    padx, pady = int(22 * L.k), int(10 * L.k)
    f = ajustar(d, texto, True, max(12, int(size * L.k)), L.vw - 4 * padx)
    bb = d.textbbox((0, 0), texto, font=f)
    w, h = (bb[2] - bb[0]) + 2 * padx, (bb[3] - bb[1]) + 2 * pady
    x = L.vx + (L.vw - w) // 2
    d.rounded_rectangle((x, y, x + w, y + h), radius=h // 2, fill=color)
    T(d, (x + w // 2, y + h // 2), texto, f, fg, "mm")
    return h


def dibujar_header(d, L, F, titulo, subtitulo, acento=ACENTO):
    k = L.k
    d.rectangle((0, L.header_h - 3, L.total_w, L.header_h - 1), fill=acento)
    x = L.pad + int(4 * k)
    T(d, (x, int(6 * k)), titulo, F["titulo"], BLANCO)
    T(d, (x, int(36 * k)), subtitulo, F["sub"], MUTED)

    cy = (L.header_h - 3) // 2
    reloj = time.strftime("%H:%M:%S")
    xr = L.total_w - L.pad - int(4 * k)
    T(d, (xr, cy), reloj, F["cont"], BLANCO, "rm")
    wr = d.textlength(reloj, font=F["cont"])
    xl = xr - wr - int(18 * k)
    T(d, (xl, cy), "EN VIVO", F["cont"], MUTED, "rm")
    wl = d.textlength("EN VIVO", font=F["cont"])
    r = max(4, int(5 * k))
    cx = xl - wl - int(12 * k)
    color = ROJO if int(time.time() * 1.5) % 2 == 0 else (110, 40, 48)
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)


def dibujar_footer(d, L, F, izq, der="Procesamiento local en este equipo"):
    cy = (L.vy + L.vh + L.total_h) // 2
    wd = d.textlength(der, font=F["small"])
    T(d, (L.total_w - L.pad, cy), der, F["small"], MUTED, "rm")
    T(d, (L.vx, cy), recortar(d, izq, F["small"], L.total_w - 3 * L.pad - wd),
      F["small"], MUTED, "lm")


def dibujar_card(d, L):
    d.rounded_rectangle((L.px, L.py, L.px + L.pw, L.py + L.ph),
                        radius=int(14 * L.k), fill=CARD)
    return int(16 * L.k)


def barra(d, x, y, w, h, frac, color):
    frac = min(max(frac, 0.0), 1.0)
    d.rounded_rectangle((x, y, x + w, y + h), radius=h // 2, fill=TRACK)
    if frac > 0:
        d.rounded_rectangle((x, y, x + max(h, int(w * frac)), y + h), radius=h // 2, fill=color)


def dibujar_stats(d, L, F, fps, ms):
    """Dibuja FPS y ms al fondo del panel. Devuelve la y donde empieza esa zona."""
    k, ip = L.k, int(16 * L.k)
    x0, x1 = L.px + ip, L.px + L.pw - ip
    y_fin = L.py + L.ph - ip - int(60 * k)
    d.line((x0, y_fin, x1, y_fin), fill=TRACK, width=1)
    r1, r2 = y_fin + int(18 * k), y_fin + int(42 * k)
    T(d, (x0, r1), "Video", F["small"], MUTED, "lm")
    T(d, (x1, r1), f"{fps:.0f} FPS", F["cont"], BLANCO, "rm")
    T(d, (x0, r2), "Análisis IA", F["small"], MUTED, "lm")
    T(d, (x1, r2), f"{ms:.0f} ms", F["cont"], BLANCO, "rm")
    return y_fin