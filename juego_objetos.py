"""juego_objetos.py - Juego "Encuentra el objeto" con ESP32-CAM + YOLOv8
Requiere esp32_comun.py en la misma carpeta."""

import argparse
import math
import os
import random
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from esp32_comun import (
    ACENTO, AMBAR, BLANCO, CARD, MUTED, PALETA, ROJO, TRACK, VERDE,
    Audio, HiloAnalisis, Ventana, agregar_args_comunes, ajustar, alto_fuente, barra,
    cargar_yolo, dibujar_banner, dibujar_caja, dibujar_card, dibujar_chip,
    dibujar_footer, dibujar_header, dibujar_stats, iniciar_camara, mezclar, parrafo,
    T, video_en_canvas,
)

WIN = "Encuentra el Objeto - ESP32-CAM"
PARAM2_MONEDA = 38  # sube (30-60) si detecta "monedas" donde no las hay

OBJETOS = {
    # clave (clase COCO): (nombre en pantalla, frase hablada)
    "bird": ("PÁJARO", "un pájaro"),
    "clock": ("RELOJ", "un reloj"),
    "book": ("LIBRO", "un libro"),
    "mouse": ("MOUSE", "un mouse"),
    "remote": ("CONTROL", "un control remoto"),
    "cell phone": ("TELÉFONO", "un teléfono"),
    "sports ball": ("BALÓN", "un balón"),
    "coin": ("MONEDA", "una moneda"),  # no existe en YOLO: se detecta con círculos
    "bottle": ("BOTELLA", "una botella"),
}
CLASES_YOLO = [k for k in OBJETOS if k != "coin"]
COLOR_OBJ = {k: PALETA[i % len(PALETA)] for i, k in enumerate(OBJETOS)}


# ============================================================
#  DETECCIÓN
# ============================================================
def detectar_monedas(frame, dets):
    """Busca círculos pequeños (experimental). Ignora los que caen dentro de otros objetos."""
    h, w = frame.shape[:2]
    lado = min(h, w)
    gris = cv2.medianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 5)
    circulos = cv2.HoughCircles(
        gris, cv2.HOUGH_GRADIENT, dp=1.2, minDist=lado * 0.15,
        param1=100, param2=PARAM2_MONEDA,
        minRadius=int(lado * 0.05), maxRadius=int(lado * 0.20),
    )
    salida = []
    if circulos is None:
        return salida
    for cx, cy, r in np.round(circulos[0]).astype(int).tolist()[:3]:
        if any(x1 <= cx <= x2 and y1 <= cy <= y2 for (x1, y1, x2, y2, _, _) in dets):
            continue
        salida.append((cx - r, cy - r, cx + r, cy + r, "coin", 0.80))
    return salida


class DetectorJuego(HiloAnalisis):
    def __init__(self, cam, model, device, imgsz, conf, ids, usar_moneda):
        super().__init__(cam)
        self.model, self.device, self.imgsz, self.conf = model, device, imgsz, conf
        self.ids = ids
        self.usar_moneda = usar_moneda
        self.buscar_moneda = False
        self.nombres = model.names

    def procesar(self, frame):
        r = self.model.predict(
            frame, imgsz=self.imgsz, conf=self.conf, device=self.device,
            classes=self.ids, verbose=False,
        )[0]
        dets = []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy().astype(int).tolist()
            cls = r.boxes.cls.cpu().numpy().astype(int).tolist()
            confs = r.boxes.conf.cpu().numpy().tolist()
            for (x1, y1, x2, y2), c, p in zip(xyxy, cls, confs):
                dets.append((int(x1), int(y1), int(x2), int(y2),
                             self.nombres[int(c)], float(p)))
        if self.usar_moneda and self.buscar_moneda:
            dets += detectar_monedas(frame, dets)
        return dets


# ============================================================
#  LÓGICA DEL JUEGO
# ============================================================
class Juego:
    PAUSA = 2.2  # segundos mostrando acierto / fallo

    def __init__(self, claves, rondas, tiempo, mantener):
        self.claves = claves
        self.rondas = min(rondas, len(claves))
        self.tiempo = tiempo
        self.mantener = mantener
        self.estado = "inicio"
        self.orden, self.resultados = [], []
        self.puntos = self.aciertos = self.ronda = self.ganado = 0
        self.objetivo = None
        self.t_ini = self.t_estado = self.t_visto = 0.0
        self.t_hold = None
        self.progreso = 0.0

    def restante(self, ahora):
        return max(0.0, self.tiempo - (ahora - self.t_ini))

    def iniciar(self):
        self.orden = random.sample(self.claves, self.rondas)
        self.resultados = []
        self.puntos = self.aciertos = self.ronda = 0
        self._empezar_ronda()

    def _empezar_ronda(self):
        self.objetivo = self.orden[self.ronda]
        self.estado = "jugando"
        self.t_ini = time.time()
        self.t_hold = None
        self.progreso = 0.0

    def saltar(self):
        if self.estado == "jugando":
            self.estado, self.t_estado = "fallo", time.time()
            self.resultados.append("no")
            return ["fallo"]
        return []

    def actualizar(self, dets, ahora):
        ev = []
        if self.estado == "jugando":
            visto = any(d[4] == self.objetivo for d in dets)
            if visto:
                self.t_visto = ahora
                if self.t_hold is None:
                    self.t_hold = ahora
            elif self.t_hold is not None and ahora - self.t_visto > 0.6:
                self.t_hold = None
            self.progreso = 0.0 if self.t_hold is None else min(
                1.0, (ahora - self.t_hold) / self.mantener)

            if self.progreso >= 1.0:
                self.ganado = 100 + int(self.restante(ahora)) * 5
                self.puntos += self.ganado
                self.aciertos += 1
                self.resultados.append("ok")
                self.estado, self.t_estado = "acierto", ahora
                ev.append("acierto")
            elif self.restante(ahora) <= 0:
                self.resultados.append("no")
                self.estado, self.t_estado = "fallo", ahora
                ev.append("fallo")
        elif self.estado in ("acierto", "fallo") and ahora - self.t_estado >= self.PAUSA:
            self.ronda += 1
            if self.ronda >= self.rondas:
                self.estado, self.t_estado = "fin", ahora
                ev.append("fin")
            else:
                self._empezar_ronda()
                ev.append("ronda")
        return ev


class Confeti:
    def __init__(self):
        self.p = []

    def lanzar(self, w, h, n=90):
        for _ in range(n):
            self.p.append([random.uniform(0, w), random.uniform(-h * 0.25, 0),
                           random.uniform(-40, 40), random.uniform(60, 220),
                           random.choice(PALETA), random.randint(3, 6)])

    def dibujar(self, canvas, dt, L):
        vivos = []
        for p in self.p:
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            p[3] += 300 * dt
            if p[1] < L.vh:
                vivos.append(p)
                if 0 <= p[1] and 0 <= p[0] < L.vw:
                    cv2.circle(canvas, (L.vx + int(p[0]), L.vy + int(p[1])),
                               max(2, int(p[5] * L.k)), p[4], -1, cv2.LINE_AA)
        self.p = vivos


# ============================================================
#  DIBUJO DEL JUEGO
# ============================================================
def dibujar_panel_juego(d, L, F, J, ahora, fps, ms):
    k = L.k
    ip = dibujar_card(d, L)
    x0, x1 = L.px + ip, L.px + L.pw - ip
    ancho = x1 - x0
    y = L.py + ip
    y_stats = dibujar_stats(d, L, F, fps, ms)

    if J.estado == "inicio":
        T(d, (x0, y), "JUEGO", F["panel_t"], MUTED)
    else:
        T(d, (x0, y), f"RONDA {min(J.ronda + 1, J.rondas)}/{J.rondas}", F["panel_t"], MUTED)
        T(d, (x1, y), f"{J.puntos} pts", F["cont"], ACENTO, "ra")
    y += int(28 * k)

    if J.estado != "inicio":
        r, sep = max(5, int(7 * k)), int(8 * k)
        cx = x0
        for i in range(J.rondas):
            if i < len(J.resultados):
                col = VERDE if J.resultados[i] == "ok" else ROJO
            else:
                col = TRACK
            d.ellipse((cx, y, cx + 2 * r, y + 2 * r), fill=col)
            if i == J.ronda and J.estado == "jugando":
                d.ellipse((cx - 3, y - 3, cx + 2 * r + 3, y + 2 * r + 3), outline=BLANCO, width=2)
            cx += 2 * r + sep
        y += 2 * r + int(20 * k)

    if J.estado == "jugando":
        nombre = OBJETOS[J.objetivo][0]
        col = COLOR_OBJ[J.objetivo]
        T(d, (x0, y), "BUSCA", F["panel_t"], MUTED)
        y += int(22 * k)
        f = ajustar(d, nombre, True, int(40 * k), ancho)
        T(d, (x0, y), nombre, f, col)
        y += int(alto_fuente(f) * 1.3)
        T(d, (x0, y), "Muéstralo a la cámara", F["small"], MUTED)
        y += int(34 * k)
        rest = J.restante(ahora)
        frac_t = rest / J.tiempo
        colt = VERDE if frac_t > 0.5 else AMBAR if frac_t > 0.25 else ROJO
        T(d, (x0, y), "TIEMPO", F["small"], MUTED)
        T(d, (x1, y), f"{math.ceil(rest)} s", F["cont"], colt, "ra")
        y += int(24 * k)
        barra(d, x0, y, ancho, int(8 * k), frac_t, colt)
        y += int(32 * k)
        if J.progreso > 0:
            T(d, (x0, y), "¡Lo veo! No lo muevas", F["item"], col)
        else:
            T(d, (x0, y), "Buscando…", F["item"], MUTED)
        y += int(28 * k)
        barra(d, x0, y, ancho, int(8 * k), J.progreso, col)

    elif J.estado == "acierto":
        f = ajustar(d, "¡MUY BIEN!", True, int(34 * k), ancho)
        T(d, (x0, y), "¡MUY BIEN!", f, VERDE)
        y += int(alto_fuente(f) * 1.3)
        T(d, (x0, y), f"+{J.ganado} puntos", F["cont"], ACENTO)
        y += int(34 * k)
        T(d, (x0, y), "Lo encontraste:", F["small"], MUTED)
        y += int(20 * k)
        T(d, (x0, y), OBJETOS[J.objetivo][0], ajustar(d, OBJETOS[J.objetivo][0], True, int(30 * k), ancho),
          COLOR_OBJ[J.objetivo])

    elif J.estado == "fallo":
        f = ajustar(d, "¡CASI!", True, int(34 * k), ancho)
        T(d, (x0, y), "¡CASI!", f, AMBAR)
        y += int(alto_fuente(f) * 1.3)
        T(d, (x0, y), "Era:", F["small"], MUTED)
        y += int(20 * k)
        T(d, (x0, y), OBJETOS[J.objetivo][0], ajustar(d, OBJETOS[J.objetivo][0], True, int(30 * k), ancho),
          COLOR_OBJ[J.objetivo])
        y += int(46 * k)
        parrafo(d, "¡Vamos con el siguiente!", F["item"], BLANCO, x0, y, ancho, int(22 * k))

    elif J.estado == "fin":
        f = ajustar(d, "¡FIN DEL JUEGO!", True, int(26 * k), ancho)
        T(d, (x0, y), "¡FIN DEL JUEGO!", f, ACENTO)
        y += int(alto_fuente(f) * 1.4)
        T(d, (x0, y), "PUNTOS", F["small"], MUTED)
        y += int(18 * k)
        T(d, (x0, y), str(J.puntos), F["num"], BLANCO)
        y += int(alto_fuente(F["num"]) * 1.25)
        T(d, (x0, y), f"Aciertos: {J.aciertos} de {J.rondas}", F["item"], BLANCO)
        y += int(30 * k)
        ratio = J.aciertos / max(1, J.rondas)
        msg = "¡Excelente!" if ratio >= 0.8 else "¡Buen trabajo!" if ratio >= 0.5 else "¡Sigue practicando!"
        T(d, (x0, y), msg, F["cont"], VERDE if ratio >= 0.5 else AMBAR)
        y += int(34 * k)
        parrafo(d, "Presiona ESPACIO para jugar otra vez", F["small"], ACENTO, x0, y, ancho, int(18 * k))

    else:  # inicio
        f = ajustar(d, "¡Encuentra el objeto!", True, int(24 * k), ancho)
        T(d, (x0, y), "¡Encuentra el objeto!", f, BLANCO)
        y += int(alto_fuente(f) * 1.5)
        y = parrafo(d, "Te digo un objeto y tienes que mostrárselo a la cámara antes de que se acabe el tiempo.",
                    F["small"], MUTED, x0, y, ancho, int(18 * k))
        y += int(10 * k)
        y = parrafo(d, "Presiona ESPACIO para empezar", F["item"], ACENTO, x0, y, ancho, int(22 * k))
        y += int(14 * k)
        cx, cy, h = x0, y, int(22 * k)
        for key in J.claves:
            txt = OBJETOS[key][0].capitalize()
            w = int(d.textlength(txt, font=F["small"])) + int(18 * k)
            if cx + w > x1:
                cx, cy = x0, cy + h + int(6 * k)
            if cy + h > y_stats - int(6 * k):
                break
            d.rounded_rectangle((cx, cy, cx + w, cy + h), radius=h // 2,
                                fill=mezclar(COLOR_OBJ[key], CARD, 0.65))
            T(d, (cx + w // 2, cy + h // 2), txt, F["small"], BLANCO, "mm")
            cx += w + int(6 * k)


def dibujar_overlays_juego(d, L, F, J):
    centro = L.vy + L.vh // 2
    if J.estado == "jugando":
        nombre = OBJETOS[J.objetivo][0]
        dibujar_banner(d, L, f"BUSCA:  {nombre}", COLOR_OBJ[J.objetivo], L.vy + int(12 * L.k), size=26)
    elif J.estado == "acierto":
        dibujar_banner(d, L, "¡LO ENCONTRASTE!", VERDE, centro - int(30 * L.k), size=44)
    elif J.estado == "fallo":
        dibujar_banner(d, L, f"¡CASI!  ERA: {OBJETOS[J.objetivo][0]}", AMBAR, centro - int(30 * L.k), size=38)
    elif J.estado == "inicio":
        y = centro - int(40 * L.k)
        h = dibujar_banner(d, L, "ENCUENTRA EL OBJETO", ACENTO, y, size=44)
        T(d, (L.vx + L.vw // 2, y + h + int(26 * L.k)), "Presiona ESPACIO para empezar",
          F["item"], BLANCO, "mm")
    elif J.estado == "fin":
        y = centro - int(40 * L.k)
        h = dibujar_banner(d, L, "¡JUEGO TERMINADO!", ACENTO, y, size=44)
        T(d, (L.vx + L.vw // 2, y + h + int(26 * L.k)),
          f"{J.puntos} puntos  ·  {J.aciertos}/{J.rondas} aciertos", F["item"], BLANCO, "mm")


# ============================================================
#  MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Juego "Encuentra el objeto" con ESP32-CAM y YOLOv8')
    agregar_args_comunes(parser, imgsz=416, conf=0.35)
    parser.add_argument("--rondas", type=int, default=5, help="Número de rondas (máx. 9)")
    parser.add_argument("--tiempo", type=int, default=25, help="Segundos por ronda")
    parser.add_argument("--mantener", type=float, default=1.0,
                        help="Segundos que hay que mantener el objeto visible")
    parser.add_argument("--sin-moneda", action="store_true", help="Quitar la moneda del juego")
    args = parser.parse_args()

    audio = Audio(voz=not args.sin_voz, activo=not args.silencio)
    model, device = cargar_yolo(args.model, args.imgsz)
    ids = [i for i, n in model.names.items() if n in CLASES_YOLO] or None
    claves = [k for k in OBJETOS if not (k == "coin" and args.sin_moneda)]

    cam, frame, last_id = iniciar_camara(args.url, args.framesize, args.quality)
    det = DetectorJuego(cam, model, device, args.imgsz, args.conf, ids, not args.sin_moneda)
    det.start()

    juego = Juego(claves, args.rondas, args.tiempo, args.mantener)
    ventana = Ventana(WIN, args.tam, args.ancho)
    confeti = Confeti()
    ventana.preparar(frame)

    def reaccionar(ev):
        if ev == "ronda":
            audio.decir(f"Muéstrame {OBJETOS[juego.objetivo][1]}")
        elif ev == "acierto":
            audio.ok()
            audio.decir("¡Muy bien!")
            confeti.lanzar(ventana.L.vw, ventana.L.vh)
        elif ev == "fallo":
            audio.fallo()
        elif ev == "fin":
            audio.fin()
            audio.decir(f"Juego terminado. Hiciste {juego.puntos} puntos")
            if juego.aciertos:
                confeti.lanzar(ventana.L.vw, ventana.L.vh, 140)

    print("Listo. ESPACIO empezar | N saltar | S foto | M sonido | +/- tamaño | F pantalla completa | Q salir")
    fps, n, t_fps, t_prev = 0.0, 0, time.time(), time.time()

    try:
        while True:
            frame, last_id = cam.wait_new(last_id, timeout=0.5)
            if frame is None:
                continue
            ahora = time.time()
            dt, t_prev = ahora - t_prev, ahora
            n += 1
            if ahora - t_fps >= 1.0:
                fps, n, t_fps = n / (ahora - t_fps), 0, ahora

            L, F = ventana.preparar(frame)
            det.buscar_moneda = juego.estado == "jugando" and juego.objetivo == "coin"
            dets, ms = det.obtener()
            dets = dets or []
            for ev in juego.actualizar(dets, ahora):
                reaccionar(ev)

            canvas = video_en_canvas(frame, L)
            if juego.estado in ("inicio", "fin"):
                roi = canvas[L.vy : L.vy + L.vh, L.vx : L.vx + L.vw]
                roi[:] = (roi * 0.45).astype(np.uint8)

            sx, sy = L.vw / frame.shape[1], L.vh / frame.shape[0]
            grosor = max(2, int(2 * L.k))
            chips = []
            if juego.estado in ("jugando", "acierto", "fallo"):
                for (x1, y1, x2, y2, key, p) in dets:
                    X1, Y1 = L.vx + int(x1 * sx), L.vy + int(y1 * sy)
                    X2, Y2 = L.vx + int(x2 * sx), L.vy + int(y2 * sy)
                    if key == juego.objetivo:
                        col = VERDE if juego.estado == "acierto" else COLOR_OBJ[key]
                        if key == "coin":
                            cv2.circle(canvas, ((X1 + X2) // 2, (Y1 + Y2) // 2),
                                       max(4, (X2 - X1) // 2), col, grosor + 1, cv2.LINE_AA)
                        else:
                            dibujar_caja(canvas, X1, Y1, X2, Y2, col, grosor)
                        chips.append((X1, Y1, col, f"{OBJETOS[key][0].capitalize()} {int(p * 100)}%"))
                    else:
                        cv2.rectangle(canvas, (X1, Y1), (X2, Y2), (110, 118, 140), 1, cv2.LINE_AA)

            confeti.dibujar(canvas, dt, L)

            img = Image.fromarray(canvas)
            d = ImageDraw.Draw(img)
            for X1, Y1, col, label in chips:
                dibujar_chip(d, L, F, X1, Y1, col, label)
            dibujar_header(d, L, F, "Encuentra el Objeto",
                           "ESP32-CAM  ·  Juego de reconocimiento con inteligencia artificial")
            dibujar_panel_juego(d, L, F, juego, ahora, fps, ms)
            dibujar_overlays_juego(d, L, F, juego)
            dibujar_footer(d, L, F, "ESPACIO empezar   ·   N saltar   ·   S foto   ·   M sonido   ·   "
                                    "+/- tamaño   ·   F pantalla completa   ·   Q salir")
            ventana.dibujar_toast(d)
            bgr = ventana.mostrar(img)

            tecla = ventana.tecla()
            if tecla == "q":
                break
            elif tecla == " " and juego.estado in ("inicio", "fin"):
                juego.iniciar()
                reaccionar("ronda")
            elif tecla == "n":
                for ev in juego.saltar():
                    reaccionar(ev)
            elif tecla == "m":
                audio.activo = not audio.activo
                ventana.toast("Sonido activado" if audio.activo else "Sonido silenciado")
            elif tecla == "s":
                os.makedirs("capturas", exist_ok=True)
                ruta = os.path.join("capturas", time.strftime("captura_%Y%m%d_%H%M%S.png"))
                cv2.imwrite(ruta, bgr)
                ventana.toast("Foto guardada en la carpeta capturas", 2.5)
    except KeyboardInterrupt:
        print("Saliendo...")

    det.detenido = True
    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()