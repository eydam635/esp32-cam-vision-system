"""alerta_intruso.py - Detecta personas y rostros; si el rostro está cubierto
(mascarilla, barbijo, pasamontañas...) o no se ve, activa alarma visual y sonora.
Requiere esp32_comun.py en la misma carpeta."""

import argparse
import os
import time
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image, ImageDraw

from esp32_comun import (
    ACENTO, AMBAR, BLANCO, CARD, MUTED, ROJO, VERDE,
    Audio, HiloAnalisis, Ventana, agregar_args_comunes, ajustar, alto_fuente, barra,
    cargar_yolo, dibujar_banner, dibujar_caja, dibujar_card, dibujar_chip,
    dibujar_footer, dibujar_header, dibujar_stats, envolver, iniciar_camara, mezclar,
    T, video_en_canvas,
)

WIN = "Sistema de Seguridad - ESP32-CAM"


# ============================================================
#  ANÁLISIS DE LA ZONA DE LA BOCA
# ============================================================
def analizar_cubierto(ycc, x, y, w, h, tol, min_piel):
    """Compara el color (croma) de la zona boca/mentón con el de la frente.
    Devuelve (cubierto, proporción_de_piel_en_la_boca)."""

    def parche(fx1, fy1, fx2, fy2):
        a, b = max(0, y + int(h * fy1)), y + int(h * fy2)
        c, e = max(0, x + int(w * fx1)), x + int(w * fx2)
        p = ycc[a:b, c:e]
        return p.reshape(-1, 3) if p.size else None

    ref = parche(0.25, 0.08, 0.75, 0.28)   # frente
    baja = parche(0.25, 0.62, 0.75, 0.92)  # debajo de la nariz / boca / mentón
    if ref is None or baja is None or len(ref) < 20 or len(baja) < 20:
        return None, None
    cr0, cb0 = float(np.median(ref[:, 1])), float(np.median(ref[:, 2]))
    if np.hypot(cr0 - 128, cb0 - 128) < 8:  # la referencia no parece piel (gorro, luz rara)
        return None, None
    dist = np.hypot(baja[:, 1].astype(np.float32) - cr0, baja[:, 2].astype(np.float32) - cb0)
    ratio = float(np.mean(dist < tol))
    return ratio < min_piel, ratio


class Analizador(HiloAnalisis):
    def __init__(self, cam, model, device, imgsz, conf, tol, min_piel):
        super().__init__(cam)
        self.model, self.device, self.imgsz, self.conf = model, device, imgsz, conf
        self.tol, self.min_piel = tol, min_piel
        base = cv2.data.haarcascades
        self.frontal = cv2.CascadeClassifier(base + "haarcascade_frontalface_alt2.xml")
        self.perfil = cv2.CascadeClassifier(base + "haarcascade_profileface.xml")
        if self.frontal.empty() or self.perfil.empty():
            raise RuntimeError("No se pudieron cargar los clasificadores Haar de OpenCV")

    def _rostros(self, frame):
        gris = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        h, w = gris.shape
        minimo = max(28, int(min(h, w) * 0.10))
        cajas = self.frontal.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5,
                                              minSize=(minimo, minimo))
        rostros = []
        if len(cajas):
            ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            for (x, y, fw, fh) in cajas.tolist():
                cubierto, ratio = (None, None)
                if fw >= 48:
                    cubierto, ratio = analizar_cubierto(ycc, x, y, fw, fh, self.tol, self.min_piel)
                rostros.append({"box": (x, y, x + fw, y + fh), "tipo": "frontal",
                                "cubierto": cubierto, "ratio": ratio})
        else:  # cabeza de lado: probar perfil (y su espejo)
            for espejo in (False, True):
                img = cv2.flip(gris, 1) if espejo else gris
                perf = self.perfil.detectMultiScale(img, 1.1, 4, minSize=(minimo, minimo))
                for (x, y, fw, fh) in (perf.tolist() if len(perf) else []):
                    if espejo:
                        x = w - (x + fw)
                    rostros.append({"box": (x, y, x + fw, y + fh), "tipo": "perfil",
                                    "cubierto": None, "ratio": None})
        return rostros

    def procesar(self, frame):
        h, w = frame.shape[:2]
        r = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, device=self.device,
                               classes=[0], verbose=False)[0]  # clase 0 = persona
        personas = []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy().astype(int).tolist()
            confs = r.boxes.conf.cpu().numpy().tolist()
            for (x1, y1, x2, y2), p in zip(xyxy, confs):
                if (x2 - x1) * (y2 - y1) >= 0.03 * w * h:  # ignorar personas diminutas
                    personas.append({"box": (int(x1), int(y1), int(x2), int(y2)),
                                     "conf": float(p), "con_rostro": False})
        rostros = self._rostros(frame)
        for p in personas:
            x1, y1, x2, y2 = p["box"]
            limite_y = y1 + 0.65 * (y2 - y1)
            for rs in rostros:
                fx1, fy1, fx2, fy2 = rs["box"]
                cx, cy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
                if x1 <= cx <= x2 and y1 <= cy <= limite_y:
                    p["con_rostro"] = True
                    break
        return SimpleNamespace(personas=personas, rostros=rostros)


# ============================================================
#  LÓGICA DE ALARMA (con tiempos para evitar falsas alarmas)
# ============================================================
class Vigilante:
    def __init__(self, t_cubierto, t_sin_rostro, t_limpio=1.5, gracia=0.8):
        self.t_cub = max(t_cubierto, 0.1)
        self.t_sin = t_sin_rostro
        self.t_limpio, self.gracia = t_limpio, gracia
        self.desde = {"cub": None, "sin": None}
        self.visto = {"cub": 0.0, "sin": 0.0}
        self.alarma = False
        self.motivo = ""
        self.t_motivo = 0.0
        self.estado = "espera"
        self.progreso = 0.0
        self.n_alertas = 0
        self.ultima = ""
        self.forzado_hasta = 0.0

    def forzar(self, ahora, seg=4.0):
        self.forzado_hasta = ahora + seg

    def _cond(self, nombre, valor, ahora):
        if valor:
            self.visto[nombre] = ahora
            if self.desde[nombre] is None:
                self.desde[nombre] = ahora
        elif self.desde[nombre] is not None and ahora - self.visto[nombre] > self.gracia:
            self.desde[nombre] = None
        return self.desde[nombre]

    def actualizar(self, res, ahora):
        cubierto = any(r["cubierto"] for r in res.rostros)
        sin_rostro = self.t_sin > 0 and any(not p["con_rostro"] for p in res.personas)
        hay_gente = bool(res.personas or res.rostros)
        t_cub = self._cond("cub", cubierto, ahora)
        t_sin = self._cond("sin", sin_rostro, ahora)

        motivo, progreso = None, 0.0
        if t_cub is not None:
            progreso = max(progreso, (ahora - t_cub) / self.t_cub)
            if ahora - t_cub >= self.t_cub:
                motivo = "ROSTRO CUBIERTO"
        if t_sin is not None and self.t_sin > 0:
            progreso = max(progreso, (ahora - t_sin) / self.t_sin)
            if motivo is None and ahora - t_sin >= self.t_sin:
                motivo = "ROSTRO NO VISIBLE"
        if ahora < self.forzado_hasta:
            motivo = "PRUEBA DE ALARMA"
        self.progreso = min(progreso, 1.0)

        ev = None
        if motivo:
            self.motivo, self.t_motivo = motivo, ahora
            if not self.alarma:
                self.alarma = True
                self.n_alertas += 1
                self.ultima = time.strftime("%H:%M:%S")
                ev = "inicio"
        elif self.alarma and ahora - self.t_motivo > self.t_limpio:
            self.alarma = False
            ev = "fin"

        if self.alarma:
            self.estado = "alerta"
        elif self.progreso > 0:
            self.estado = "verificando"
        elif hay_gente:
            self.estado = "ok"
        else:
            self.estado = "espera"
        return ev


# ============================================================
#  PANEL
# ============================================================
def dibujar_panel_alerta(d, L, F, vig, res, fps, ms):
    k = L.k
    ip = dibujar_card(d, L)
    x0, x1 = L.px + ip, L.px + L.pw - ip
    ancho = x1 - x0
    y = L.py + ip
    dibujar_stats(d, L, F, fps, ms)

    T(d, (x0, y), "ESTADO", F["panel_t"], MUTED)
    y += int(24 * k)

    estados = {
        "espera": ("EN ESPERA", MUTED, "Sin personas frente a la cámara"),
        "ok": ("ACCESO PERMITIDO", VERDE, "Rostro visible y despejado"),
        "verificando": ("VERIFICANDO…", AMBAR, "Analizando el rostro"),
        "alerta": ("ALERTA", ROJO, vig.motivo),
    }
    titulo, col, detalle = estados[vig.estado]
    alto, pad = int(96 * k), int(12 * k)
    d.rounded_rectangle((x0, y, x1, y + alto), radius=int(10 * k),
                        fill=mezclar(col, CARD, 0.82), outline=col, width=2)
    f = ajustar(d, titulo, True, int(26 * k), ancho - 2 * pad)
    T(d, (x0 + pad, y + int(12 * k)), titulo, f, col)
    yy = y + int(12 * k) + int(alto_fuente(f) * 1.25)
    for linea in envolver(d, detalle, F["small"], ancho - 2 * pad)[:2]:
        T(d, (x0 + pad, yy), linea, F["small"], BLANCO)
        yy += int(17 * k)
    if vig.estado == "verificando":
        barra(d, x0 + pad, y + alto - int(16 * k), ancho - 2 * pad, int(6 * k), vig.progreso, col)
    y += alto + int(20 * k)

    visibles = sum(1 for r in res.rostros if not r["cubierto"])
    cubiertos = sum(1 for r in res.rostros if r["cubierto"])
    filas = [
        ("Personas", len(res.personas), BLANCO),
        ("Rostros visibles", visibles, BLANCO),
        ("Rostros cubiertos", cubiertos, ROJO if cubiertos else BLANCO),
        ("Alertas", vig.n_alertas, ROJO if vig.n_alertas else BLANCO),
    ]
    for etiqueta, valor, color in filas:
        T(d, (x0, y), etiqueta, F["item"], MUTED)
        T(d, (x1, y), str(valor), F["cont"], color, "ra")
        y += int(30 * k)
    T(d, (x0, y), "Última alerta", F["item"], MUTED)
    T(d, (x1, y), vig.ultima or "—", F["cont"], BLANCO, "ra")


# ============================================================
#  MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Alerta por rostro cubierto con ESP32-CAM")
    agregar_args_comunes(parser, imgsz=320, conf=0.4)
    parser.add_argument("--t-cubierto", type=float, default=1.0,
                        help="Segundos con el rostro cubierto antes de activar la alarma")
    parser.add_argument("--t-sin-rostro", type=float, default=2.5,
                        help="Segundos con una persona sin rostro visible antes de la alarma (0 = desactivar)")
    parser.add_argument("--tol-piel", type=float, default=12.0,
                        help="Tolerancia de color de piel (sube si hay falsas alarmas, baja si no detecta mascarillas)")
    parser.add_argument("--min-piel", type=float, default=0.45,
                        help="Proporción mínima de piel en la zona de la boca (0 a 1)")
    parser.add_argument("--frase", default="Alerta. Intruso detectado.", help="Frase que dice la alarma")
    parser.add_argument("--guardar-evidencia", action="store_true",
                        help="Guardar una foto en la carpeta 'intrusos' cuando se active la alarma")
    parser.add_argument("--debug", action="store_true", help="Mostrar el valor de piel junto al rostro")
    args = parser.parse_args()

    audio = Audio(voz=not args.sin_voz, activo=not args.silencio)
    model, device = cargar_yolo(args.model, args.imgsz)
    cam, frame, last_id = iniciar_camara(args.url, args.framesize, args.quality)
    analizador = Analizador(cam, model, device, args.imgsz, args.conf, args.tol_piel, args.min_piel)
    analizador.start()

    vig = Vigilante(args.t_cubierto, args.t_sin_rostro)
    ventana = Ventana(WIN, args.tam, args.ancho)
    print("Listo. T probar alarma | M sonido | +/- tamaño | F pantalla completa | Q salir")

    fps, n, t_fps = 0.0, 0, time.time()
    guardar = False
    vacio = SimpleNamespace(personas=[], rostros=[])

    try:
        while True:
            frame, last_id = cam.wait_new(last_id, timeout=0.5)
            if frame is None:
                continue
            ahora = time.time()
            n += 1
            if ahora - t_fps >= 1.0:
                fps, n, t_fps = n / (ahora - t_fps), 0, ahora

            L, F = ventana.preparar(frame)
            res, ms = analizador.obtener()
            res = res or vacio
            ev = vig.actualizar(res, ahora)
            if ev == "inicio":
                audio.sirena(True, args.frase)
                guardar = args.guardar_evidencia
            elif ev == "fin":
                audio.sirena(False)

            alerta = vig.alarma
            parpadeo = int(ahora * 3) % 2 == 0
            canvas = video_en_canvas(frame, L)
            if alerta and parpadeo:
                roi = canvas[L.vy : L.vy + L.vh, L.vx : L.vx + L.vw]
                rojo = np.empty_like(roi)
                rojo[:] = ROJO
                roi[:] = cv2.addWeighted(roi, 0.78, rojo, 0.22, 0)

            sx, sy = L.vw / frame.shape[1], L.vh / frame.shape[0]
            grosor = max(2, int(2 * L.k))
            chips = []

            for p in res.personas:
                x1, y1, x2, y2 = p["box"]
                X1, Y1 = L.vx + int(x1 * sx), L.vy + int(y1 * sy)
                X2, Y2 = L.vx + int(x2 * sx), L.vy + int(y2 * sy)
                if p["con_rostro"]:
                    cv2.rectangle(canvas, (X1, Y1), (X2, Y2), (110, 118, 140), 1, cv2.LINE_AA)
                else:
                    col = ROJO if alerta else AMBAR
                    cv2.rectangle(canvas, (X1, Y1), (X2, Y2), col, grosor, cv2.LINE_AA)
                    chips.append((X1, Y1, col, "SIN ROSTRO"))

            for rs in res.rostros:
                x1, y1, x2, y2 = rs["box"]
                X1, Y1 = L.vx + int(x1 * sx), L.vy + int(y1 * sy)
                X2, Y2 = L.vx + int(x2 * sx), L.vy + int(y2 * sy)
                if rs["cubierto"]:
                    col, txt = ROJO, "ROSTRO CUBIERTO"
                else:
                    col, txt = VERDE, "ROSTRO VISIBLE" if rs["tipo"] == "frontal" else "ROSTRO (PERFIL)"
                dibujar_caja(canvas, X1, Y1, X2, Y2, col, grosor)
                chips.append((X1, Y1, col, txt))
                if args.debug and rs["ratio"] is not None:
                    cv2.putText(canvas, f"piel boca: {rs['ratio']:.2f}", (X1, Y2 + int(16 * L.k)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * L.k, col, 1, cv2.LINE_AA)

            if alerta and parpadeo:
                cv2.rectangle(canvas, (L.vx, L.vy), (L.vx + L.vw - 1, L.vy + L.vh - 1),
                              ROJO, max(3, int(8 * L.k)))

            img = Image.fromarray(canvas)
            d = ImageDraw.Draw(img)
            for X1, Y1, col, label in chips:
                dibujar_chip(d, L, F, X1, Y1, col, label, fg=BLANCO if col == ROJO else (14, 16, 24))
            dibujar_header(d, L, F, "Sistema de Seguridad",
                           "ESP32-CAM  ·  Detección de rostro cubierto",
                           acento=ROJO if alerta else ACENTO)
            dibujar_panel_alerta(d, L, F, vig, res, fps, ms)
            if alerta:
                dibujar_banner(d, L, f"ALERTA  ·  {vig.motivo}", ROJO, L.vy + int(12 * L.k),
                               size=30, fg=BLANCO)
            dibujar_footer(d, L, F, "T probar alarma   ·   M sonido   ·   +/- tamaño   ·   "
                                    "F pantalla completa   ·   Q salir")
            ventana.dibujar_toast(d)
            bgr = ventana.mostrar(img)

            if guardar:
                os.makedirs("intrusos", exist_ok=True)
                cv2.imwrite(os.path.join("intrusos", time.strftime("alerta_%Y%m%d_%H%M%S.png")), bgr)
                guardar = False

            tecla = ventana.tecla()
            if tecla == "q":
                break
            elif tecla == "t":
                vig.forzar(ahora)
                ventana.toast("Probando alarma...", 2.0)
            elif tecla == "m":
                audio.activo = not audio.activo
                ventana.toast("Sonido activado" if audio.activo else "Sonido silenciado")
    except KeyboardInterrupt:
        print("Saliendo...")

    audio.sirena(False)
    analizador.detenido = True
    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()