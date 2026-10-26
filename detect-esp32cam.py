import argparse
import threading
import time
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
import torch
from ultralytics import YOLO


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
                                del buf[:-1]  # conserva 1 byte por si SOI quedó partido
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
        """Espera un frame distinto a last_id. Devuelve (frame, id) o (None, last_id)."""
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


class Detector(threading.Thread):
    """Corre YOLO en un hilo aparte, siempre sobre el frame más reciente."""

    def __init__(self, cam, model, imgsz, conf, device):
        super().__init__(daemon=True)
        self.cam, self.model = cam, model
        self.imgsz, self.conf, self.device = imgsz, conf, device
        self.lock = threading.Lock()
        self.detections = []
        self.infer_ms = 0.0
        self.stopped = False

    def get(self):
        with self.lock:
            return self.detections, self.infer_ms

    def run(self):
        last_id = 0
        names = self.model.names
        while not self.stopped:
            frame, last_id = self.cam.wait_new(last_id, timeout=1.0)
            if frame is None:
                continue
            try:
                t0 = time.time()
                r = self.model.predict(
                    frame,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    device=self.device,
                    verbose=False,
                )[0]
                dets = []
                if r.boxes is not None and len(r.boxes):
                    xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
                    cls = r.boxes.cls.cpu().numpy().astype(int)
                    confs = r.boxes.conf.cpu().numpy()
                    for (x1, y1, x2, y2), c, p in zip(
                        xyxy.tolist(), cls.tolist(), confs.tolist()
                    ):
                        dets.append(
                            (int(x1), int(y1), int(x2), int(y2), int(c),
                             f"{names[int(c)]} {p:.2f}")
                        )
                ms = (time.time() - t0) * 1000
                with self.lock:
                    self.detections, self.infer_ms = dets, ms
            except Exception as e:
                print(f"Error en detección: {e}")
                time.sleep(0.5)


def color_clase(c):
    c = int(c)
    return (
        int((c * 47) % 200 + 55),
        int((c * 91) % 200 + 55),
        int((c * 153) % 200 + 55),
    )


def configurar_camara(url, framesize, quality):
    """Cambia resolución/calidad de la ESP32-CAM (endpoint /control del puerto 80)."""
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


def main():
    parser = argparse.ArgumentParser(
        description="Detección de objetos en tiempo real con ESP32-CAM y YOLOv8"
    )
    parser.add_argument("--url", default="IP AQUI",
                        help="URL del stream (ej: IP AQUI)")
    parser.add_argument("--model", default="yolov8n.pt", help="Modelo YOLO")
    parser.add_argument("--imgsz", type=int, default=320,
                        help="Tamaño de inferencia (320 rápido, 416/640 más preciso)")
    parser.add_argument("--conf", type=float, default=0.4, help="Confianza mínima")
    parser.add_argument("--framesize", type=int, default=None,
                        help="Resolución de la cámara: 5=QVGA(320x240), 8=VGA(640x480)")
    parser.add_argument("--quality", type=int, default=None,
                        help="Calidad JPEG de la cámara (10-63, más alto = más liviano)")
    parser.add_argument("--no-show", dest="show", action="store_false",
                        help="No mostrar ventana de video")
    parser.set_defaults(show=True)
    args = parser.parse_args()

    configurar_camara(args.url, args.framesize, args.quality)

    use_cuda = torch.cuda.is_available()
    device = 0 if use_cuda else "cpu"
    print(f"Cargando modelo {args.model} en {'GPU' if use_cuda else 'CPU'}...")
    model = YOLO(args.model)
    # Calentamiento: evita el tirón del primer frame
    model.predict(np.zeros((args.imgsz, args.imgsz, 3), np.uint8),
                  imgsz=args.imgsz, device=device, verbose=False)

    print(f"Conectando al stream: {args.url}")
    cam = ESP32CamStream(args.url).start()

    frame, last_id = None, 0
    while frame is None:
        frame, last_id = cam.wait_new(last_id, timeout=1.0)
    print("Listo. Presiona 'q' en la ventana (o Ctrl+C) para salir.")

    det = Detector(cam, model, args.imgsz, args.conf, device)
    det.start()

    fps, n, t_fps = 0.0, 0, time.time()
    last_print = 0.0

    try:
        while True:
            frame, last_id = cam.wait_new(last_id, timeout=0.5)
            if frame is None:
                continue

            n += 1
            now = time.time()
            if now - t_fps >= 1.0:
                fps, n, t_fps = n / (now - t_fps), 0, now

            dets, ms = det.get()

            if args.show:
                vis = frame.copy()  # copia: el hilo de detección usa el original
                for x1, y1, x2, y2, c, label in dets:
                    col = color_clase(c)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
                    cv2.putText(vis, label, (x1, max(y1 - 6, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
                cv2.putText(vis, f"Video: {fps:.0f} FPS | YOLO: {ms:.0f} ms",
                            (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.imshow("ESP32-CAM YOLOv8 Realtime", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            elif now - last_print > 1.0:
                print(f"{fps:.0f} FPS | {ms:.0f} ms | " +
                      (", ".join(d[5] for d in dets) or "sin detecciones"))
                last_print = now
    except KeyboardInterrupt:
        print("Saliendo...")

    det.stopped = True
    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()