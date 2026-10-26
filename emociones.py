import cv2
import time
from deepface import DeepFace

# ====== CONFIGURACIÓN ======
IP = "COLOCS LS IP AQUI "
URL_STREAM = f"http://{IP}:81/stream"   # stream del CameraWebServer
ANALIZAR_CADA = 5                        # analizar 1 de cada N frames (menos = más carga)
# ===========================

# Traducción a las 4 emociones que quieres
TRADUCCION = {
    "happy": "FELIZ",
    "angry": "ENOJADO",
    "sad": "TRISTE",
    "neutral": "NORMAL",
}
COLORES = {
    "FELIZ": (0, 220, 0),
    "ENOJADO": (0, 0, 255),
    "TRISTE": (255, 120, 0),
    "NORMAL": (200, 200, 200),
}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def conectar():
    cap = cv2.VideoCapture(URL_STREAM)
    if not cap.isOpened():
        raise RuntimeError(
            f"No se pudo abrir {URL_STREAM}. Revisa la IP y que el stream esté activo."
        )
    return cap


def clasificar(rostro_bgr):
    """Devuelve (emocion_en_español, confianza) usando solo las 4 emociones."""
    res = DeepFace.analyze(
        rostro_bgr,
        actions=["emotion"],
        enforce_detection=False,
        detector_backend="skip",   # ya recortamos la cara nosotros
        silent=True,
    )
    if isinstance(res, list):
        res = res[0]
    scores = res["emotion"]

    # Nos quedamos solo con las 4 emociones y elegimos la mayor
    filtrado = {k: scores[k] for k in TRADUCCION}
    mejor = max(filtrado, key=filtrado.get)
    total = sum(filtrado.values()) or 1
    return TRADUCCION[mejor], filtrado[mejor] / total * 100


def main():
    cap = conectar()
    contador = 0
    ultimo = {}  # guarda el último resultado por posición de rostro
    fallos = 0

    print("Presiona 'q' para salir.")
    while True:
        ok, frame = cap.read()
        if not ok:
            fallos += 1
            if fallos > 20:
                print("Se perdió la conexión, reconectando...")
                cap.release()
                time.sleep(2)
                try:
                    cap = conectar()
                    fallos = 0
                except RuntimeError as e:
                    print(e)
            continue
        fallos = 0
        contador += 1

        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostros = face_cascade.detectMultiScale(
            gris, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
        )

        for i, (x, y, w, h) in enumerate(rostros):
            if contador % ANALIZAR_CADA == 0 or i not in ultimo:
                rostro = frame[y:y + h, x:x + w]
                try:
                    ultimo[i] = clasificar(rostro)
                except Exception as e:
                    print("Error analizando:", e)
                    continue

            emocion, conf = ultimo[i]
            color = COLORES[emocion]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame, f"{emocion} {conf:.0f}%", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
            )

        cv2.imshow("ESP32-CAM - Emociones", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()