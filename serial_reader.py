"""
serial_reader.py
-----------------
Lecture du port série Arduino dans un thread séparé, pour ne jamais bloquer
le rafraîchissement de l'interface Streamlit.

Format de ligne attendu depuis l'Arduino (séparé par des virgules) :
    accX,accY,accZ,temp,humidity

Exemple de sketch Arduino correspondant (MPU6050 + DHT11) :
    Serial.print(accX); Serial.print(",");
    Serial.print(accY); Serial.print(",");
    Serial.print(accZ); Serial.print(",");
    Serial.print(temperature); Serial.print(",");
    Serial.println(humidity);
"""

import threading
import time
from collections import deque
from datetime import datetime

import serial
import serial.tools.list_ports


def list_available_ports():
    """Retourne la liste des ports série détectés sur la machine."""
    ports = serial.tools.list_ports.comports()
    return [p.device for p in ports]


class SerialReader:
    """
    Lit en continu le port série dans un thread de fond et empile les
    échantillons valides dans un buffer borné (deque) thread-safe.
    """

    EXPECTED_FIELDS = ["accX", "accY", "accZ", "temp", "humidity"]

    def __init__(self, port, baudrate=9600, max_samples=5000):
        self.port = port
        self.baudrate = baudrate
        self.max_samples = max_samples

        self._buffer = deque(maxlen=max_samples)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._serial_conn = None

        self.connected = False
        self.last_error = None
        self.total_lines = 0
        self.malformed_lines = 0
        self.last_sample_time = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self):
        if self._thread and self._thread.is_alive():
            return  # already running
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._serial_conn:
            try:
                self._serial_conn.close()
            except Exception:
                pass
        self.connected = False

    # ------------------------------------------------------------------ #
    # Internal loop (runs in background thread)
    # ------------------------------------------------------------------ #
    def _run(self):
        try:
            self._serial_conn = serial.Serial(self.port, self.baudrate, timeout=2)
            time.sleep(2)  # laisser le temps à l'Arduino de redémarrer après ouverture du port
            self.connected = True
            self.last_error = None
        except Exception as e:
            self.last_error = f"Connexion impossible sur {self.port} : {e}"
            self.connected = False
            return

        while not self._stop_event.is_set():
            try:
                raw = self._serial_conn.readline().decode("utf-8", errors="ignore").strip()
                if not raw:
                    continue

                parts = raw.split(",")
                self.total_lines += 1

                if len(parts) != len(self.EXPECTED_FIELDS):
                    self.malformed_lines += 1
                    continue

                try:
                    values = [float(p) for p in parts]
                except ValueError:
                    self.malformed_lines += 1
                    continue

                sample = dict(zip(self.EXPECTED_FIELDS, values))
                sample["timestamp"] = datetime.now()

                with self._lock:
                    self._buffer.append(sample)
                    self.last_sample_time = sample["timestamp"]

            except Exception as e:
                self.last_error = f"Erreur de lecture : {e}"
                time.sleep(0.5)

    # ------------------------------------------------------------------ #
    # Public accessors (thread-safe)
    # ------------------------------------------------------------------ #
    def get_dataframe(self):
        """Retourne une copie snapshot du buffer sous forme de liste de dicts."""
        with self._lock:
            return list(self._buffer)

    def sample_count(self):
        with self._lock:
            return len(self._buffer)

    def clear(self):
        with self._lock:
            self._buffer.clear()
        self.total_lines = 0
        self.malformed_lines = 0
