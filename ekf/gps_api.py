import pigpio
import time

class GPSApi:
    def __init__(self, rx_pin=24, tx_pin=23, en_pin=17, baudrate=9600):
        """
        Classe GPS utilisant pigpio pour simuler un UART (bit-banging).
        :param rx_pin: GPIO pour la réception (TX du module GPS)
        :param tx_pin: GPIO pour l'émission (RX du module GPS)
        :param en_pin: GPIO pour l'activation du module (EN) 
        :param baudrate: Vitesse (9600 par défaut) [cite: 160]
        """
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("Impossible de se connecter à pigpiod. Lancez 'sudo pigpiod'.")

        self.rx_pin = rx_pin
        self.tx_pin = tx_pin
        self.en_pin = en_pin
        self.baudrate = baudrate

        # Données de sortie
        self.latitude = 0.0
        self.longitude = 0.0
        self.speed_ms = 0.0  # Vitesse en mètres par seconde
        self.is_fixed = False # Indicateur de fix GPS valide [cite: 308, 331]

        # Commandes UBX (Datasheet VK2828U7G5LF)
        self.CMD_10HZ = bytes.fromhex("B5 62 06 08 06 00 64 00 01 00 01 00 7A 12 B5 62 06 08 00 00 30")
        self.ENABLE_RMC = bytes.fromhex("B5 62 06 01 03 00 F0 04 01 FF 18")
        self.ENABLE_GGA = bytes.fromhex("B5 62 06 01 03 00 F0 00 01 FB 10")

        self._setup_pins()

    def _setup_pins(self):
        """Configuration des pins GPIO via pigpio."""
        self.pi.set_mode(self.en_pin, pigpio.OUTPUT)
        self.pi.write(self.en_pin, 1) # Activer le module 
        
        self.pi.set_mode(self.rx_pin, pigpio.INPUT)
        self.pi.set_mode(self.tx_pin, pigpio.OUTPUT)
        
        # Ouvre le port série simulé en lecture
        try:
            self.pi.bb_serial_read_open(self.rx_pin, self.baudrate)
        except:
            self.pi.bb_serial_read_close(self.rx_pin)
            self.pi.bb_serial_read_open(self.rx_pin, self.baudrate)

    def _send_command(self, cmd):
        """Envoie une commande HEX via bit-banging TX."""
        self.pi.wave_clear()
        self.pi.wave_add_serial(self.tx_pin, self.baudrate, cmd)
        wid = self.pi.wave_create()
        if wid >= 0:
            self.pi.wave_send_once(wid)
            while self.pi.wave_tx_busy():
                time.sleep(0.01)
            self.pi.wave_delete(wid)
        time.sleep(0.1)

    def initialize(self):
        """Initialise la configuration 10Hz du module."""
        print("[GPS] Configuration du module à 10Hz...")
        time.sleep(0.5) # Attente démarrage module [cite: 549]
        self._send_command(self.ENABLE_GGA)
        self._send_command(self.ENABLE_RMC)
        self._send_command(self.CMD_10HZ)
        return True

    def _nmea_to_decimal(self, coord_str, hemisphere):
        """Convertit ddmm.mmmmm en degrés décimaux[cite: 509]."""
        if not coord_str or not hemisphere: return 0.0
        try:
            idx = coord_str.find('.')
            deg_len = idx - 2
            deg = float(coord_str[:deg_len])
            minutes = float(coord_str[deg_len:])
            decimal = deg + (minutes / 60.0)
            if hemisphere in ['S', 'W']: decimal = -decimal
            return decimal
        except: return 0.0

    def update(self):
        """Lit les données du port série simulé et parse les trames."""
        (count, data) = self.pi.bb_serial_read(self.rx_pin)
        if count > 0:
            raw_lines = data.decode('ascii', errors='ignore').split('\r\n')
            for line in raw_lines:
                if "RMC" in line: # Trame recommandée pour vitesse et position [cite: 329, 331]
                    parts = line.split(',')
                    if len(parts) >= 8:
                        self.is_fixed = (parts[2] == 'A')
                        if self.is_fixed:
                            self.latitude = self._nmea_to_decimal(parts[3], parts[4])
                            self.longitude = self._nmea_to_decimal(parts[5], parts[6])
                            # Conversion Noeuds -> m/s (1 noeud = 0.514444 m/s) 
                            knots = float(parts[7]) if parts[7] else 0.0
                            self.speed_ms = knots * 0.514444
                elif "GGA" in line:
                    parts = line.split(',')
                    if len(parts) >= 7:
                        # Fix valide si mode différent de 0 [cite: 299]
                        self.is_fixed = parts[6] in ['1', '2', '3']

    def cleanup(self):
        """Arrête le module et libère les ressources."""
        self.pi.bb_serial_read_close(self.rx_pin)
        self.pi.write(self.en_pin, 0)
        self.pi.stop()
        print("[GPS] Ressources libérées.")

if __name__ == "__main__":
    gps = GPSApi(rx_pin=24, tx_pin=23, en_pin=17)
    try:
        gps.initialize()
        while True:
            gps.update()
            if gps.is_fixed:
                print(f"FIX OK | Lat: {gps.latitude:.6f} | Lon: {gps.longitude:.6f} | Vitesse: {gps.speed_ms:.2f} m/s")
            else:
                print("Attente de signal...")
            time.sleep(0.1) # Boucle à 10Hz
    except KeyboardInterrupt:
        gps.cleanup()