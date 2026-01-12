import time
import struct
import serial

# Protocole
SYNC1 = 0xAA
SYNC2 = 0x55

# Format du paquet Arduino (little-endian):
# uint32_t seq, float ax,ay,az, float gx,gy,gz, float mx,my,mz, float tempC, uint16_t crc
PACKET_FMT = "<I10fH"
PACKET_SIZE = struct.calcsize(PACKET_FMT)  # 4 + 40 + 2 = 46 bytes


def crc16_ccitt(data: bytes) -> int:
    """CRC-16-CCITT identique à l'Arduino"""
    crc = 0xFFFF
    for b in data:
        crc ^= (b << 8) & 0xFFFF
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


class ImuReader:
    def __init__(self, port: str = "/dev/ttyS0", baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.buf = bytearray()
    
    def open(self):
        self.ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=0.1
        )
        self.ser.reset_input_buffer()
        time.sleep(0.1)
    
    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
    
    def read(self, timeout: float = 1.0):
        """Lit une mesure IMU. Retourne dict ou None si timeout."""
        t0 = time.time()
        
        while time.time() - t0 < timeout:
            # Lire données disponibles
            if self.ser.in_waiting > 0:
                self.buf.extend(self.ser.read(self.ser.in_waiting))
            
            # Chercher sync
            while True:
                idx = self.buf.find(bytes([SYNC1, SYNC2]))
                if idx < 0:
                    if len(self.buf) > 1:
                        self.buf = self.buf[-1:]
                    break
                
                if idx > 0:
                    del self.buf[:idx]
                
                # Paquet complet ?
                if len(self.buf) < 2 + PACKET_SIZE:
                    break
                
                payload = bytes(self.buf[2:2 + PACKET_SIZE])
                del self.buf[:2 + PACKET_SIZE]
                
                # Vérifier CRC
                rx_crc = struct.unpack_from("<H", payload, -2)[0]
                calc_crc = crc16_ccitt(payload[:-2])
                
                if rx_crc != calc_crc:
                    continue
                
                # Décoder
                seq, ax, ay, az, gx, gy, gz, mx, my, mz, tempC, _ = struct.unpack(PACKET_FMT, payload)
                
                return {
                    "seq": seq,
                    "ax": ax, "ay": ay, "az": az,
                    "gx": gx, "gy": gy, "gz": gz,
                    "mx": mx, "my": my, "mz": mz,
                    "tempC": tempC
                }
            
            time.sleep(0.001)
        
        return None
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, *args):
        self.close()