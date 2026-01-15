#include <Wire.h>
#include "ICM_20948.h"
#include <string.h>

ICM_20948_I2C imu;

const ICM_20948_ACCEL_CONFIG_FS_SEL_e ACC_RANGE = gpm8;    // options: gpm2, gpm4, gpm8, gpm16
const ICM_20948_GYRO_CONFIG_1_FS_SEL_e GYR_RANGE = dps500; // options: dps250, dps500, dps1000, dps2000

const uint16_t SEND_HZ = 100;
const uint32_t SEND_PERIOD_US = 1000000UL / SEND_HZ;

const uint32_t UART_BAUD = 115200;

static inline float mg_to_ms2(float mg) { return mg * 0.00980665f; }
static inline float dps_to_rads(float dps) { return dps * 0.01745329252f; }

struct ImuPacket;
static void sendPacket(const ImuPacket &pkt);

struct __attribute__((packed)) ImuPacket {
  uint32_t seq;
  float ax, ay, az;
  float gx, gy, gz;
  float mx, my, mz;
  float tempC;
  uint16_t crc;
};

static uint16_t crc16_ccitt(const uint8_t *data, size_t len) {
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < len; i++) {
    crc ^= (uint16_t)data[i] << 8;
    for (uint8_t b = 0; b < 8; b++) {
      crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
    }
  }
  return crc;
}

static const uint8_t SYNC1 = 0xAA;
static const uint8_t SYNC2 = 0x55;

static void sendPacket(const ImuPacket &pkt) {
  Serial1.write(SYNC1);
  Serial1.write(SYNC2);
  Serial1.write((const uint8_t *)&pkt, sizeof(pkt));
}

float ax, ay, az;
float gx, gy, gz;
float mx, my, mz;
float tempC;

uint32_t seqCounter = 0;
uint32_t lastSendUs = 0;

void setup() {
  Serial.begin(115200);
  uint32_t t0 = millis();
  while (!Serial && (millis() - t0 < 2000)) {
    // attend max 2 secondes
  }

  Wire.begin();
  Wire.setClock(400000);

  while (true) {
    imu.begin(Wire, 1);
    if (imu.status == ICM_20948_Stat_Ok) break;
    delay(200);
  }

  imu.swReset();
  delay(250);
  imu.sleep(false);
  imu.lowPower(false);

  imu.setSampleMode((ICM_20948_Internal_Acc | ICM_20948_Internal_Gyr),
                    ICM_20948_Sample_Mode_Continuous);

  ICM_20948_fss_t fss;
  fss.a = ACC_RANGE;
  fss.g = GYR_RANGE;
  imu.setFullScale((ICM_20948_Internal_Acc | ICM_20948_Internal_Gyr), fss);

  ICM_20948_dlpcfg_t dlp;
  dlp.a = acc_d50bw4_n68bw8;
  dlp.g = gyr_d51bw2_n73bw3;
  imu.setDLPFcfg((ICM_20948_Internal_Acc | ICM_20948_Internal_Gyr), dlp);

  imu.enableDLPF(ICM_20948_Internal_Acc, true);
  imu.enableDLPF(ICM_20948_Internal_Gyr, true);

  imu.startupMagnetometer();

  Serial.println("initialisation faites, config appliquee");

  Serial1.begin(UART_BAUD);
}

void loop() {
  if (imu.dataReady()) {
    imu.getAGMT();

    // Lecture des valeurs brutes directement
    ax = mg_to_ms2(imu.accX());
    ay = mg_to_ms2(imu.accY());
    az = mg_to_ms2(imu.accZ());

    gx = dps_to_rads(imu.gyrX());
    gy = dps_to_rads(imu.gyrY());
    gz = dps_to_rads(imu.gyrZ());

    mx = imu.magX();
    my = imu.magY();
    mz = imu.magZ();

    tempC = imu.temp();
  }

  uint32_t nowUs = micros();
  if ((uint32_t)(nowUs - lastSendUs) >= SEND_PERIOD_US) {
    lastSendUs += SEND_PERIOD_US;

    ImuPacket pkt;
    pkt.seq = ++seqCounter;
    pkt.ax = ax; pkt.ay = ay; pkt.az = az;
    pkt.gx = gx; pkt.gy = gy; pkt.gz = gz;
    pkt.mx = mx; pkt.my = my; pkt.mz = mz;
    pkt.tempC = tempC;
    pkt.crc = 0;
    pkt.crc = crc16_ccitt((const uint8_t *)&pkt, sizeof(ImuPacket) - sizeof(pkt.crc));

    sendPacket(pkt);
  }
}