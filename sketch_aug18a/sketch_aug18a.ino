#include <Wire.h>
#include <LSM6DS3.h>          // SparkFun LSM6DS3
#include <bluefruit.h>        // Adafruit Bluefruit nRF52 (BLE)

// ---------- IMU ----------
LSM6DS3 imu(I2C_MODE, 0x6A);   // XIAO Sense default

// ---------- BLE UART ----------
BLEUart bleuart;               // Nordic UART Service

// ---------- Settings ----------
const char* DEVICE_NAME = "BridgeNode";  // shows in scanner
const uint16_t SEND_PERIOD_MS = 20;      // ~50 Hz
const float LPF_ALPHA = 0.15f;           // 0..1; smaller = smoother

// low-pass state
float y_lp = 0.0f, z_lp = 0.0f;

void startAdv() {
  Bluefruit.Advertising.stop();

  Bluefruit.Advertising.addFlags(BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE);
  Bluefruit.Advertising.addTxPower();
  Bluefruit.Advertising.addService(bleuart);
  Bluefruit.ScanResponse.addName();

  Bluefruit.Advertising.restartOnDisconnect(true);
  Bluefruit.Advertising.setInterval(32, 244);   // 20ms to ~152.5ms
  Bluefruit.Advertising.setFastTimeout(30);     // fast mode for 30s
  Bluefruit.Advertising.start(0);               // 0 = keep advertising
}

void setup() {
  // Serial for debug (optional)
  Serial.begin(115200);
  while (!Serial) { delay(5); }

  // ---- BLE init ----
  Bluefruit.begin();
  Bluefruit.setTxPower(4);           // dBm
  Bluefruit.setName(DEVICE_NAME);
  bleuart.begin();                   // start Nordic UART (NUS)
  startAdv();

  // ---- IMU init ----
  // (Optional) adjust settings before begin
  imu.settings.accelEnabled    = 1;
  imu.settings.accelRange      = 2;    // ±2g (try 4/8 for less sensitivity)
  imu.settings.accelSampleRate = 104;  // 104 Hz
  imu.settings.accelBandWidth  = 50;   // low-pass ~50 Hz
  if (imu.begin() != 0) {
    Serial.println("IMU init FAILED");
  } else {
    Serial.println("IMU ready, advertising BLE UART…");
  }
}

void loop() {
  // Read Y,Z in g
  float ay = imu.readFloatAccelY();
  float az = imu.readFloatAccelZ();

  // Low-pass filter
  static bool first = true;
  if (first) { y_lp = ay; z_lp = az; first = false; }
  y_lp = (1.0f - LPF_ALPHA) * y_lp + LPF_ALPHA * ay;
  z_lp = (1.0f - LPF_ALPHA) * z_lp + LPF_ALPHA * az;

  // Send as CSV over BLE UART (Nordic UART)
  // ex: "0.612,0.987\n"
  char line[48];
  int n = snprintf(line, sizeof(line), "%.6f,%.6f\n", y_lp, z_lp);
  if (n > 0) bleuart.write( (uint8_t*)line, n );

  delay(SEND_PERIOD_MS);
}
