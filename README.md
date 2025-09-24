# Bridge Health Detector 🏗️

The **Bridge Health Detector** is a real-time system built with **Seeed Studio hardware** that monitors vibrations from a bridge and classifies its condition as **Normal ✅** or **Damage ⚠️** using an embedded machine learning model.  

This project combines **IoT hardware, embedded ML, and web technologies** to provide a cost-effective and scalable solution for monitoring bridge health.

---

## 🚀 Features
- Built with **Seeed Studio XIAO nRF52840 Sense** (onboard **BMI270 IMU**)  
- Real-time vibration monitoring and classification  
- On-device ML inference (runs offline, no cloud required)  
- Serial communication to a **Python Flask server**  
- Live web dashboard with automatic updates every 2 seconds  
- Color-coded visualization (green = Normal, red = Damage)  

---

## 🛠️ Tech Stack

### Hardware
- **Seeed Studio XIAO nRF52840 Sense**  
- **BMI270 IMU sensor** (accelerometer, gyroscope)  

### Software
- **Edge Impulse** → Data collection & ML model training  
- **Arduino IDE** → Firmware upload to Seeed Studio board  
- **Python (Flask, PySerial)** → Backend server & API  
- **HTML + JavaScript** → Frontend dashboard  

---

## 📂 Project Structure
Bridge-Health-Detector/

-SeeedStudio/ # Firmware for Seeed Studio board
    main.ino # Arduino sketch with Edge Impulse model
- dataset/ # Collected vibration dataset
   data.csv
- stream_server.py # Python Flask server for serial data
- bridge_model.tflite # TensorFlow Lite model (optional for reference)
- model_data.h  # Model exported as C++ header for Arduino
- scaler.pkl # Preprocessing scaler (used for retraining)
- scaler.pkl # Preprocessing scaler (used for retraining)
- stream_infer_ble.py # Python script for BLE-based inference streaming
- README.md # Project documentation (this file)
