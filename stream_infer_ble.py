# save as stream_infer_ble.py
import asyncio
import numpy as np
import sys

# ---- TFLite load ----
from tensorflow.lite import Interpreter

from bleak import BleakClient, BleakScanner

DEVICE_NAME = "BridgeNode"

# Nordic UART Service (NUS) UUIDs (Bluefruit/Nordic UART)
UART_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
UART_TX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"  # Notify from peripheral -> PC

# --- Load model ---
MODEL_PATH = "bridge_model.tflite"
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
in_det = interpreter.get_input_details()[0]
out_det = interpreter.get_output_details()[0]

# Figure out expected input shape
in_shape = list(in_det["shape"])
in_dtype = in_det["dtype"]

# Configure window and reshape function based on input
def build_reshape(sample_window):
    def flat(a): return a.reshape(-1).astype(in_dtype)

    if len(in_shape) == 3 and in_shape[0] == 1 and in_shape[2] in (1, 2, 3):
        N = in_shape[1]; F = in_shape[2]
        def f(win):
            x = np.zeros((1, N, F), dtype=in_dtype)
            F_use = min(F, 2)  # we send Y,Z
            x[0, :, :F_use] = win[:, :F_use]
            return x
        return f, N, F
    elif len(in_shape) == 2 and in_shape[0] == 1:
        NF = in_shape[1]
        def f(win):
            x = np.zeros((1, NF), dtype=in_dtype)
            flat_in = flat(win)
            x[0, :min(NF, flat_in.size)] = flat_in[:min(NF, flat_in.size)]
            return x
        N = NF // 2
        return f, N, 2
    elif len(in_shape) == 2:
        N, F = in_shape
        def f(win):
            x = np.zeros((N, F), dtype=in_dtype)
            F_use = min(F, 2)
            x[:, :F_use] = win[:, :F_use]
            return x
        return f, N, F
    else:
        raise RuntimeError(f"Unrecognized input shape: {in_shape}")

reshape_input, WINDOW_N, FEATURES = build_reshape(in_shape)

# Labels
LABELS = ["Normal", "Damage"]

# Units from device
EXPECT_UNITS = "g"

# Sliding window buffer
win = np.zeros((WINDOW_N, 2), dtype=np.float32)
win_idx = 0
filled = False

# Optional low-pass
PC_LPF_ALPHA = 0.0
y_lp, z_lp = 0.0, 0.0
first = True

# Notification buffer
rx_buf = bytearray()

def line_parser(data: bytes):
    global rx_buf
    rx_buf.extend(data)
    out = []
    while True:
        try:
            i = rx_buf.index(10)  # '\n'
        except ValueError:
            break
        line = rx_buf[:i].decode(errors='ignore').strip()
        del rx_buf[:i+1]
        if line:
            out.append(line)
    return out

def classify_and_print():
    x = win.copy()
    if EXPECT_UNITS == "ms2":
        x *= 9.80665
    x_in = reshape_input(x)
    interpreter.set_tensor(in_det["index"], x_in.astype(in_dtype))
    interpreter.invoke()
    y = interpreter.get_tensor(out_det["index"]).squeeze()

    # Softmax safety
    y = np.array(y, dtype=np.float32)
    if y.ndim == 0:
        y = np.array([1.0 - float(y), float(y)], dtype=np.float32)
    if y.size == 1:
        y = np.array([1.0 - float(y[0]), float(y[0])], dtype=np.float32)

    pred_idx = int(np.argmax(y))
    probs = {LABELS[i]: float(y[i]) for i in range(min(len(LABELS), y.size))}
    status = LABELS[pred_idx]
    print(f"{status}  |  {probs}")

async def main():
    print("Scanning for device:", DEVICE_NAME)
    dev = await BleakScanner.find_device_by_filter(
        lambda d, ad: (d.name == DEVICE_NAME) or (ad and ad.local_name == DEVICE_NAME),
        timeout=15.0
    )
    if not dev:
        print("Device not found. Make sure it’s advertising.")
        sys.exit(1)

    print("Connecting to:", dev)
    async with BleakClient(dev) as client:
        print("Connected. Subscribing to UART notifications…")

        async def handle_notify(_, data: bytearray):
            global win_idx, filled, first, y_lp, z_lp

            for line in line_parser(data):
                try:
                    y_str, z_str = line.split(",")
                    y = float(y_str)
                    z = float(z_str)
                except Exception:
                    continue

                if PC_LPF_ALPHA > 0.0:
                    if first:
                        y_lp, z_lp, first = y, z, False
                    y_lp = (1.0 - PC_LPF_ALPHA) * y_lp + PC_LPF_ALPHA * y
                    z_lp = (1.0 - PC_LPF_ALPHA) * z_lp + PC_LPF_ALPHA * z
                    y_use, z_use = y_lp, z_lp
                else:
                    y_use, z_use = y, z

                win[win_idx, 0] = y_use
                win[win_idx, 1] = z_use
                win_idx += 1
                if win_idx >= WINDOW_N:
                    win_idx = 0
                    filled = True

                if filled:
                    classify_and_print()

        await client.start_notify(UART_TX_CHAR_UUID, handle_notify)
        print("Receiving… (Ctrl+C to stop)")
        while True:
            await asyncio.sleep(1.0)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
