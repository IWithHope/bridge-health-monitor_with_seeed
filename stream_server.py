import asyncio
import numpy as np
import sys
from flask import Flask, render_template, Response
from threading import Thread

# TFLite
from tensorflow.lite.python.interpreter import Interpreter
from bleak import BleakClient, BleakScanner

DEVICE_NAME = "BridgeNode"
UART_TX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

# --- Load TFLite model ---
MODEL_PATH = "bridge_model.tflite"
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
in_det = interpreter.get_input_details()[0]
out_det = interpreter.get_output_details()[0]

in_shape = list(in_det["shape"])
in_dtype = in_det["dtype"]

# Reshape function
def build_reshape(sample_window):
    def flat(a): return a.reshape(-1).astype(in_dtype)

    if len(in_shape) == 3 and in_shape[0] == 1:
        N, F = in_shape[1], in_shape[2]
        def f(win):
            x = np.zeros((1, N, F), dtype=in_dtype)
            F_use = min(F, 2)
            x[0, :, :F_use] = win[:, :F_use]
            return x
        return f, N, F
    elif len(in_shape) == 2 and in_shape[0] == 1:
        NF = in_shape[1]
        def f(win):
            x = np.zeros((1, NF), dtype=in_dtype)
            x[0, :min(NF, win.size)] = flat(win)[:min(NF, win.size)]
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
        raise RuntimeError(f"Unexpected input shape: {in_shape}")

reshape_input, WINDOW_N, FEATURES = build_reshape(in_shape)

LABELS = ["Normal", "Damage"]
EXPECT_UNITS = "g"

# Sliding window buffer
win = np.zeros((WINDOW_N, 2), dtype=np.float32)
win_idx = 0
filled = False

# Latest classification result (for HTML)
latest_result = {"status": "Waiting...", "prob": {"Normal": 0, "Damage": 0}}

rx_buf = bytearray()

def line_parser(data: bytes):
    global rx_buf
    rx_buf.extend(data)
    out = []
    while True:
        try:
            i = rx_buf.index(10)
        except ValueError:
            break
        line = rx_buf[:i].decode(errors='ignore').strip()
        del rx_buf[:i+1]
        if line:
            out.append(line)
    return out

def classify_and_update():
    global latest_result
    x = win.copy()
    x_in = reshape_input(x)
    interpreter.set_tensor(in_det["index"], x_in.astype(in_dtype))
    interpreter.invoke()
    y = interpreter.get_tensor(out_det["index"]).squeeze()
    y = np.array(y, dtype=np.float32)
    if y.ndim == 0:
        y = np.array([1.0 - float(y), float(y)], dtype=np.float32)
    if y.size == 1:
        y = np.array([1.0 - float(y[0]), float(y[0])], dtype=np.float32)
    pred_idx = int(np.argmax(y))
    probs = {LABELS[i]: float(y[i]) for i in range(min(len(LABELS), y.size))}
    latest_result = {"status": LABELS[pred_idx], "prob": probs}

async def ble_loop():
    global win_idx, filled
    dev = await BleakScanner.find_device_by_filter(
        lambda d, ad: (d.name == DEVICE_NAME) or (ad and ad.local_name == DEVICE_NAME),
        timeout=15.0
    )
    if not dev:
        print("Device not found")
        sys.exit(1)
    async with BleakClient(dev) as client:
        async def handle_notify(_, data: bytearray):
            global win_idx, filled
            for line in line_parser(data):
                try:
                    y_str, z_str = line.split(",")
                    y = float(y_str)
                    z = float(z_str)
                except Exception:
                    continue
                win[win_idx, 0] = y
                win[win_idx, 1] = z
                win_idx += 1
                if win_idx >= WINDOW_N:
                    win_idx = 0
                    filled = True
                if filled:
                    classify_and_update()
        await client.start_notify(UART_TX_CHAR_UUID, handle_notify)
        while True:
            await asyncio.sleep(1.0)

def start_ble_loop():
    asyncio.run(ble_loop())

# --- Flask Server ---
app = Flask(__name__)

@app.route("/")
def index():
    return """
    <html>
    <head>
        <title>Bridge Status</title>
        <style>
            table {border-collapse: collapse; width: 50%; font-size: 24px;}
            td, th {border: 1px solid black; text-align: center; padding: 10px;}
            .Normal {background-color: #a0ffa0;}
            .Damage {background-color: #ff8080;}
        </style>
    </head>
    <body>
        <h1>Bridge Real-Time Status</h1>
        <table>
            <tr><th>Status</th><th>Normal</th><th>Damage</th></tr>
            <tr id="dataRow">
                <td>Waiting...</td><td>0.0</td><td>0.0</td>
            </tr>
        </table>
        <script>
            var es = new EventSource("/stream");
            es.onmessage = function(event){
                var data = JSON.parse(event.data);
                var row = document.getElementById("dataRow");
                row.innerHTML = "<td class='" + data.status + "'>" + data.status + "</td>" +
                                "<td>" + data.prob.Normal.toFixed(3) + "</td>" +
                                "<td>" + data.prob.Damage.toFixed(3) + "</td>";
            }
        </script>
    </body>
    </html>
    """

@app.route("/stream")
def stream():
    def event_stream():
        import time
        while True:
            time.sleep(0.1)
            yield f"data: {latest_result}\n\n"
    return Response(event_stream(), mimetype="text/event-stream")

if __name__ == "__main__":
    # Start BLE loop in separate thread
    t = Thread(target=start_ble_loop, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5000)
