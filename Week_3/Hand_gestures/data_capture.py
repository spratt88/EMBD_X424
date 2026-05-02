# Untitled - By: spratt - Sat Apr 25 2026
# Updated for OpenMV Cam AE3 (from H7 Plus)

import sensor, image, time, os
from machine import Pin

# configure camera (ML-safe)
sensor.reset()
sensor.set_pixformat(sensor.GRAYSCALE)
sensor.set_framesize(sensor.B64X64)

# Auto exposure/gain changes image statistics frame-to-frame
# That injects label-independent variance
# Small models (64×64!) are very sensitive to this
sensor.set_auto_gain(False)
sensor.set_auto_exposure(False)
try:
    sensor.set_auto_whitebal(False)   # harmless in GRAYSCALE, still disable
    sensor.set_exposure_us(5000)
    sensor.set_gain_db(0)
except Exception:
    pass

sensor.skip_frames(time=2000)

# ----------------------------
# Dataset configuration
# ----------------------------
LABELS = ["LEFT", "RIGHT", "NONE"]
label_index = 0
samples_per_label = 50  # change as needed

BASE_PATH = "/sd"
# AE3 uses machine.Pin (active-low: pressed = 0)
btn = Pin('BTN', Pin.IN, Pin.PULL_UP)

def btn_pressed():
    return not btn.value()

# Create directories
for lbl in LABELS:
    path = "%s/%s" % (BASE_PATH, lbl)
    if lbl not in os.listdir(BASE_PATH):
        os.mkdir(path)

print("=== DATASET CAPTURE MODE ===")
print("Press button to capture image")
print("Hold button (>1s) to switch label")

# ----------------------------
# Helper functions
# ----------------------------
def wait_for_release():
    while btn_pressed():
        time.sleep_ms(10)


def next_label():
    global label_index
    label_index = (label_index + 1) % len(LABELS)
    print("\n>>> SWITCH TO:", LABELS[label_index])

# ----------------------------
# Capture loop
# ----------------------------
while True:
    label = LABELS[label_index]
    path = "%s/%s" % (BASE_PATH, label)

    counter = len(os.listdir(path))

    print("\nLabel:", label)
    print("Samples:", counter, "/", samples_per_label)

    # Capture preview (not saved)
    sensor.snapshot()

    if btn_pressed():
        t0 = time.ticks_ms()

        # Button still pressed?
        while btn_pressed():
            time.sleep_ms(10)

        dt = time.ticks_diff(time.ticks_ms(), t0)

        # Long press → switch label
        if dt > 1000:
            next_label()
            time.sleep_ms(300)
            continue

        # Short press → capture
        img = sensor.snapshot()
        filename = "%s/%03d.pgm" % (path, counter)
        img.save(filename)

        print("Saved:", filename)

        time.sleep_ms(300)  # debounce
