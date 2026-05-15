# Untitled - By: spratt - Sat Apr 25 2026

import sensor, image, time, os, machine  # pyb - legacy

# configure camera (ML-safe)
sensor.reset()
sensor.set_pixformat(sensor.GRAYSCALE)
# sensor.set_framesize(sensor.B64X64)
sensor.set_framesize(sensor.QVGA)

# Auto exposure/gain changes image statistics frame-to-frame
# That injects label-independent variance
# Small models (64×64!) are very sensitive to this
sensor.set_auto_gain(False)
sensor.set_auto_whitebal(False)   # harmless in GRAYSCALE, still disable
sensor.set_auto_exposure(False)

# fix exposure/gain explicitly (advanced, but ideal):
# sensor.set_exposure_us(5000)
sensor.set_auto_exposure(False, exposure_us=100000)
# sensor.set_gain_db(0)
sensor.set_auto_gain(False, gain_db=15)

sensor.skip_frames(time=2000)

# ----------------------------
# Dataset configuration
# ----------------------------
LABELS = ["LEFT", "RIGHT", "NONE"]
label_index = 0
samples_per_label = 50  # change as needed

# BASE_PATH = "/sd"
BASE_PATH = '/flash'
btn = machine.Pin('SW', machine.Pin.IN, machine.Pin.PULL_UP)

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
    while not btn.value():
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

    # Button is active-low
    if not btn.value():
        t0 = time.ticks_ms()

        # Button still pressed?
        while not btn.value():
            time.sleep_ms(10)

        print(btn.value())
        dt = time.ticks_diff(time.ticks_ms(), t0)
        print(dt)

        # Long press → switch label
        if dt > 1500:
            next_label()
            time.sleep_ms(300)
            continue

        # Short press → capture
        img = sensor.snapshot()
        img = img.copy(roi=(0, 0, img.width(), img.height()), x_scale=64/img.width(), y_scale=64/img.height())
        filename = "%s/%03d.pgm" % (path, counter)
        img.save(filename)

        print("Saved:", filename)

        time.sleep_ms(300)  # debounce
