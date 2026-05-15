# Garbage Classification - By: spratt - Wed May 6 2026

import sensor
import time
import ml

sensor.reset()  # Reset and initialize the sensor.
sensor.set_pixformat(sensor.RGB565)  # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)  # Set frame size to QVGA (320x240)
sensor.set_windowing((112, 72, 96, 96))  # Set 96 x 96 window.
sensor.skip_frames(time=2000)  # Let the camera adjust.

# Load model
model = ml.Model("/rom/custom_objects_int8.tflite", load_to_fb=True)
print(model)

garbage_labels = [line.rstrip('\n') for line in open("/rom/custom_objects_labels.txt")]
print(garbage_labels)

clock = time.clock()
garbage_type = ""
confidence = 0.0

while True:
    clock.tick()
    img = sensor.snapshot()

    output = model.predict([img])

    cls = output.index(max(output))
    confidence = max(output[0][0])

    if confidence > 0.5:
        garbage_type = garbage_labels[cls]

    print("FPS:", clock.fps(), "Garbage:", garbage_type, "Conf:", confidence)
