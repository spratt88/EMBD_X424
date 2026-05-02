# gesture_classificaton - By: spratt - Mon Apr 27 2026

# main.py – OpenMV deployment reference
# EdgeAI Gesture Recognition (INT8 TFLite)

import sensor, image, time, os, machine, ml  # pyb - legacy

sensor.reset()
sensor.set_pixformat(sensor.GRAYSCALE)
# sensor.set_framesize(sensor.B64X64)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(time=2000)

# net = tf.load("/flash/gesture_int8.tflite", load_to_fb=True)
net = ml.Model("gesture_int8.tflite", load_to_fb=True)
labels = ["NONE", "LEFT", "RIGHT"]

clock = time.clock()

state = "IDLE"

while True:
    clock.tick()
    img = sensor.snapshot()
    img = img.copy(roi=(0, 0, img.width(), img.height()), x_scale=64/img.width(), y_scale=64/img.height())

    # net.predict() replaces net.classify()
    # Returns a flat list of output tensors directly — no wrapper object
    output = net.predict([img])[0].flatten().tolist()

    cls = output.index(max(output))
    confidence = max(output)

    if state == "IDLE" and cls != 0 and confidence > 0.8:
        state = labels[cls]
    elif state in ["LEFT", "RIGHT"] and cls == 0:
        print("Gesture detected:", state)
        state = "IDLE"

    print("FPS:", clock.fps(), "State:", state, "Conf:", confidence)
