# This work is licensed under the MIT license.
# Copyright (c) 2013-2024 OpenMV LLC. All rights reserved.
# https://github.com/openmv/openmv/blob/master/LICENSE
#
# TensorFlow Lite Object Detection Example
#
# This example uses the built-in FOMO model to detect faces.
# Low-light improvements applied — see comments marked [LOW-LIGHT].

import sensor
import time
import ml
from ml.postprocessing.edgeimpulse import Fomo
import math

sensor.reset()                         # Reset and initialize the sensor.
sensor.set_pixformat(sensor.RGB565)    # Set pixel format to RGB565.
sensor.set_framesize(sensor.QVGA)      # Set frame size to QVGA (320x240).
sensor.set_windowing((240, 240))       # Set 240x240 window.

# [LOW-LIGHT] Slow the frame rate so the sensor integrates (collects) light
# for longer per frame.  16 fps gives ~2x the integration time of 30 fps.
# Lower values (e.g., 10) help more in very dark scenes but reduce smoothness.
sensor.set_framerate(16)

# [LOW-LIGHT] Keep auto-exposure ON so the sensor adapts as lighting changes.
# The second argument (exposure_us) can be set to a fixed value if you prefer
# manual control — e.g., sensor.set_auto_exposure(False, exposure_us=33000).
sensor.set_auto_exposure(True)

# [LOW-LIGHT] Keep auto-gain ON.  The sensor will push gain higher in dim
# light to compensate.  This increases noise, but the preprocessing below
# helps reduce its effect on the model.
sensor.set_auto_gain(True)

# [LOW-LIGHT] Boost sensor-level contrast and brightness slightly.
# Values range from -3 (less) to +3 (more).  Adjust to taste.
sensor.set_contrast(1)
sensor.set_brightness(1)

sensor.skip_frames(time=2000)          # Let the camera settle.

# [LOW-LIGHT] Lower the FOMO confidence threshold slightly.
# In poor lighting the model's raw scores are naturally lower, so reducing
# the threshold from 0.4 to 0.3 recovers some detections.  Watch for false
# positives and raise it back if needed.
model = ml.Model("/rom/fomo_face_detection.tflite", postprocess=Fomo(threshold=0.3))
print(model)

colors = [
    (255, 0,   0),
    (0,   255, 0),
    (255, 255, 0),
    (0,   0,   255),
    (255, 0,   255),
    (0,   255, 255),
    (255, 255, 255),
]

clock = time.clock()
while True:
    clock.tick()
    img = sensor.snapshot()

    # ------------------------------------------------------------------ #
    # LOW-LIGHT IMAGE PREPROCESSING                                        #
    # These steps run on the CPU before the image is passed to the model.  #
    # Apply them in order — each builds on the last.                       #
    # ------------------------------------------------------------------ #

    # [LOW-LIGHT] Step 1 — Adaptive histogram equalization (CLAHE).
    # Redistributes pixel intensities locally so dark regions gain contrast
    # without over-exposing bright ones.  This is the single most impactful
    # change for low-light face detection.
    # adaptive=True enables CLAHE; clip_limit clamps amplification to reduce
    # noise.  Lower clip_limit (e.g., 2) = less noise, less contrast boost.
    img.histeq(adaptive=True, clip_limit=3)

    # [LOW-LIGHT] Step 2 — Gamma correction.
    # gamma < 1.0 brightens shadow areas (dark pixels gain more than bright
    # pixels).  0.8 is a gentle lift; try 0.6-0.7 for very dark scenes.
    img.gamma_corr(gamma=0.8)

    # [LOW-LIGHT] Step 3 — Median noise filter.
    # High sensor gain amplifies pixel noise.  A median filter with size=1
    # (3x3 kernel) smooths salt-and-pepper noise while preserving edges
    # better than a mean/blur filter.
    # NOTE: this adds a small CPU cost (~2-3 ms). Remove it if FPS matters
    # more than noise reduction.
    img.median(1)

    # ------------------------------------------------------------------ #
    # Run inference and draw results                                       #
    # ------------------------------------------------------------------ #
    for i, detection_list in enumerate(model.predict([img])):
        if i == 0:
            continue  # skip background class
        if len(detection_list) == 0:
            continue  # no detections for this class

        print("********** %s **********" % model.labels[i])
        print("Current processor time (in seconds): %i" % time.time())
        for (x, y, w, h), score in detection_list:
            center_x = math.floor(x + (w / 2))
            center_y = math.floor(y + (h / 2))
            print(f"x {center_x}\ty {center_y}\tscore {score}")
            img.draw_circle((center_x, center_y, 12), color=colors[i])

    # print(clock.fps(), "fps", end="\n")
