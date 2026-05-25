# svdf_wakeword - By: spratt - Fri May 15 2026

import audio
import tf

model = tf.load("wake_word_svdf.tflite")

while True:
    samples = capture_audio()

    features = compute_features(samples)

    prediction = model.predict(features)

    if prediction["hey_jarvis"] > 0.85:
        print("Wake word detected")
