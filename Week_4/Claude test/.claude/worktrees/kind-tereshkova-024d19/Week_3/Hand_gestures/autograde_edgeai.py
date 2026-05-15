
"""
Auto-grading / validation script for EdgeAI OpenMV Gesture Assignment
UCSC Extension

Usage:
  python autograde_edgeai.py path/to/student_notebook.ipynb

This script checks:
- Presence of gesture_int8.tflite
- INT8 quantization
- Model size constraint
- Basic inference sanity
"""

import sys
import os
import numpy as np
import tensorflow as tf

MAX_MODEL_SIZE_BYTES = 30000  # 30 KB


def grade_model(tflite_path):
    score = 0
    feedback = []

    if not os.path.exists(tflite_path):
        feedback.append("❌ gesture_int8.tflite not found")
        return score, feedback

    score += 25
    size = os.path.getsize(tflite_path)

    if size <= MAX_MODEL_SIZE_BYTES:
        score += 25
        feedback.append(f"✔ Model size OK ({size} bytes)")
    else:
        feedback.append(f"❌ Model too large ({size} bytes)")

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    if input_details['dtype'] == np.int8:
        score += 25
        feedback.append("✔ INT8 input confirmed")
    else:
        feedback.append("❌ Model not INT8")

    # Sanity inference
    dummy = np.zeros(input_details['shape'], dtype=np.int8)
    interpreter.set_tensor(input_details['index'], dummy)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details['index'])

    if output.shape[-1] == 3:
        score += 25
        feedback.append("✔ Output shape correct (3 classes)")
    else:
        feedback.append("❌ Output shape incorrect")

    return score, feedback


if __name__ == "__main__":
    tflite_path = "gesture_int8.tflite"
    score, feedback = grade_model(tflite_path)

    print("Auto-grade score:", score, "/ 100")
    print("Feedback:")
    for f in feedback:
        print(" -", f)
