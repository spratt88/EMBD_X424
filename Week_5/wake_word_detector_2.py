import audio
import ml # Changed from 'tf' to 'ml'
import sensor
import time
# import pyb
from ulab import numpy as np # Correctly import numpy from ulab
# from ulab import math as um # Import ulab's math module
# from ulab import random as urandom # Import ulab's random module
import math as um
import random as urandom

# --- Configuration (Must match training parameters) ---
SAMPLE_RATE = 16000
CLIP_DURATION_MS = 1000
N_MFCC = 40 # Number of MFCCs (FEATURE_BIN_COUNT)
TIME_STEPS = 51 # From X_train.shape[1]
WINDOW_SIZE_MS = 30 # For MFCC calculation
WINDOW_STRIDE_MS = 20 # For MFCC calculation

WAKE_WORD_THRESHOLD = 0.85 # Confidence threshold for detection
# Index of 'hey_jarvis' from encoder.classes_ ['hey_jarvis' 'negative' 'noise']
WAKE_WORD_LABEL_INDEX = 0

# --- Initialize OpenMV hardware ---
sensor.reset()
sensor.set_framesize(sensor.QVGA) # Can be any valid framesize, not used for audio
sensor.set_pixformat(sensor.RGB565)

# Initialize audio streaming.
# audio.init() configures the audio peripheral. For start_streaming, it might only need the buffer size.
# We specify the buffer size to ensure we get 1 second of audio.
audio_buffer_size = int(SAMPLE_RATE * (CLIP_DURATION_MS / 1000))
# Initialize audio with the sample rate and channels first
audio.init(channels=1, frequency=SAMPLE_RATE)
audio.start_streaming(audio_buffer_size) # Only pass buffer size to start_streaming

print("Initializing ML model...")
# Load the quantized TFLite model using ml.Model
# Make sure 'wake_word_svdf_int8.tflite' is on the OpenMV's SD card or internal flash
model = ml.Model("/rom/wake_word_svdf_int8.tflite") # Changed from tf.load to ml.Model

# Check input shape of the model (ml.Model might not have input_shape() directly accessible)
# You might need to rely on documentation for getting input details.
# For now, we'll keep the print but note it might not work as before.
# print("Model input shape:", model.input_shape())
print("Model loaded successfully.")

# These need to be pre-calculated from your training data and hardcoded
# or loaded into the OpenMV device.
# Example (replace with actual values):
MFCC_MEAN = np.zeros(N_MFCC, dtype=np.float)
MFCC_STD = np.ones(N_MFCC, dtype=np.float)

# --- Helper for Mel-scale conversion ---
def hz_to_mel(hz):
    return 2595 * um.log10(1 + hz / 700)

def mel_to_hz(mel):
    return 700 * (10**(mel / 2595) - 1)

# --- Placeholder for DCT (ulab might not have it directly, manual is needed) ---
def dct(x, type=2, n=None, axis=-1, norm=None):
    # A simple DCT-II for demonstration. Actual ulab implementation may differ
    # or require more complex handling. For a real system, you might precompute
    # cosine factors.
    if n is None:
        n = len(x)
    if type != 2 or axis != -1 or norm is not None:
        print("WARNING: Only basic DCT-II is conceptually shown here.")

    c = np.zeros(n, dtype=np.float32)
    for k in range(n):
        sum_val = 0.0
        for i in range(len(x)): # This loop was missing
            sum_val += x[i] * um.cos(um.pi * k * (2 * i + 1) / (2 * n))

        # Use um.sqrt for ulab's math sqrt function
        c[k] = sum_val * um.sqrt(2 / n)

    if norm == 'ortho': # Not fully implemented, just for conceptual completeness
        c[0] /= um.sqrt(2)
    return c

    if norm == 'ortho': # Not fully implemented, just for conceptual completeness
        c[0] /= um.sqrt(2)
    return c


def compute_features_openmv(audio_samples):
    """
    Conceptual MFCC feature extraction for OpenMV.
    This is a simplified outline and requires detailed implementation.
    """
    # 1. Convert to float32 and normalize to [-1, 1]
    float_samples = audio_samples.astype(np.float32) / 32768.0

    # Parameters for framing
    frame_length = int(SAMPLE_RATE * WINDOW_SIZE_MS / 1000)
    hop_length = int(SAMPLE_RATE * WINDOW_STRIDE_MS / 1000)

    # Ensure frame_length is a power of 2 for FFT
    n_fft = 1
    while n_fft < frame_length:
        n_fft *= 2

    # Number of Mel filters (often N_MFCC or slightly more for better resolution)
    n_mels = N_MFCC + 4 # Using slightly more filters than MFCCs for better resolution

    # Create Mel filter banks (this is a complex step, often pre-calculated)
    # This part is highly simplified and illustrative.
    # You would need to create a matrix of triangular filters.
    # Example: create `n_mels` filters covering the frequency range.
    # For real use, implement `librosa.filters.mel` logic.
    # This is *very* basic and for concept only:
    mel_filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    # Populate mel_filters with actual triangular filter shapes here
    # ... (detailed implementation of mel filter bank generation)

    mfccs_list = []
    for i in range(0, len(float_samples) - frame_length + 1, hop_length):
        frame = float_samples[i : i + frame_length]

        # Apply Hamming window (conceptual, ulab might have a window function)
        windowed_frame = frame # * np.hamming(frame_length) if available

        # Perform FFT
        # audio.FFT is usually for 'image' data. For raw audio use audio.fft() function if available
        # Assuming audio.fft() exists and returns magnitude spectrum
        # e.g., spectrum = audio.fft(windowed_frame, n_fft)
        # For OpenMV, you might need to use `ulab.fft` or `audio.FFT` for `image.Image` objects.

        # For now, let's use a dummy spectrum calculation as a placeholder
        # Replace with actual FFT magnitude calculation
        spectrum = urandom.rand(n_fft // 2 + 1).astype(np.float32) # Use urandom.rand

        # Apply Mel filters to the power spectrum
        mel_energies = np.dot(mel_filters, spectrum**2) # Using dot product for conceptual filter application
        mel_energies = um.log(mel_energies + 1e-6) # Use um.log

        # Apply DCT
        current_mfccs = dct(mel_energies, n=N_MFCC)
        mfccs_list.append(current_mfccs)

    # Stack all MFCC frames
    if len(mfccs_list) == 0:
        return np.zeros((TIME_STEPS, N_MFCC), dtype=np.float32)

    mfccs = np.array(mfccs_list)

    # Pad or truncate mfccs to match TIME_STEPS
    if mfccs.shape[0] < TIME_STEPS:
        padding = TIME_STEPS - mfccs.shape[0]
        mfccs = np.pad(mfccs, ((0, padding), (0,0)), mode='constant')
    elif mfccs.shape[0] > TIME_STEPS:
        mfccs = mfccs[:TIME_STEPS, :]

    # Normalize (using pre-calculated mean and std from training)
    mfccs = (mfccs - MFCC_MEAN) / (MFCC_STD + 1e-6)

    print("WARNING: compute_features_openmv is conceptual and requires full implementation!")
    return mfccs.astype(np.float32)

# --- Feature Extraction Function (Placeholder) ---
def compute_features(audio_samples):
    """
    Placeholder function for MFCC feature extraction on OpenMV.
    This needs to be implemented using OpenMV's audio processing capabilities.

    Args:
        audio_samples (np.array): 1-second audio samples (int16 or float32).

    Returns:
        np.array: MFCC features of shape (TIME_STEPS, N_MFCC), float32.
                  Normalized to match training data.
    """
    # 1. Convert int16 audio_samples to float32 (normalized to -1.0 to 1.0)
    #    e.g., float_samples = audio_samples.astype(np.float32) / 32768.0

    # 2. Implement MFCC calculation:
    #    - Apply short-time Fourier transform (STFT) using FFT.
    #    - Apply Mel filter banks to spectrogram.
    #    - Apply Discrete Cosine Transform (DCT) to log-mel energies.
    #    - Ensure correct frame length (WINDOW_SIZE_MS) and hop length (WINDOW_STRIDE_MS).

    # 3. Reshape and normalize MFCCs:
    #    - Reshape to (TIME_STEPS, N_MFCC).
    #    - Normalize (mean=0, std=1) using the mean/std from your training data.
    #      You will need to hardcode or pass these values from your training notebook.

    # Example: dummy features for demonstration (replace with actual MFCCs)
    # These dimensions must match your model's input shape (TIME_STEPS, N_MFCC)
    #rand_num = np.random.Generator(None)
    #dummy_features = rand_num.random(TIME_STEPS, N_MFCC).astype(np.float32)
    dummy_features = None

    print("WARNING: Using dummy features. Implement actual MFCC extraction!")
    return dummy_features


# --- Main Loop ---
print("Starting audio capture and inference loop...")
clock = time.clock()

while(True):
    clock.tick()

    # Capture 1 second of audio. audio.get_buffer() returns the latest filled buffer.
    # Convert the raw bytes buffer from audio.get_buffer() to a ulab numpy array.
    audio_data_raw = audio.get_buffer()
    current_audio_samples = np.frombuffer(audio_data_raw, dtype=np.int16)

    # Process audio and extract features
    # Ensure compute_features returns a np.array of float32 matching (1, TIME_STEPS, N_MFCC)
    # for the model's input_shape (1, 51, 40)
    features = compute_features(current_audio_samples)

    # Perform inference using model.predict(). It typically returns a list of output tensors.
    # For this model, we expect one output tensor containing the class probabilities.
    results = model.predict(features.reshape(1, TIME_STEPS, N_MFCC)) # Changed to model.predict()
    predictions = results[0] # Get the first (and likely only) output tensor

    # Get confidence for the wake word class
    wake_word_confidence = predictions[WAKE_WORD_LABEL_INDEX]

    print("FPS:", clock.fps())
    print(f"Predictions: {predictions}, Wake Word Confidence ({WAKE_WORD_LABEL_INDEX}): {wake_word_confidence:.3f}")

    if wake_word_confidence > WAKE_WORD_THRESHOLD:
        print("--- WAKE WORD DETECTED! ---")
        # Add your action here, e.g., turn on an LED, send a message, etc.
        #pyb.LED(1).on() # Example: Turn on Red LED
    else:
        #pyb.LED(1).off() # Example: Turn off Red LED
        pass

    # Add a short delay to avoid overwhelming the system
    time.sleep_ms(100)
