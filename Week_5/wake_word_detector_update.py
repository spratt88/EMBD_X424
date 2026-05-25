"""
Wake Word Detector - "Hey Jarvis"
OpenMV AE3 MicroPython Application

Hardware : OpenMV AE3
Model    : /rom/wake_word_svdf_int8.tflite  (int8 quantized)

Audio / MFCC parameters — taken directly from the training notebook
--------------------------------------------------------------------
SAMPLE_RATE       = 16000  Hz
CLIP_DURATION_MS  = 1000   ms  → 16 000 samples
WINDOW_SIZE_MS    = 30     ms  → n_fft  = 480  → padded to 512 (power-of-2)
WINDOW_STRIDE_MS  = 20     ms  → hop    = 320  samples
FEATURE_BIN_COUNT = 40         → N_MFCC = 40
TIME_STEPS        = 51         → matches X_train.shape[1]

librosa centre-pads by n_fft//2 = 256 samples on each side before framing,
which is why 16 000 samples with hop=320 yields 51 frames instead of 49.
We replicate that padding here so the feature vectors match training exactly.

Normalisation: per-clip  (mfcc - mean) / (std + 1e-6)  — same as notebook.

Encoded classes (alphabetical order from LabelEncoder):
  index 0 -> hey_jarvis   <- wake word
  index 1 -> negative
  index 2 -> noise
"""

import audio
import math
import ml
import struct
import utime
import micropython
from ulab import numpy as np

# ---------------------------------------------------------------------------
# Configuration  (must match training notebook exactly)
# ---------------------------------------------------------------------------
SAMPLE_RATE       = 16000          # Hz
NUM_CHANNELS      = 1              # Mono
CLIP_SAMPLES      = 16000          # 1 second @ 16 kHz
CONFIDENCE_THRESH = 0.85
MODEL_PATH        = "/rom/wake_word_svdf_int8.tflite"

# MFCC — mirror of training notebook values
WINDOW_SIZE_MS    = 30             # ms
WINDOW_STRIDE_MS  = 20             # ms
N_FFT             = 512            # next power-of-2 >= int(16000*30/1000)=480
HOP_LENGTH        = int(SAMPLE_RATE * WINDOW_STRIDE_MS / 1000)   # 320
NUM_MEL_BINS      = 40
NUM_MFCC_COEFFS   = 40
NUM_TIME_STEPS    = 51             # verified from model.input_shape
PAD_LEN           = N_FFT // 2    # 256 — librosa centre-pad each side

# ---------------------------------------------------------------------------
# Model initialisation
# ---------------------------------------------------------------------------
print("[INIT] Loading model:", MODEL_PATH)
model = ml.Model(MODEL_PATH)
print("[INIT] input_shape      =", model.input_shape)
print("[INIT] input_dtype      =", model.input_dtype)
print("[INIT] input_scale      =", model.input_scale)
print("[INIT] input_zero_point =", model.input_zero_point)

_Q_SCALE = model.input_scale[0]
_Q_ZP    = model.input_zero_point[0]

# ---------------------------------------------------------------------------
# Audio initialisation
# ---------------------------------------------------------------------------
audio.init(channels=NUM_CHANNELS,
           frequency=SAMPLE_RATE,
           gain_db=24,
           highpass=0.9883)
print("[INIT] Microphone ready @ {}Hz {}ch".format(SAMPLE_RATE, NUM_CHANNELS))

# ---------------------------------------------------------------------------
# Pre-computed Hann window  (ulab ndarray, length N_FFT)
# ---------------------------------------------------------------------------
HANN = np.array([0.5 - 0.5 * math.cos(2.0 * math.pi * n / (N_FFT - 1))
                 for n in range(N_FFT)], dtype=np.float)

# ---------------------------------------------------------------------------
# Pre-computed Mel filterbank
# Shape: (NUM_MEL_BINS, N_FFT//2+1)  i.e. (40, 257)
# Matches librosa.filters.mel(sr=16000, n_fft=512, n_mels=40, fmin=0,
#                             fmax=8000, norm=None)
# ---------------------------------------------------------------------------
def _build_mel_filterbank():
    def _hz_mel(f):
        return 2595.0 * math.log10(1.0 + f / 700.0)
    def _mel_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    n_freqs  = N_FFT // 2 + 1
    mel_min  = _hz_mel(0.0)
    mel_max  = _hz_mel(SAMPLE_RATE / 2.0)
    mel_pts  = [mel_min + i * (mel_max - mel_min) / (NUM_MEL_BINS + 1)
                for i in range(NUM_MEL_BINS + 2)]
    hz_pts   = [_mel_hz(m) for m in mel_pts]
    bin_pts  = [int(h * N_FFT / SAMPLE_RATE) for h in hz_pts]

    fb = np.zeros((NUM_MEL_BINS, n_freqs), dtype=np.float)
    for m in range(1, NUM_MEL_BINS + 1):
        lo, ctr, hi = bin_pts[m-1], bin_pts[m], bin_pts[m+1]
        for k in range(lo, ctr):
            if ctr != lo:
                fb[m-1, k] = float(k - lo) / float(ctr - lo)
        for k in range(ctr, hi):
            if hi != ctr:
                fb[m-1, k] = float(hi - k) / float(hi - ctr)
    return fb

print("[INIT] Building Mel filterbank ...")
MEL_FB = _build_mel_filterbank()   # (40, 257)
print("[INIT] Mel filterbank ready")

# ---------------------------------------------------------------------------
# DCT-II via precomputed cosine table
#
# X[k] = sum_{n=0}^{N-1} x[n] * cos(pi * k * (2n+1) / (2N))
#
# Precomputing the (NUM_MFCC_COEFFS x NUM_MEL_BINS) matrix once at start-up
# reduces each per-frame DCT to a single np.dot() call.
# This replaces the FFT-mirror approach, which required length-2N=80 FFT --
# not a power of 2, which ulab rejects with ValueError.
# ---------------------------------------------------------------------------
def _build_dct_matrix(n_filters, n_coeffs):
    M = np.zeros((n_coeffs, n_filters), dtype=np.float)
    for k in range(n_coeffs):
        for n in range(n_filters):
            M[k, n] = math.cos(math.pi * k * (2 * n + 1) / (2.0 * n_filters))
    return M

print("[INIT] Building DCT matrix ...")
DCT_MATRIX = _build_dct_matrix(NUM_MEL_BINS, NUM_MFCC_COEFFS)  # (40, 40)
print("[INIT] DCT matrix ready")

def _dct2(x):
    # DCT-II of a 1-D ulab array of length NUM_MEL_BINS.
    # Returns array of length NUM_MFCC_COEFFS via a single matrix multiply.
    return np.dot(DCT_MATRIX, x)

# Feature extraction
# Replicates training notebook:
#   librosa.feature.mfcc(y, sr, n_mfcc=40, n_fft=512, hop_length=320)
#   result.T  →  (mfcc - mean) / (std + 1e-6)
# ---------------------------------------------------------------------------
def compute_features(samples_i16):
    """
    int16 PCM (16 000 samples) -> flat float32 ulab array of length 2040.

    1. Normalise int16 -> float32 [-1, 1]
    2. Centre-pad 256 zeros each side  (librosa default)
    3. Frame with Hann window, hop=320 -> 51 frames
    4. FFT power spectrum (ulab)
    5. Mel filterbank -> log energy
    6. DCT-II -> keep first 40 coefficients
    7. Transpose to (51, 40), normalise per-clip
    8. Flatten
    """
    # 1. Normalise
    sig = np.array(samples_i16, dtype=np.float) * (1.0 / 32768.0)

    # 2. Centre-pad (mirrors librosa centre=True)
    pad = np.zeros((PAD_LEN,), dtype=np.float)
    sig = np.concatenate((pad, sig, pad))

    n_frames = (len(sig) - N_FFT) // HOP_LENGTH + 1
    n_freqs  = N_FFT // 2 + 1

    # 3+4. Frame -> Hann window -> FFT magnitude squared
    power = np.zeros((n_freqs, n_frames), dtype=np.float)
    for t in range(n_frames):
        start  = t * HOP_LENGTH
        frame  = sig[start : start + N_FFT] * HANN
        # ulab fft returns (real_array, imag_array) tuple, not a complex ndarray
        sr, si       = np.fft.fft(frame)             # each length N_FFT
        sr           = sr[:n_freqs]                  # positive-frequency bins only
        si           = si[:n_freqs]
        power[:, t]  = sr * sr + si * si

    # 5. Mel filterbank + log
    #    MEL_FB: (40, 257)  power: (257, 51)  -> mel_spec: (40, 51)
    mel_spec = np.dot(MEL_FB, power)
    log_mel  = np.log(mel_spec + 1e-6)   # shape (40, 51)

    # 6. DCT-II per frame — keeps all NUM_MEL_BINS coefficients then slices
    mfcc = np.zeros((NUM_MFCC_COEFFS, n_frames), dtype=np.float)
    for t in range(n_frames):
        dct_col      = _dct2(log_mel[:, t])          # length NUM_MEL_BINS
        mfcc[:, t]   = dct_col[:NUM_MFCC_COEFFS]    # first 40

    # 7. Transpose -> (51, 40), normalise
    mfcc = mfcc.transpose()                          # (51, 40)
    mu   = np.mean(mfcc)
    std  = np.std(mfcc)
    mfcc = (mfcc - mu) / (std + 1e-6)

    # 8. Flatten row-major -> length 2040
    return mfcc.flatten()

# ---------------------------------------------------------------------------
# Quantise float32 -> shaped int8 np.array  (1, NUM_TIME_STEPS, NUM_MFCC_COEFFS)
#
# model.input_shape = (1, 51, 40) — 3-D, dtype int8 ('b').
# Passing a bytearray gives 'Unsupported input type'.
# Passing image.Image gives 'Expected input tensor with shape (1,H,W,C)'
# because the image path requires a 4-D NHWC tensor.
# Passing a ulab np.array with the correct shape and dtype is the right path.
# ---------------------------------------------------------------------------
def quantize_features(features_flat):
    # Quantize each float32 value to int8 using model scale/zero-point
    n   = NUM_TIME_STEPS * NUM_MFCC_COEFFS   # 2040
    s   = _Q_SCALE
    zp  = _Q_ZP
    flat = np.array(
        [max(-128, min(127, int(features_flat[i] / s) + zp)) for i in range(n)],
        dtype=np.int8
    )
    # Reshape to match model.input_shape (1, 51, 40)
    return flat.reshape((1, NUM_TIME_STEPS, NUM_MFCC_COEFFS))

# ---------------------------------------------------------------------------
# Audio capture
# ---------------------------------------------------------------------------
_pcm_buf  = bytearray(CLIP_SAMPLES * 2)   # raw int16 bytes from DMA
_buf_idx  = [0]
_buf_full = [False]

_raw_preview = None

@micropython.native
def _audio_callback(buf):
    global _raw_preview
    if _raw_preview is None:
        _raw_preview = bytes(buf[:32])  # save first 32 bytes
    remaining = (CLIP_SAMPLES - _buf_idx[0]) * 2
    chunk     = min(len(buf), remaining)
    _pcm_buf[_buf_idx[0]*2 : _buf_idx[0]*2 + chunk] = buf[:chunk]
    _buf_idx[0] += chunk // 2
    if _buf_idx[0] >= CLIP_SAMPLES:
        _buf_full[0] = True

def capture_audio():
    """Block until CLIP_SAMPLES int16 samples have been captured via DMA."""
    _buf_idx[0]  = 0
    _buf_full[0] = False
    audio.start_streaming(_audio_callback)
    while not _buf_full[0]:
        utime.sleep_ms(5)
    audio.stop_streaming()
    # Unpack raw bytes -> ulab int16 array
    import struct as _s
    vals = _s.unpack('{}h'.format(CLIP_SAMPLES), _pcm_buf)
    return np.array(vals, dtype=np.int16)

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
print("[RUN]  Listening for 'Hey Jarvis' ...")
print("[RUN]  Confidence threshold: {:.0%}".format(CONFIDENCE_THRESH))
print("-" * 48)

COOLDOWN_MS       = 1500
last_detection_ms = 0

while True:
    # 1. Capture
    samples = capture_audio()

    # Diagnostic 1: is audio capture actually working?
    samples_list = samples.tolist()
    sample_min = min(samples_list)
    sample_max = max(samples_list)
    sample_mean = sum(samples_list) / len(samples_list)
    print("[DBG] audio min={} max={} mean={:.1f}".format(sample_min, sample_max, sample_mean))

    # 2. MFCC features
    features = compute_features(samples)

    # Diagnostic 2: are features varying at all?
    feat_list = features.tolist()
    feat_min = min(feat_list)
    feat_max = max(feat_list)
    print("[DBG] features min={:.3f} max={:.3f}".format(feat_min, feat_max))

    # 3. Quantize -> shaped int8 np.array -> predict
    #    model.input_shape = (1, 51, 40), dtype int8
    #    Pass the np.array directly — image.Image requires 4-D NHWC (rejected),
    #    and raw bytearray gives 'Unsupported input type' (also rejected).
    try:
        '''
        tensor = quantize_features(features)   # shape (1, 51, 40), dtype int8
        print("[DBG] tensor.shape=", tensor.shape)
        flat = tensor.tolist()
        flat0 = flat[0]  # unwrap batch dim -> (51, 40)
        all_vals = [v for row in flat0 for v in row]
        print("[DBG] tensor min={} max={} unique={}".format(
            min(all_vals), max(all_vals), len(set(all_vals))))
        print("[DBG] Q scale={} zp={} feat_range=[{:.3f},{:.3f}]".format(
            _Q_SCALE, _Q_ZP, float(min(features.tolist())), float(max(features.tolist()))))
        output = model.predict([tensor])
        '''
        # Try passing flat int8 array instead of reshaped
        flat_tensor = np.array(
            [max(-128, min(127, int(features[i] / _Q_SCALE) + _Q_ZP))
             for i in range(NUM_TIME_STEPS * NUM_MFCC_COEFFS)],
            dtype=np.int8
        )
        print("[DBG] flat_tensor.shape=", flat_tensor.shape)
        try:
            output = model.predict([flat_tensor])
            print("[DBG] flat predict succeeded:", output)
        except Exception as e:
            print("[DBG] flat predict failed:", e)
    except Exception as e:
        print("[ERR]  Inference failed:", e)
        continue

    # 4. Scores — classes: hey_jarvis=0, negative=1, noise=2
    # output[0] contains raw int8 values from the quantized model.
    # Dequantize: float_val = (int8_val - output_zero_point) * output_scale
    # Extract each element via tolist() to get a plain Python int first.
    raw_scores = output[0].tolist()[0]       # output shape (1,3): tolist() gives [[a,b,c]], index [0] to get [a,b,c]
    out_scale  = model.output_scale[0]
    out_zp     = model.output_zero_point[0]
    hey_jarvis_score = (raw_scores[0] - out_zp) * out_scale

    # 5. Threshold + cooldown
    now_ms = utime.ticks_ms()
    if hey_jarvis_score > CONFIDENCE_THRESH:
        if utime.ticks_diff(now_ms, last_detection_ms) > COOLDOWN_MS:
            last_detection_ms = now_ms
            print("[WAKE] 'Hey Jarvis' detected!  confidence={:.1%}".format(
                  hey_jarvis_score))
            try:
                import pyb
                led = pyb.LED(1)
                led.on()
                utime.sleep_ms(300)
                led.off()
            except ImportError:
                pass
    else:
        # Uncomment to log every frame during development:
        print("[DBG]  hey_jarvis={:.3f}".format(float(hey_jarvis_score)))
        pass
