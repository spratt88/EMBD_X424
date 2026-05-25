"""
Wake Word Detector - "Hey Jarvis"
OpenMV MicroPython Application

Hardware: OpenMV Cam (H7 / H7 Plus / RT recommended for TFLite)
Model:    wake_word_svdf.tflite  (place on SD card or internal flash)

Audio configuration
-------------------
Sample rate : 16 000 Hz
Channels    : Mono
Sample type : int16
Clip length : 1 second  → 16 000 samples per inference window
"""

import audio
import tf
import utime
import math
import array
import micropython
import ml

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SAMPLE_RATE       = 16000          # Hz
NUM_CHANNELS      = 1              # Mono
CLIP_SAMPLES      = SAMPLE_RATE    # 1-second window
CONFIDENCE_THRESH = 0.85           # Minimum score to accept detection
MODEL_PATH        = "/rom/wake_word_svdf_int8.tflite"

# MFCC / spectrogram hyper-parameters (must match training configuration)
FRAME_LEN         = 512            # FFT frame length (samples)
FRAME_STEP        = 160            # Hop length (10 ms @ 16 kHz)
NUM_MEL_BINS      = 40             # Mel filterbank bands
NUM_MFCC_COEFFS   = 13             # Cepstral coefficients kept
PRE_EMPHASIS      = 0.97           # Pre-emphasis filter coefficient

# ---------------------------------------------------------------------------
# Model & audio initialisation
# ---------------------------------------------------------------------------
print("[INIT] Loading TFLite model:", MODEL_PATH)
# model = ml.Model(MODEL_PATH, load_to_fb=True)   # load_to_fb uses frame buffer RAM
model = tf.load(MODEL_PATH, load_to_fb=True)   # load_to_fb uses frame buffer RAM

# Audio object - uses PDM microphone on OpenMV boards
# audio.init() signature: init(channels, frequency, gain_db, highpass)
audio.init(channels=NUM_CHANNELS,
           frequency=SAMPLE_RATE,
           gain_db=24,
           highpass=0.9883)

print("[INIT] Microphone ready  @ {}Hz  {}ch".format(SAMPLE_RATE, NUM_CHANNELS))

# ---------------------------------------------------------------------------
# Pre-emphasis filter
# ---------------------------------------------------------------------------
def apply_pre_emphasis(samples, coeff=PRE_EMPHASIS):
    """High-pass filter that boosts high-frequency content before FFT."""
    out = array.array('f', samples)
    for i in range(len(out) - 1, 0, -1):
        out[i] = out[i] - coeff * out[i - 1]
    return out


# ---------------------------------------------------------------------------
# Mel filterbank (computed once at start-up)
# ---------------------------------------------------------------------------
def _hz_to_mel(hz):
    return 2595.0 * math.log10(1.0 + hz / 700.0)

def _mel_to_hz(mel):
    return 700.0 * (math.pow(10.0, mel / 2595.0) - 1.0)

def _build_mel_filterbank(sr, n_fft, n_mels, f_min=20.0, f_max=None):
    """Returns a (n_mels x (n_fft//2+1)) filterbank as a list of lists."""
    if f_max is None:
        f_max = sr / 2.0
    n_freqs = n_fft // 2 + 1

    mel_min = _hz_to_mel(f_min)
    mel_max = _hz_to_mel(f_max)
    mel_pts = [mel_min + i * (mel_max - mel_min) / (n_mels + 1)
               for i in range(n_mels + 2)]
    hz_pts  = [_mel_to_hz(m) for m in mel_pts]
    bin_pts = [int(h * n_fft / sr) for h in hz_pts]

    filters = []
    for m in range(1, n_mels + 1):
        filt = [0.0] * n_freqs
        for k in range(bin_pts[m - 1], bin_pts[m]):
            if bin_pts[m] != bin_pts[m - 1]:
                filt[k] = (k - bin_pts[m - 1]) / (bin_pts[m] - bin_pts[m - 1])
        for k in range(bin_pts[m], bin_pts[m + 1]):
            if bin_pts[m + 1] != bin_pts[m]:
                filt[k] = (bin_pts[m + 1] - k) / (bin_pts[m + 1] - bin_pts[m])
        filters.append(filt)
    return filters

print("[INIT] Building Mel filterbank …")
MEL_FILTERS = _build_mel_filterbank(SAMPLE_RATE, FRAME_LEN, NUM_MEL_BINS)
print("[INIT] Mel filterbank ready ({} bands)".format(NUM_MEL_BINS))


# ---------------------------------------------------------------------------
# Hann window (pre-computed)
# ---------------------------------------------------------------------------
HANN_WINDOW = array.array('f',
    [0.5 - 0.5 * math.cos(2.0 * math.pi * n / (FRAME_LEN - 1))
     for n in range(FRAME_LEN)])


# ---------------------------------------------------------------------------
# Minimal fixed-point FFT (radix-2 DIT, returns magnitude spectrum)
# ---------------------------------------------------------------------------
def _fft_magnitude(frame):
    """
    Compute magnitude spectrum of *frame* (length must be power-of-two).
    Returns a list of length (N//2 + 1).
    """
    N = len(frame)
    # Bit-reverse copy into complex list [real, imag, real, imag …]
    bits = int(math.log(N, 2))
    rev  = [0] * N
    for i in range(N):
        r = 0
        x = i
        for _ in range(bits):
            r = (r << 1) | (x & 1)
            x >>= 1
        rev[r] = frame[i]

    # Work with separate real/imag arrays for speed
    re = array.array('f', rev)
    im = array.array('f', [0.0] * N)

    step = 2
    while step <= N:
        half  = step >> 1
        angle = -2.0 * math.pi / step
        for k in range(0, N, step):
            for j in range(half):
                theta    = angle * j
                wr       = math.cos(theta)
                wi       = math.sin(theta)
                idx_even = k + j
                idx_odd  = k + j + half
                tr = wr * re[idx_odd] - wi * im[idx_odd]
                ti = wr * im[idx_odd] + wi * re[idx_odd]
                re[idx_odd]  = re[idx_even] - tr
                im[idx_odd]  = im[idx_even] - ti
                re[idx_even] = re[idx_even] + tr
                im[idx_even] = im[idx_even] + ti
        step <<= 1

    # Magnitude of positive-frequency bins only
    mag = [math.sqrt(re[k] * re[k] + im[k] * im[k]) for k in range(N // 2 + 1)]
    return mag


# ---------------------------------------------------------------------------
# DCT-II (for converting log-mel energies → MFCC)
# ---------------------------------------------------------------------------
def _dct2(x, n_keep):
    """Return first *n_keep* DCT-II coefficients of vector *x*."""
    N    = len(x)
    out  = []
    for k in range(n_keep):
        s = 0.0
        for n in range(N):
            s += x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N))
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# Feature extraction: int16 PCM  →  MFCC matrix  →  flat float32 array
# ---------------------------------------------------------------------------
def compute_features(samples_i16):
    """
    Convert a 1-second int16 PCM buffer into a flattened MFCC feature vector
    suitable for the wake-word SVDF model.

    Steps
    -----
    1. Convert int16 → normalised float32  [-1, 1]
    2. Apply pre-emphasis
    3. Slice into overlapping frames (FRAME_LEN, hop FRAME_STEP)
    4. Apply Hann window
    5. FFT → magnitude spectrum
    6. Mel filterbank → log energy
    7. DCT → MFCC coefficients
    8. Return flat array('f')
    """
    # 1. Normalise
    scale  = 1.0 / 32768.0
    floats = array.array('f', [s * scale for s in samples_i16])

    # 2. Pre-emphasis
    floats = apply_pre_emphasis(floats)

    mfcc_frames = []

    # 3. Frame the signal
    start = 0
    while start + FRAME_LEN <= len(floats):
        # 4. Window
        frame = array.array('f',
            [floats[start + i] * HANN_WINDOW[i] for i in range(FRAME_LEN)])

        # 5. FFT magnitude
        mag = _fft_magnitude(frame)

        # 6. Mel filterbank + log
        mel_energies = []
        for filt in MEL_FILTERS:
            energy = sum(filt[k] * mag[k] for k in range(len(mag)))
            mel_energies.append(math.log(energy + 1e-6))

        # 7. DCT → MFCC
        coeffs = _dct2(mel_energies, NUM_MFCC_COEFFS)
        mfcc_frames.append(coeffs)

        start += FRAME_STEP

    # Flatten to 1-D float32 array expected by the model
    flat = array.array('f')
    for frame_coeffs in mfcc_frames:
        flat.extend(frame_coeffs)

    return flat


# ---------------------------------------------------------------------------
# Audio capture
# ---------------------------------------------------------------------------
# Internal ring buffer filled by the DMA callback
_pcm_buf  = array.array('h', [0] * CLIP_SAMPLES)   # int16 ring buffer
_buf_idx  = [0]                                      # mutable index in list
_buf_full = [False]

@micropython.native
def _audio_callback(buf):
    """Called by the audio DMA interrupt with a chunk of int16 samples."""
    global _pcm_buf, _buf_idx, _buf_full
    remaining = CLIP_SAMPLES - _buf_idx[0]
    chunk_len  = min(len(buf) // 2, remaining)   # buf is bytes; each sample = 2 bytes

    for i in range(chunk_len):
        # Read little-endian int16 from bytes object
        lo = buf[i * 2]
        hi = buf[i * 2 + 1]
        val = (hi << 8) | lo
        if val >= 0x8000:
            val -= 0x10000
        _pcm_buf[_buf_idx[0] + i] = val

    _buf_idx[0] += chunk_len
    if _buf_idx[0] >= CLIP_SAMPLES:
        _buf_full[0] = True


def capture_audio():
    """
    Block until exactly CLIP_SAMPLES (1 second) of audio has been captured
    via the microphone DMA callback.  Returns an array('h') of int16 samples.
    """
    # Reset state
    _buf_idx[0]  = 0
    _buf_full[0] = False

    audio.start_streaming(_audio_callback)

    # Spin-wait; the callback fills _pcm_buf in background
    while not _buf_full[0]:
        utime.sleep_ms(5)

    audio.stop_streaming()

    # Return a copy so the buffer can be re-used immediately
    return array.array('h', _pcm_buf)


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------
print("[RUN]  Listening for wake word: 'Hey Jarvis' …")
print("[RUN]  Confidence threshold: {:.0%}".format(CONFIDENCE_THRESH))
print("-" * 48)

detection_cooldown_ms = 1500    # Ignore re-triggers within this window
last_detection_ms     = 0

while True:
    # --- 1. Capture audio ---
    samples = capture_audio()

    # --- 2. Compute MFCC features ---
    features = compute_features(samples)

    # --- 3. Run TFLite model ---
    # tf.classify() returns a list of tf.Classification objects
    # Each has .index() and .value() (confidence score 0-1)
    try:
        predictions = model.classify(features, min_confidence=0.0)
    except Exception as e:
        print("[ERR]  Inference failed:", e)
        continue

    # --- 4. Extract "hey_jarvis" class score ---
    # By convention class 0 = background/silence, class 1 = wake word.
    # Adjust the index below if your model uses a different ordering.
    hey_jarvis_score = 0.0
    for pred in predictions:
        if pred.index() == 1:          # index 1 = "hey_jarvis"
            hey_jarvis_score = pred.value()
            break

    # --- 5. Apply confidence threshold & trigger action ---
    now_ms = utime.ticks_ms()

    if hey_jarvis_score > CONFIDENCE_THRESH:
        if utime.ticks_diff(now_ms, last_detection_ms) > detection_cooldown_ms:
            last_detection_ms = now_ms

            print("[WAKE] 'Hey Jarvis' detected!  confidence={:.1%}".format(hey_jarvis_score))

            # ----------------------------------------------------------------
            # TODO: Replace / extend the action below with your application
            # logic (e.g. blink an LED, send a UART command, start a task).
            # ----------------------------------------------------------------
            # Example: toggle the red LED for visual feedback
            try:
                import pyb
                led = pyb.LED(1)   # LED 1 = red on most OpenMV boards
                led.on()
                utime.sleep_ms(300)
                led.off()
            except ImportError:
                pass   # pyb not available on all builds; safe to ignore

        # else: within cooldown window, silently drop duplicate trigger

    else:
        # Optional: uncomment to log every frame score during development
        # print("[DBG]  hey_jarvis={:.3f}".format(hey_jarvis_score))
        pass
