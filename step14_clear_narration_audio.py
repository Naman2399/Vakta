import os
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from scipy.signal import butter, lfilter

def _butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return b, a

def _highpass_filter(data, cutoff, fs, order=5):
    b, a = _butter_highpass(cutoff, fs, order=order)
    return lfilter(b, a, data)

def enhance_narration(
    input_path: str,
    output_path: str,
    noise_reduce: bool = True,
    target_rms: float = -20.0,  # LUFS-ish loudness target
    highpass_hz: int = 80,
    clarity_boost_db: float = 3.0,
):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    # Load mono
    y, sr = librosa.load(input_path, sr=None, mono=True)

    # Noise reduction
    if noise_reduce:
        # Estimate noise from first 0.5s
        noise_clip = y[: int(sr * 0.5)]
        y = nr.reduce_noise(y=y, sr=sr, y_noise=noise_clip)

    # High-pass filter (remove rumble, AC noise)
    y = _highpass_filter(y, highpass_hz, sr)

    # Slight clarity boost in presence band (2-4kHz)
    S = librosa.stft(y)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=S.shape[0] * 2 - 2)
    boost_band = (freqs >= 2000) & (freqs <= 4000)
    gain = np.ones_like(freqs)
    gain[boost_band] *= 10 ** (clarity_boost_db / 20)
    S_boosted = S * gain[:, None]
    y = librosa.istft(S_boosted)

    # Normalize to target RMS
    rms = np.sqrt(np.mean(y ** 2))
    target_amp = 10 ** (target_rms / 20)
    if rms > 0:
        y = y * (target_amp / rms)

    # Apply soft limiter to avoid clipping
    peak = np.max(np.abs(y))
    if peak > 0.99:
        y = y / peak * 0.99

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sf.write(output_path, y.astype(np.float32), sr)
    print(f"✅ Enhanced narration saved: {output_path}")

# Example:
# enhance_narration("raw_narration.wav", "cleaned_narration.wav")

if __name__ == "__main__":

    dir_path = "books/sample/musical_prompt/test_10"
    output_path = "books/sample/musical_prompt/test_10_converted"
    for i in range(1, 35):
        narration_path = os.path.join(dir_path, f"narration_{i}.wav")
        output_path = os.path.join(dir_path, f"enhanced_narration_{i}.wav")
        enhance_narration(narration_path, output_path)