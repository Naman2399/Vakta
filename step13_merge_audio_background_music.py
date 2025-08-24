import os
import numpy as np
import librosa
import soundfile as sf

def _ensure_output_path(output_path: str) -> str:
    if os.path.isdir(output_path) or os.path.splitext(output_path)[1].lower() != ".wav":
        os.makedirs(output_path if os.path.isdir(output_path) else os.path.dirname(output_path) or ".", exist_ok=True)
        return os.path.join(output_path if os.path.isdir(output_path) else os.path.dirname(output_path), "final_mix.wav")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    return output_path

def _db_to_amp(db: float) -> float:
    return 10.0 ** (db / 20.0)

def _fade_in_out(x: np.ndarray, sr: int, fade_ms: int = 300) -> np.ndarray:
    n = len(x)
    f = int(sr * fade_ms / 1000)
    if f > 0 and n > 2 * f:
        ramp_in  = np.linspace(0.0, 1.0, f, endpoint=True, dtype=np.float32)
        ramp_out = np.linspace(1.0, 0.0, f, endpoint=True, dtype=np.float32)
        x[:f]  *= ramp_in
        x[-f:] *= ramp_out
    return x

def _tile_or_trim(y: np.ndarray, target_len: int) -> np.ndarray:
    if len(y) >= target_len:
        return y[:target_len]
    reps = int(np.ceil(target_len / len(y)))
    return np.tile(y, reps)[:target_len]

def _activity_envelope(narr: np.ndarray, sr: int, frame_ms: int, attack_ms: int, release_ms: int) -> np.ndarray:
    # Short-time RMS → normalized → binary activity
    frame_len = max(256, int(sr * frame_ms / 1000))
    hop_len = max(128, frame_len // 2)
    rms = librosa.feature.rms(y=narr, frame_length=frame_len, hop_length=hop_len, center=True)[0]
    if rms.max() > 0:
        rms_norm = rms / rms.max()
    else:
        rms_norm = rms

    # Adaptive threshold: keep narration leading
    thr = max(0.02, float(np.quantile(rms_norm, 0.6)))
    act = (rms_norm > thr).astype(np.float32)

    # Upsample to per-sample
    act_samples = np.repeat(act, hop_len)
    if len(act_samples) < len(narr):
        act_samples = np.pad(act_samples, (0, len(narr) - len(act_samples)), mode="edge")
    else:
        act_samples = act_samples[:len(narr)]

    # Attack/Release smoothing (envelope follower)
    attack_a  = np.exp(-1.0 / max(1, int(sr * attack_ms  / 1000)))
    release_a = np.exp(-1.0 / max(1, int(sr * release_ms / 1000)))
    env = np.zeros_like(act_samples, dtype=np.float32)
    prev = 0.0
    for i, x in enumerate(act_samples):
        if x > prev:
            prev = attack_a * prev + (1 - attack_a) * x
        else:
            prev = release_a * prev + (1 - release_a) * x
        env[i] = prev
    return env

def dynamic_ducking_numpy(
    narration_path: str,
    music_path: str,
    output_path: str,
    # Make narration clearly dominant:
    music_base_db: float = -14.0,   # constant under-bed level (always applied)
    duck_extra_db: float = -10.0,   # additional reduction during speech
    frame_ms: int = 160,
    attack_ms: int = 60,
    release_ms: int = 220,
    final_fade_ms: int = 300,
):
    if not os.path.exists(narration_path):
        raise FileNotFoundError(f"Narration not found: {narration_path}")
    if not os.path.exists(music_path):
        raise FileNotFoundError(f"Music not found: {music_path}")

    # Load (mono); keep narration sample rate as master
    narr, sr = librosa.load(narration_path, sr=None, mono=True)
    music, _ = librosa.load(music_path, sr=sr, mono=True)

    # Length match
    music = _tile_or_trim(music, len(narr))

    # Build smoothed activity envelope (0..1)
    env = _activity_envelope(narr, sr, frame_ms=frame_ms, attack_ms=attack_ms, release_ms=release_ms)

    # Convert dB params to gains
    base_gain = _db_to_amp(music_base_db)     # e.g., -14 dB → ~0.20
    duck_gain = _db_to_amp(duck_extra_db)     # e.g., -10 dB → ~0.32

    # Music gain curve: base under-bed, and when env→1 apply extra duck
    # gain = base * [ 1 - env * (1 - duck_gain) ]
    gain_curve = base_gain * (1.0 - env * (1.0 - duck_gain))

    # Apply gain to music, add narration on top
    music_ducked = (music * gain_curve).astype(np.float32)
    mix = narr.astype(np.float32) + music_ducked

    # Soft-clip / normalize to avoid peaks
    peak = float(np.max(np.abs(mix)) + 1e-12)
    if peak > 1.0:
        mix = (mix / peak).astype(np.float32)

    # Gentle intro/outro fades
    mix = _fade_in_out(mix, sr, fade_ms=final_fade_ms)

    # Save
    out_path = _ensure_output_path(output_path)
    sf.write(out_path, mix, sr)
    print(f"✅ Final mix saved: {out_path}")

# Example:
dynamic_ducking_numpy(
    "books/sample_v2/chapter_1/narration_speech/narration_1.wav",
    "books/sample_v2/chapter_1/background_music/background_music_1.wav",
    "books/sample_v2/chapter_1/merge_narration_background_music",

    # --- Adjusted values ---
    music_base_db=-14.0,   # lower constant level (was -14.0)
    duck_extra_db=-10.0,   # stronger dip during speech (was -10.0)

    frame_ms=160,
    attack_ms=60,
    release_ms=220,
    final_fade_ms=300,
)
