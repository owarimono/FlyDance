"""Detect beats in an audio file -> beat grid + onset envelope + chroma.

Offline: returns a time grid, not a live stream. The animation is then a
pure function of t, so there is no sync drift.
"""

import json
import sys

import numpy as np
import librosa

# librosa's default hop; onset_env indices map to time as t = i * hop / sr
HOP_LENGTH = 512


def analyze(path, sr=22050):
    y, _ = librosa.load(path, sr=sr, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    tempo_bpm = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    # peak-normalise to 0..1 so loud choruses do not blow up the controller
    env = onset_env / (onset_env.max() + 1e-9)
    env = np.clip((env - env.min()) / (np.ptp(env) + 1e-9), 0, 1)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    return {
        "sr": sr,
        "duration_s": float(len(y) / sr),
        "tempo_bpm": tempo_bpm,
        "beat_times": beat_times.tolist(),
        "onset_env": env.tolist(),
        "onset_hop": HOP_LENGTH,
        # dominant pitch class per second-ish bucket, 0..11
        "chroma": chroma.mean(axis=1).tolist(),
    }


if __name__ == "__main__":
    out = analyze(sys.argv[1])
    dest = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1] + ".beats.json"
    with open(dest, "w") as f:
        json.dump(out, f)
    print(f"tempo {out['tempo_bpm']:.1f} BPM, {len(out['beat_times'])} beats, "
          f"{out['duration_s']:.1f}s -> {dest}")
