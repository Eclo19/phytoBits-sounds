import numpy as np
import wave
import matplotlib.pyplot as plt
import librosa  

SR = 44100

# Pad Chord
PAD_NOTES = ["C2", "G2", "C3", "C4", "E4", "G4", "B4", "D5"]   

# Main melody
LEAD_MELODY = ["C4","D4","G4","B4","G4","G4","E4","D4"]

# Scale
MAJOR_SCALE_C = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]

# Note to freq math
_NOTE_INDEX = {
    "C": 0, "C#": 1, "Db": 1,
    "D": 2, "D#": 3, "Eb": 3,
    "E": 4,
    "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10,
    "B": 11,
}

def note_to_freq(note: str) -> float:
    """
    Examples:
      'A4'  -> 440.0
      'C4'  -> 261.625...
      'F#3' -> 184.997...
    """
    note = note.strip()
    if len(note) < 2:
        raise ValueError(f"Bad note: {note}")

    # pitch class + octave
    if len(note) >= 3 and note[1] in ("#", "b"):
        pc = note[:2]
        octave_str = note[2:]
    else:
        pc = note[:1]
        octave_str = note[1:]

    if pc not in _NOTE_INDEX:
        raise ValueError(f"Bad pitch class in note: {note}")

    octave = int(octave_str)
    midi = (octave + 1) * 12 + _NOTE_INDEX[pc]  # C-1=0, A4=69
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))

# ============================================================
# Core helpers
# ============================================================

def fade_out(x, sr=SR, fade_ms=200.0):
    """
    Apply a linear fade-out over the last fade_ms milliseconds.
    """
    x = np.asarray(x, dtype=np.float32).copy()
    fadeN = int((fade_ms / 1000.0) * sr)
    fadeN = max(1, min(fadeN, len(x)))
    w = np.ones(len(x), dtype=np.float32)
    w[-fadeN:] = np.linspace(1.0, 0.0, fadeN, dtype=np.float32)
    return x * w

def normalize_audio(x, peak=0.98, eps=1e-12):
    m = np.max(np.abs(x)) + eps
    return (x / m) * peak

def clamp01(x):
    return np.clip(x, 0.0, 1.0)

def adsr(N, a=0.01, d=0.15, s=0.6, r=0.2):
    aN = int(a * N)
    dN = int(d * N)
    rN = int(r * N)
    sN = max(0, N - aN - dN - rN)
    env = np.zeros(N, dtype=np.float32)
    i = 0
    if aN > 0:
        env[i:i+aN] = np.linspace(0, 1, aN, endpoint=False); i += aN
    if dN > 0:
        env[i:i+dN] = np.linspace(1, s, dN, endpoint=False); i += dN
    if sN > 0:
        env[i:i+sN] = s; i += sN
    if rN > 0:
        env[i:i+rN] = np.linspace(s, 0, rN, endpoint=True)
    return env

def resample_linear(control, N):
    if len(control) == N:
        return control.astype(np.float32)
    xp = np.linspace(0, 1, len(control))
    xq = np.linspace(0, 1, N)
    return np.interp(xq, xp, control).astype(np.float32)

def safe_control(control):
    """
    Normalize to 0..1. If flat, return all zeros (NOT NaNs).
    """
    c = control.astype(np.float32)
    mn = float(np.min(c))
    mx = float(np.max(c))
    if mx - mn < 1e-12:
        return np.zeros_like(c, dtype=np.float32)
    c = (c - mn) / (mx - mn)
    return clamp01(c)

# ============================================================
# Filters 
# ============================================================
def one_pole_lpf_varying(x, cutoff_hz, sr=SR, y0=0.0):
    """
    One-pole LPF with per-sample cutoff (array) and continuous state.
    cutoff_hz can be scalar or array length N.
    """
    x = np.asarray(x, dtype=np.float32)
    N = len(x)

    if np.isscalar(cutoff_hz):
        cutoff = np.full(N, float(cutoff_hz), dtype=np.float32)
    else:
        cutoff = np.asarray(cutoff_hz, dtype=np.float32)
        if len(cutoff) != N:
            raise ValueError("cutoff_hz must be scalar or same length as x")

    cutoff = np.clip(cutoff, 5.0, 0.49 * sr)

    y = np.zeros(N, dtype=np.float32)
    y_prev = np.float32(y0)

    for n in range(N):
        a = np.exp(-2.0 * np.pi * cutoff[n] / sr).astype(np.float32)
        y_prev = a * y_prev + (1.0 - a) * x[n]
        y[n] = y_prev

    return y

def smooth_signal(x, cutoff_hz=2.0, sr=SR):
    """
    Smooth any control-like signal (e.g., cutoff array) with a slow LPF.
    """
    return one_pole_lpf_varying(np.asarray(x, dtype=np.float32), cutoff_hz, sr)

def one_pole_lpf(x, cutoff_hz, sr=SR):
    # convenience wrapper for scalar cutoff
    return one_pole_lpf_varying(x, float(cutoff_hz), sr)

def one_pole_hpf(x, cutoff_hz, sr=SR):
    # HPF via x - LPF(x)
    return np.asarray(x, dtype=np.float32) - one_pole_lpf(np.asarray(x, dtype=np.float32), cutoff_hz, sr)

# ============================================================
# Instruments & Their helpers
# ============================================================

# ---------- PAD ---------- #

def _soft_saw(phase, n_harm=12):
    """
    Band-limited-ish saw approximation via harmonic sum.
    phase: radians array
    """
    y = np.zeros_like(phase, dtype=np.float32)
    for k in range(1, n_harm + 1):
        y += (1.0 / k) * np.sin(k * phase).astype(np.float32)
    # normalize-ish
    y *= (1.0 / np.sum(1.0 / np.arange(1, n_harm + 1)))
    return y.astype(np.float32)

def _schroeder_reverb(x, sr=SR, wet=0.32):
    """
    Lightweight Schroeder reverb: 4 combs in parallel -> 2 allpasses in series.
    Pure numpy, stable, good enough for a pad.
    """
    x = np.asarray(x, dtype=np.float32)
    N = len(x)

    def comb_filter(x, delay_samps, feedback):
        y = np.zeros_like(x, dtype=np.float32)
        for n in range(N):
            y[n] = x[n]
            if n - delay_samps >= 0:
                y[n] += feedback * y[n - delay_samps]
        return y

    def allpass_filter(x, delay_samps, g):
        y = np.zeros_like(x, dtype=np.float32)
        for n in range(N):
            xn = x[n]
            xd = x[n - delay_samps] if n - delay_samps >= 0 else 0.0
            yd = y[n - delay_samps] if n - delay_samps >= 0 else 0.0
            y[n] = -g * xn + xd + g * yd
        return y

    # Comb delays (seconds -> samples) chosen to avoid obvious ringing
    comb_delays = [int(sr * d) for d in (0.0297, 0.0371, 0.0411, 0.0437)]
    comb_fb = 0.78

    comb_sum = np.zeros_like(x, dtype=np.float32)
    for d in comb_delays:
        comb_sum += comb_filter(x, d, comb_fb)
    comb_sum *= 0.25

    # Allpass chain
    ap1 = allpass_filter(comb_sum, int(sr * 0.005), 0.7)
    ap2 = allpass_filter(ap1,      int(sr * 0.0017), 0.7)

    # Wet/dry
    return (1.0 - wet) * x + wet * ap2


def _osc_from_freq(f_hz, sr=SR, phase0=0.0):
    """
    Numerically stable oscillator phase:
      - float64 accumulation
      - wrap modulo 2π to prevent huge phase values
    Returns phase array in float64.
    """
    f_hz = np.asarray(f_hz, dtype=np.float64)
    inc = (2.0 * np.pi * f_hz) / float(sr)                  # radians/sample
    phase = phase0 + np.cumsum(inc, dtype=np.float64)
    phase = np.mod(phase, 2.0 * np.pi)                      # keep bounded
    return phase

def pad(control, sr=SR, notes=PAD_NOTES):
    """
    More aggressive control modulation pad (still stable, no long-term pitch drift):
      - control boosts peaks (gamma curve) -> bigger perceived movement
      - much wider amp + cutoff modulation
      - faster cutoff smoothing (more reactive)
      - stronger vibrato modulation
      - optional "edge" component that increases with control
    """
    N = len(control)
    t = (np.arange(N, dtype=np.float64) / float(sr))         # float64 time
    c = clamp01(control).astype(np.float64)

    # --- make control hit harder near peaks ---
    c_hot = np.power(c, 0.55)  # gamma < 1 => emphasizes highs (aggressive)

    # --- musical params (more aggressive) ---
    base_amp = 0.10
    amp = (base_amp + 0.55 * c_hot).astype(np.float64)       # big dynamic swing

    base_cutoff = 350.0
    cutoff = base_cutoff + 12000.0 * c_hot                   # huge sweep range
    cutoff = smooth_signal(cutoff.astype(np.float32), cutoff_hz=5.0, sr=sr).astype(np.float64)
    cutoff *= (0.85 + 0.15 * np.sin(2*np.pi*0.03*t))         # a bit more motion

    freqs = np.array([note_to_freq(n) for n in notes], dtype=np.float64)

    # Fixed detune (constant, small)
    dets = np.array([0.9992, 1.0000, 1.0008], dtype=np.float64)

    # Vibrato (bounded), then force EXACT zero-mean so it cannot bias pitch
    vib_rate  = 0.18 + 0.35 * c_hot
    vib_depth = 0.0006 + 0.0012 * c_hot
    vib_raw = vib_depth * np.sin(2*np.pi*vib_rate*t)
    vib = vib_raw - np.mean(vib_raw)                         # exact zero-mean over buffer

    rng = np.random.default_rng(0)
    vib_phase_per_note = rng.uniform(0, 2*np.pi, size=len(freqs)).astype(np.float64)

    x = np.zeros(N, dtype=np.float64)
    voice_gain = 0.22 / max(1, len(freqs))

    for idx, f0 in enumerate(freqs):
        # per-note vibrato phase offset; also force zero-mean
        vib_n_raw = vib_depth * np.sin(2*np.pi*vib_rate*t + vib_phase_per_note[idx])
        vib_n = vib_n_raw - np.mean(vib_n_raw)

        for d in dets:
            f = f0 * d * (1.0 + vib_n)                       # float64, stable
            phase = _osc_from_freq(f, sr=sr, phase0=0.0)      # wrapped float64 phase

            # layers (cast to float32 at the LAST moment for speed)
            sine = np.sin(phase).astype(np.float32)
            saw  = _soft_saw(phase.astype(np.float32), n_harm=10)
            sub  = np.sin(0.5 * phase).astype(np.float32)

            voice = (0.58 * sine + 0.32 * saw + 0.10 * sub).astype(np.float32)
            x += voice_gain * voice.astype(np.float64)

    # amplitude breathing (not pitch)
    lfo = (0.84 + 0.16*np.sin(2*np.pi*0.07*t)).astype(np.float64)
    x *= amp * lfo

    # filter
    y = one_pole_lpf_varying(x.astype(np.float32), cutoff.astype(np.float32), sr).astype(np.float32)

    # --- optional "edge / bite" that increases with control ---
    edge = (y - one_pole_lpf(y, 900.0, sr)).astype(np.float32)
    y = (y + (0.10 + 0.35 * c_hot.astype(np.float32)) * edge).astype(np.float32)

    # reverb + envelope + safety
    y = _schroeder_reverb(y, sr=sr, wet=0.32)
    y *= adsr(N, a=0.03, d=0.12, s=0.95, r=0.35)
    y = np.tanh(y * 1.1).astype(np.float32)

    return y


# ---------- Rhythm ---------- #

def _beats_to_steps_in_pattern(beats, steps_per_beat, steps_per_pattern):
    """
    Convert beat positions within a multi-bar pattern to integer 16th-step indices.
    beats: iterable of floats, where 0.0..pattern_beats defines one full pattern.
    """
    s = []
    for b in beats:
        step = int(np.round(float(b) * steps_per_beat)) % steps_per_pattern
        s.append(step)
    return sorted(set(s))


def rhythm(
    control,
    bpm=110,
    sr=SR,
    # Patterns are given in BEATS within a repeating pattern window.
    # If bar_beats=4 and pattern_bars=2, valid beat positions are [0..8).
    kick_beats=(0.0, 1.0, 2.0, 3.0),                 # default: 4-on-the-floor (per 1 bar)
    hat_beats=(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5),  # default: 8ths (per 1 bar)
    bar_beats=4.0,
    pattern_bars=1,   # <-- set 2 for 2-bar patterns, etc.
    # Optional “conditional extras” (like your old control-driven extra 16ths)
    hat_extra_16ths_when_control_gt=0.65,
    hat_extra_16ths_beats=(0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75),
):
    """
    Kick + hats with beat-specified patterns over a repeating multi-bar pattern window.

    Timing grid matches your original drum engine:
      step_dur = spb/4 (16th), step_samps = int(step_dur*sr)

    Examples (4/4):
      - pattern_bars=1 => beats in [0..4)
      - pattern_bars=2 => beats in [0..8)  (two measures)
    """
    N = len(control)
    c = clamp01(control)

    # --- Drum timing grid (unchanged) ---
    spb = 60.0 / bpm
    step_dur = spb / 4.0                # 16th note duration (seconds)
    step_samps = max(1, int(step_dur * sr))
    n_steps = int(np.ceil(N / step_samps))

    # control per 16th step
    c_step = np.array(
        [float(np.mean(c[i * step_samps : min(N, (i + 1) * step_samps)])) for i in range(n_steps)],
        dtype=np.float32,
    )

    out = np.zeros(N, dtype=np.float32)

    # --- Pattern window definition ---
    steps_per_beat = 4                           # 16ths: 4 steps per quarter note
    steps_per_bar = int(round(bar_beats * steps_per_beat))
    pattern_bars = int(max(1, pattern_bars))
    steps_per_pattern = steps_per_bar * pattern_bars

    # Map beats -> step positions within pattern
    kick_steps = _beats_to_steps_in_pattern(kick_beats, steps_per_beat, steps_per_pattern)
    hat_steps  = _beats_to_steps_in_pattern(hat_beats,  steps_per_beat, steps_per_pattern)
    hat_extra_steps = _beats_to_steps_in_pattern(hat_extra_16ths_beats, steps_per_beat, steps_per_pattern)

    # --- KICK rendering ---
    for s in range(n_steps):
        if (s % steps_per_pattern) not in kick_steps:
            continue

        start = s * step_samps
        if start >= N:
            continue

        dur = int(0.20 * sr)  # slightly longer for clean decay
        end = min(N, start + dur)
        L = end - start
        tt = np.arange(L, dtype=np.float32) / sr

        cs = c_step[s]
        punch = 1.0 + 0.25 * cs

        # pitch sweep
        f0 = 120.0 + 50.0 * cs
        f1 = 35.0
        f = f1 + (f0 - f1) * np.exp(-tt * 28.0)
        phase = 2 * np.pi * np.cumsum(f) / sr

        env = np.exp(-tt * 18.0).astype(np.float32)
        click = (np.random.randn(L).astype(np.float32) * 0.015) * np.exp(-tt * 160.0)

        kick = (np.sin(phase).astype(np.float32) * env + click) * punch

        # POP FIXES
        kick -= np.mean(kick).astype(np.float32)                   # remove DC
        tail = kick[-1]
        kick = kick - np.linspace(0.0, tail, L, dtype=np.float32)  # end exactly at 0

        fade_ms = 6.0
        fadeN = min(int((fade_ms / 1000.0) * sr), L)
        if fadeN > 1:
            w = np.ones(L, dtype=np.float32)
            w[-fadeN:] = np.linspace(1.0, 0.0, fadeN, dtype=np.float32)
            kick *= w

        out[start:end] += 0.9 * kick

    # --- HAT rendering ---
    for s in range(n_steps):
        pos = (s % steps_per_pattern)
        cs = c_step[s]

        is_hat = (pos in hat_steps)
        is_extra = (cs > hat_extra_16ths_when_control_gt) and (pos in hat_extra_steps)

        if not (is_hat or is_extra):
            continue

        start = s * step_samps
        if start >= N:
            continue

        dur = int(0.045 * sr)
        end = min(N, start + dur)
        L = end - start
        tt = np.arange(L, dtype=np.float32) / sr

        env = np.exp(-tt * (70.0 + 30.0 * cs)).astype(np.float32)

        noise = np.random.randn(L).astype(np.float32)
        cutoff = 3500.0 + 7000.0 * cs
        hat = one_pole_hpf(one_pole_lpf(noise, cutoff, sr), 1200.0, sr)

        # Slightly quieter for "extra" hits so it doesn't get too busy
        hat_gain = 0.10 if is_hat else 0.07
        out[start:end] += hat_gain * hat * env

    out = np.tanh(out * 1.2).astype(np.float32)
    return out


def _quarter_note_boundaries(N, sr, bpm, quarters_per_change=2):
    """
    Returns sample indices where note changes should happen, aligned to true BPM:
      idx_k = round(k * quarters_per_change * spb * sr)

    This avoids cumulative drift because every index is computed from absolute time.
    """
    spb = 60.0 / float(bpm)
    sec = N / float(sr)
    total_quarters = sec / spb

    n_changes = int(np.ceil(total_quarters / float(quarters_per_change)))  # number of segments
    idx = np.round(np.arange(n_changes + 1) * quarters_per_change * spb * sr).astype(np.int64)

    idx[0] = 0
    idx = idx[idx <= N]
    if len(idx) == 0 or idx[-1] != N:
        idx = np.append(idx, N)

    # enforce monotonic increasing (handle rare duplicate rounding)
    idx = np.maximum.accumulate(idx)
    return idx

def _add_beat_accent(env, idx, sr, accent_ms=35.0, amount=0.35):
    """
    Adds a short decaying accent starting EXACTLY at each idx boundary.
    This makes pitch changes perceptually "hit" the beat.
    """
    N = len(env)
    L = int((accent_ms / 1000.0) * sr)
    L = max(4, L)
    decay = np.exp(-np.arange(L, dtype=np.float32) / (0.012 * sr)).astype(np.float32)  # ~12ms tau

    for a in idx[:-1]:
        a = int(a)
        b = min(N, a + L)
        env[a:b] += amount * decay[: b - a]

    return env

# ---------- Lead ---------- #

def _fat_lead_reverb(x, sr=SR, wet=0.55, pre_delay_ms=18.0, tone_hz=5200.0):
    """
    Big, lush lead reverb:
      - pre-delay for separation
      - Schroeder reverb (wet)
      - slight tone shaping on the reverb return (keeps it 'fat' not harsh)
    """
    x = np.asarray(x, dtype=np.float32)
    N = len(x)

    # Pre-delay (simple sample delay)
    D = int((pre_delay_ms / 1000.0) * sr)
    if D > 0:
        xd = np.zeros_like(x)
        if D < N:
            xd[D:] = x[:-D]
        else:
            xd[:] = 0.0
    else:
        xd = x

    # Reverb core (use your existing Schroeder)
    rv = _schroeder_reverb(xd, sr=sr, wet=1.0)  # 100% wet return

    # Tone the tail a bit: lowpass + remove low rumble
    rv = one_pole_lpf(rv, tone_hz, sr)
    rv = rv - one_pole_lpf(rv, 180.0, sr)

    # Wet/dry mix
    return (1.0 - wet) * x + wet * rv


def lead_synth(
    control,
    bpm=110,
    sr=SR,
    melody_notes=MAJOR_SCALE_C,
    change_every_beats=2,      # quarter-notes per change (2 = every 2 beats)
    accent_ms=35.0,            # transient accent length
    keep_highend=True,
):
    """
    Continuous ethereal lead with note changes that happen exactly on true quarter-note boundaries.

    - Pitch steps exactly at boundary samples (no pre-smoothing across boundary).
    - A short beat-synced accent makes the change feel locked on the beat.
    - Control modulates brightness + vibrato depth, but melody advances even if control is flat.
    """
    N = len(control)
    c = clamp01(control).astype(np.float32)

    if melody_notes is None or len(melody_notes) == 0:
        melody_notes = MAJOR_SCALE_C

    # --- boundaries on TRUE bpm quarter-note grid ---
    idx = _quarter_note_boundaries(N, sr, bpm, quarters_per_change=change_every_beats)

    # --- build piecewise-constant frequency schedule (exact steps at idx) ---
    f = np.zeros(N, dtype=np.float32)
    n_segments = len(idx) - 1
    for k in range(n_segments):
        a = int(idx[k])
        b = int(idx[k + 1])
        if b <= a:
            continue
        note = melody_notes[k % len(melody_notes)]
        f[a:b] = float(note_to_freq(note))

    # --- subtle vibrato (doesn't shift the beat boundaries) ---
    t = (np.arange(N, dtype=np.float32) / sr)
    vib_rate  = 0.25 + 0.10 * c
    vib_depth = 0.0008 + 0.0014 * c
    vib = vib_depth * np.sin(2.0 * np.pi * vib_rate * t).astype(np.float32)

    f_mod = f * (1.0 + vib)

    # --- stable phase accumulation (float64 + wrap) ---
    phase = 2.0 * np.pi * np.cumsum(f_mod.astype(np.float64)) / float(sr)
    phase = np.mod(phase, 2.0 * np.pi).astype(np.float32)

    # --- ethereal oscillator (bright, not dull) ---
    core = np.sin(phase).astype(np.float32)
    air1 = 0.38 * np.sin(2.0 * phase).astype(np.float32)
    air2 = 0.20 * np.sin(3.0 * phase).astype(np.float32)
    x = (core + air1 + air2).astype(np.float32)

    # --- amplitude: slow bloom + BEAT-ACCENT (critical for "on beat" feel) ---
    bloom = (0.75 + 0.25 * np.sin(2*np.pi*0.05*t).astype(np.float32))
    amp = (0.07 + 0.10 * c) * bloom

    accent_env = np.ones(N, dtype=np.float32)
    # Accent amount can be slightly control-dependent; flat control still accents.
    accent_amount = 0.35 + 0.15 * float(np.mean(c))
    accent_env = _add_beat_accent(accent_env, idx, sr, accent_ms=accent_ms, amount=accent_amount)

    y = x * amp * accent_env

    # --- keep more high-end ---
    if keep_highend:
        cutoff = (5500.0 + 12000.0 * c).astype(np.float32)
        cutoff = smooth_signal(cutoff, cutoff_hz=3.0, sr=sr)
        y = one_pole_lpf_varying(y, cutoff, sr)
        # little "air" HPF: y - LPF(y)
        y = y - one_pole_lpf(y, 180.0, sr)
    else:
        cutoff = (3500.0 + 9000.0 * c).astype(np.float32)
        cutoff = smooth_signal(cutoff, cutoff_hz=2.5, sr=sr)
        y = one_pole_lpf_varying(y, cutoff, sr)

    # --- librosa: sheen without timing smear ---
    y = librosa.effects.preemphasis(y.astype(np.float32), coef=0.97).astype(np.float32)

    # --- FAT reverb on lead ---
    y = _fat_lead_reverb(y, sr=sr, wet=0.8, pre_delay_ms=18.0, tone_hz=5200.0)


    # glue
    y = np.tanh(y * 1.10).astype(np.float32)
    return y

# ============================================================
# Plotting (updated: raw control w/ timestamps + day/night shading)
# ============================================================

from datetime import datetime, timedelta
from pytz import timezone
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


CHICAGO_TZ = timezone("America/Chicago")
LIGHT_ON  = (10, 0)  # 10:00 AM
LIGHT_OFF = (22, 0)  # 10:00 PM


def _title_for_signal(which: int) -> str:
    return "CAM Data" if int(which) == 1 else "C3-CAM"


def _to_py_datetimes_chicago(ts_arr, tz=CHICAGO_TZ):
    """
    Convert timestamps array into a list of Python datetime objects in Chicago TZ.

    Accepts:
      - numpy datetime64 array (tz-naive by nature)
      - Python datetime objects (naive or tz-aware)
      - Anything that can be np.asarray(...) into a 1D array of datetimes

    Behavior:
      - If timestamps are tz-naive, assume they are already in Chicago local time
        and localize them to CHICAGO_TZ.
      - If tz-aware, convert to CHICAGO_TZ.
    """
    ts = np.asarray(ts_arr)

    # numpy datetime64 -> python datetime (naive)
    if np.issubdtype(ts.dtype, np.datetime64):
        # Convert to microsecond resolution to avoid weird rounding
        ts_us = ts.astype("datetime64[us]")
        py = [d.astype(datetime) for d in ts_us]
    else:
        # Already python datetimes (or list-like)
        py = list(ts)

    out = []
    for d in py:
        if d is None:
            continue
        # If timezone-aware -> convert to Chicago
        if getattr(d, "tzinfo", None) is not None:
            out.append(d.astimezone(tz))
        else:
            # tz-naive -> assume Chicago local time, localize
            out.append(tz.localize(d))
    return out


def _shade_day_night(ax, t_min, t_max, tz=CHICAGO_TZ, light_on=LIGHT_ON, light_off=LIGHT_OFF):
    """
    Shade nights: before 10AM and after 10PM, per day, between t_min and t_max.
    t_min/t_max must be tz-aware datetimes in the same tz.
    """
    start_date = t_min.date()
    end_date   = t_max.date()

    current = start_date
    while current <= end_date:
        day_start = tz.localize(datetime.combine(current, datetime.min.time()))
        on  = day_start + timedelta(hours=light_on[0],  minutes=light_on[1])
        off = day_start + timedelta(hours=light_off[0], minutes=light_off[1])
        day_end = day_start + timedelta(days=1)

        # night: midnight -> 10AM
        ax.axvspan(day_start, on, alpha=0.30, color="darkgrey", zorder=0)
        # night: 10PM -> midnight
        ax.axvspan(off, day_end, alpha=0.30, color="darkgrey", zorder=0)

        current += timedelta(days=1)


def save_control_plot_raw_with_timestamps(
    control_raw: np.ndarray,
    control_time: np.ndarray,
    which: int,
    out_dir: str = ".",
    tz=CHICAGO_TZ,
    light_on=LIGHT_ON,
    light_off=LIGHT_OFF,
):
    """
    RAW plot requirements:
      1) Plot the raw data of the selected signal
      2) Title: "CAM Data" if sig1, "C3-CAM" if sig2
      3) X-axis uses timestamps that go with the raw data
      4) Shade nighttimes (after 10PM and before 10AM Chicago time)
    """
    y = np.asarray(control_raw, dtype=np.float32).squeeze()
    if y.ndim != 1:
        raise ValueError(f"control_raw must be 1D; got shape {y.shape}")

    t_list = _to_py_datetimes_chicago(control_time, tz=tz)
    if len(t_list) != len(y):
        raise ValueError(
            f"control_time length ({len(t_list)}) must match control_raw length ({len(y)})."
        )

    title = _title_for_signal(which)

    plt.figure(figsize=(12, 9))
    ax = plt.gca()

    # Shade nights first (behind data)
    t_min, t_max = min(t_list), max(t_list)
    _shade_day_night(ax, t_min, t_max, tz=tz, light_on=light_on, light_off=light_off)

    # Plot raw signal
    ax.plot(t_list, y, linewidth=1.0)

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Timestamp (Chicago)", fontsize=12)
    ax.set_ylabel("CO₂ (ppm, raw)", fontsize=12)
    ax.grid(True, alpha=0.30)

    # Nice datetime formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d, %I %p", tz=tz))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=-30, ha="left")

    # Tight x-limits
    ax.set_xlim(t_min, t_max)

    plt.tight_layout()

    png_name = f"control_sig{int(which)}_raw.png"
    png_path = f"{out_dir.rstrip('/')}/{png_name}"
    plt.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved control plot -> {png_path}")


# Optional: keep plot_audio as-is (unchanged), included here for completeness.
def plot_audio(x, sr):
    """
    Plot audio amplitude vs time (seconds).
    """
    x = np.asarray(x)
    t = np.arange(len(x)) / float(sr)

    plt.figure(figsize=(12, 4))
    plt.plot(t, x)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Audio waveform")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# WAV writer + render
# ============================================================
def write_wav(filename, x, sr=SR):
    """
    Writes mono 16-bit PCM wav. Normalizes once here (avoid double normalization).
    """
    x = np.asarray(x, dtype=np.float32)
    x = normalize_audio(x, peak=0.98)
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype(np.int16)

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())

def render_mix(control_signal, seconds=10, sr=SR, bpm=110, fade_out_ms=500.0):
    """
    Returns raw mix (no normalization here).
    Normalization happens in write_wav().

    fade_out_ms: end fade duration in milliseconds (default 200 ms).
    """

    # Get control sig
    N = int(seconds * sr)
    control = resample_linear(control_signal, N)
    control = safe_control(control)

    # Get instrumen sigs
    pad_x = pad(control, sr=sr, notes=PAD_NOTES)

    drum_x = rhythm(
    control, bpm=bpm, sr=sr,
    kick_beats=(0.0, 2.0, 4.0, 6.0, 7.5),
    hat_beats=(0.0,
               1.0,
               2.5, 
               3.5
               ),
)
    
    lead_x = lead_synth(control, bpm=bpm, sr=SR, melody_notes=LEAD_MELODY, change_every_beats=4)

    # Mix them together
    mix = 0.98 * pad_x + 0.14 * drum_x + 0.95 * lead_x  # adjust as desired
    mix = mix.astype(np.float32)

    # --- end fade to prevent file-tail pop ---
    mix = fade_out(mix, sr=sr, fade_ms=fade_out_ms)

    return mix


def write_wave(control_signal, filename="output_sound.wav", seconds=12, sr=SR, bpm=110):
    audio = render_mix(control_signal, seconds=seconds, sr=sr, bpm=bpm)
    write_wav(filename, audio, sr=sr)
    return audio

# ============================================================
# Loading CO2 Control Data
# ============================================================
def load_control_signal(path: str) -> np.ndarray:
    """
    Load a raw CO2 control signal OR timestamps from .npy OR .npz.

    - If the stored array is datetime64, it is returned as-is (NO float casting).
    - Otherwise, returns float32 1D array.

    .npy: expects a single array
    .npz: uses the only array if one exists, otherwise tries common keys
    """
    obj = np.load(path, allow_pickle=False)

    # .npy case -> ndarray
    if isinstance(obj, np.ndarray):
        x = obj
    else:
        # .npz case -> NpzFile
        data = obj
        if len(data.files) == 1:
            x = data[data.files[0]]
        else:
            for k in ("time", "timestamp", "timestamps", "t", "co2", "CO2", "sig", "signal", "x", "y", "values"):
                if k in data.files:
                    x = data[k]
                    break
            else:
                x = data[data.files[0]]

    x = np.asarray(x).squeeze()
    if x.ndim != 1:
        raise ValueError(f"Loaded array must be 1D after squeeze; got shape {x.shape} from {path}")

    # IMPORTANT: preserve datetime64 timestamps
    if np.issubdtype(x.dtype, np.datetime64):
        return x

    # numeric control signal
    return x.astype(np.float32)


def _signal_to_time(raw_sig: np.ndarray, seconds: float, sr: int) -> np.ndarray:
    """
    Convert an irregularly-sampled raw signal into a per-sample control vector
    of length N = seconds*sr by "zero-hold" (a.k.a. step interpolation).

    In other words:
      - We place each raw sample at evenly-spaced times across the duration
      - We "interpolate zeros" between them by holding the most recent sample value
        until the next sample time.

    This preserves the "stair-step" feel and avoids adding smoothing/interpolation
    artifacts. (We can add smoothing later if you want.)
    """
    raw_sig = np.asarray(raw_sig, dtype=np.float32).squeeze()
    if raw_sig.size == 0:
        raise ValueError("raw_sig is empty")

    N = int(round(seconds * sr))
    N = max(1, N)

    # If raw_sig already matches target length, return as-is
    if len(raw_sig) == N:
        return raw_sig.astype(np.float32)

    # Map raw samples evenly across the duration
    # idx_raw[i] gives the sample index in the audio timeline where raw_sig[i] "arrives"
    idx_raw = np.linspace(0, N - 1, num=len(raw_sig), dtype=np.int64)

    out = np.zeros(N, dtype=np.float32)

    # Place the samples
    out[idx_raw] = raw_sig

    # Zero-hold fill: carry forward last nonzero value
    # (works even if raw_sig contains actual zeros; it's still a step-hold)
    last = out[0]
    for i in range(1, N):
        if out[i] == 0.0:
            out[i] = last
        else:
            last = out[i]

    return out


# ============================================================
# Main 
# ============================================================
if __name__ == "__main__":

    # Choose which control to use: 1 or 2
    CONTROL_SELECT = 1

    # --- Render config ---
    OUT_WAV = f"output_sig{CONTROL_SELECT}.wav"
    BPM = 120
    SECONDS = 20
    SR = 44100  # keep consistent with your global SR

    # Control sigs (raw + time)
    CO2_SIG1_RAW  = "co2_sig1_raw.npy"
    CO2_SIG1_TIME = "co2_sig1_time.npy"
    CO2_SIG2_RAW  = "co2_sig2_raw.npy"
    CO2_SIG2_TIME = "co2_sig2_time.npy"

    # --- Choose which raw control file + time file to use ---
    raw_file  = CO2_SIG1_RAW  if CONTROL_SELECT == 1 else CO2_SIG2_RAW
    time_file = CO2_SIG1_TIME if CONTROL_SELECT == 1 else CO2_SIG2_TIME
    which = CONTROL_SELECT
    print(f"Using control files: raw={raw_file} | time={time_file}")

    # 1) Load raw CO2 samples (irregular / arbitrary length)
    raw_control = load_control_signal(raw_file)
    raw_time    = load_control_signal(time_file)  # timestamps saved as numpy datetime64 in .npy
    print(f"Loaded raw control length: {len(raw_control)} samples")
    print(f"Loaded raw time length:    {len(raw_time)} timestamps")

    # --- Plot + save RAW control (with timestamps + day/night shading) ---
    """save_control_plot_raw_with_timestamps(
        control_raw=raw_control,
        control_time=raw_time,
        which=which,
        out_dir=".",
    )"""

    # 2) Convert raw control into per-audio-sample control (step/zero-hold)
    control = _signal_to_time(raw_control, seconds=SECONDS, sr=SR)

    # --- Plot + save TIME-ALIGNED control (before normalization) ---
    t = np.arange(len(control), dtype=np.float32) / float(SR)
    plt.figure(figsize=(12, 4))
    plt.plot(t, control)
    plt.xlabel("Time (s)")
    plt.ylabel("CO₂ control (time-aligned, pre-norm)")
    plt.title(f"Control signal {which} (time-aligned, pre-norm)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    png_time = f"control_sig{which}_time_aligned_pre_norm.png"
    plt.savefig(png_time, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved time-aligned (pre-norm) control plot -> {png_time}")

    # 3) Normalize to 0..1 safely (your pipeline expects this)
    control = safe_control(control)

    # --- Plot + save NORMALIZED control (what instruments actually see) ---
    t = np.arange(len(control), dtype=np.float32) / float(SR)
    plt.figure(figsize=(12, 4))
    plt.plot(t, control)
    plt.xlabel("Time (s)")
    plt.ylabel("Control (0..1)")
    plt.title(f"Control signal {which} (normalized 0..1)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    png_norm = f"control_sig{which}_normalized.png"
    plt.savefig(png_norm, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved normalized control plot -> {png_norm}")

    # 4) Render soundtrack using chosen control
    audio = write_wave(control, filename=OUT_WAV, seconds=SECONDS, sr=SR, bpm=BPM)

    print(f"Wrote {OUT_WAV}")
    print("PAD_NOTES =", PAD_NOTES, "->", [note_to_freq(n) for n in PAD_NOTES])
    print("LEAD_MELODY =", LEAD_MELODY, "->", [note_to_freq(n) for n in LEAD_MELODY])
