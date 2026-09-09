import io
import os
from typing import List, Optional
import hashlib

import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Query, Form
from fastapi.responses import JSONResponse
from scipy.signal import butter, filtfilt, medfilt
from pydantic import BaseModel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# pip install fastapi uvicorn speechbrain soundfile librosa torch
from speechbrain.inference.speaker import EncoderClassifier
import librosa

# -----------------------------------------------------------------------------
# Server overview
# -----------------------------------------------------------------------------
# This FastAPI service powers speaker verification and visualization used by the
# iOS app. It provides:
#   - Embedding endpoints (single-shot and mean-over-windows)
#   - Cosine scoring (single pair, slice, and batch of slices)
#   - Sliding-window segmentation with hop-to-voice anchoring
#   - Score-driven head/tail drill-down to tighten segment boundaries
#   - Plotting endpoint that renders visuals from saved decisions
#
# Core pipeline concepts:
#   - Enrollment (train) embedding: built once per request (cached), optional
#     preprocessing, and optional VAD-based silence trimming.
#   - Test segmentation: use librosa VAD to find voiced "islands", align window
#     starts to onsets (optional pre-roll), step fixed hops within islands, score
#     each window, and merge consecutive accepts into segments.
#   - Refinement: iteratively trim segment tails/heads when score/margin improve.
#
# Performance:
#   - In-process LRU caches prevent re-embedding the train audio and re-loading
#     the same test WAV across calls.
#   - Batch slice scoring minimizes app<->server round-trips.
# -----------------------------------------------------------------------------

SR = 16000
WIN_SEC = 3.0
STRIDE_SEC = 1.0

app = FastAPI()


class EmbedResponse(BaseModel):
    embedding: List[float]


class ScoreResponse(BaseModel):
    score: float


class ScoreDetailed(BaseModel):
    score: float
    duration: float
    threshold: float
    margin: float
    accepted: bool

# Simple in-process caches to avoid recomputing train embeddings and test wave loads
_CACHE_MAX = 8
_cache_train: dict[tuple[str, str], np.ndarray] = {}  # key=(sha1(train_bytes), pp)
_cache_test_wave: dict[str, np.ndarray] = {}          # key=sha1(test_bytes)

def _sha1(b: bytes) -> str:
    """SHA-1 helper used to key caches by content."""
    return hashlib.sha1(b).hexdigest()

def _cache_put(d: dict, k, v):
    """Insert/move-most-recent and evict oldest if capacity exceeded."""
    if k in d:
        d.pop(k, None)
    d[k] = v
    # Evict oldest if over capacity
    while len(d) > _CACHE_MAX:
        try:
            d.pop(next(iter(d)))
        except Exception:
            break

class SegmentsResponse(BaseModel):
    # list of [start_sec, end_sec]
    segments: List[List[float]]


class DecisionItem(BaseModel):
    index: int
    start: float
    end: float
    duration: float
    score: float
    threshold: float
    accepted: bool

# Sliding segmentation tuning
# Defaults can be overridden via environment variables
SLIDING_WIN_SEC = float(os.getenv("SLIDING_WIN_SEC", "1.5"))       # Window size (seconds) for per-window scoring
SLIDING_HOP_SEC = float(os.getenv("SLIDING_HOP_SEC", "1.0"))       # Hop/stride (seconds) between windows; sets overlap
SLIDING_HEAD_TRIM_SEC = float(os.getenv("SLIDING_HEAD_TRIM_SEC", "0.0"))  # Pre-Scorer: optional head trim (applied client-side)
SLIDING_TAIL_TRIM_SEC = float(os.getenv("SLIDING_TAIL_TRIM_SEC", "0.0"))  # Post-Scorer: subtract small tail in seconds

SLIDING_MIN_KEEP_SEC = float(os.getenv("SLIDING_MIN_KEEP_SEC", "0.4"))  # Drop merged segments shorter than this duration
SLIDING_TOP_DB = float(os.getenv("SLIDING_TOP_DB", "25.0"))        # VAD threshold (dB) for silence trimming; higher trims more
SLIDING_PP = os.getenv("SLIDING_PP", "hpf_norm")      # Preprocess preset for train/windows (e.g., none|norm|hpf_norm)
SLIDING_END_HOP_OFFSET = int(os.getenv("SLIDING_END_HOP_OFFSET", "1"))  # 1 = last accepted hop; 0 = include next hop

# HOP-TO-VOICE ANCHORING
SLIDING_ALIGN_TO_VOICE = int(os.getenv("SLIDING_ALIGN_TO_VOICE", "1"))   # If 1, anchor hop starts to voiced onsets
SLIDING_PRE_ROLL_SEC = float(os.getenv("SLIDING_PRE_ROLL_SEC", "0.0"))  # Start this much before onset (s)
SLIDING_MIN_SILENCE_SEC = float(os.getenv("SLIDING_MIN_SILENCE_SEC", "0.1"))  # Merge gaps shorter than this (s)
SLIDING_MIN_ISLAND_SEC = float(os.getenv("SLIDING_MIN_ISLAND_SEC", "0.02"))    # Drop voiced islands shorter than this (s)

# Score-driven tail drill-down (post-merge, pre-return)
SLIDING_TAIL_DRILL_STEP_SEC = float(os.getenv("SLIDING_TAIL_DRILL_STEP_SEC", "0.25"))  # step size per iteration
SLIDING_TAIL_DRILL_MIN_LEN_SEC = float(os.getenv("SLIDING_TAIL_DRILL_MIN_LEN_SEC", "0.6"))  # minimum segment length to allow
SLIDING_TAIL_DRILL_MAX_SEC = float(os.getenv("SLIDING_TAIL_DRILL_MAX_SEC", "0.8"))  # cap on total tail trim
SLIDING_TAIL_DRILL_DEBUG = int(os.getenv("SLIDING_TAIL_DRILL_DEBUG", "1"))  # 1=log drill-down details
SLIDING_TAIL_DRILL_EPS = float(os.getenv("SLIDING_TAIL_DRILL_EPS", "0.000"))  # allow tiny margin drop
SLIDING_TAIL_DRILL_VWIN_SEC = float(os.getenv("SLIDING_TAIL_DRILL_VWIN_SEC", "3.0"))  # virtual window length for margin/score. Uses only last x seconds of clip for tail drill calcs
SLIDING_TAIL_DRILL_DELTA_SM_MIN = float(os.getenv("SLIDING_TAIL_DRILL_DELTA_SM_MIN", "0.010"))  # keep cut if (Δscore + Δmargin) >= this
SLIDING_HEAD_DRILL_DELTA_SM_MIN = float(os.getenv("SLIDING_HEAD_DRILL_DELTA_SM_MIN", "0.010")) 

# Train-time silence trimming (to align enrollment with test behavior)
SLIDING_TRIM_TRAIN = int(os.getenv("SLIDING_TRIM_TRAIN", "1"))  # 1=trim training audio with VAD before embedding
SLIDING_TRAIN_TOP_DB = float(os.getenv("SLIDING_TRAIN_TOP_DB", str(SLIDING_TOP_DB)))  # VAD threshold for training trim


 


device = "cuda" if torch.cuda.is_available() else "cpu"
model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
model.to(device)


def load_resample_mono_bytes(data: bytes, target_sr: int = SR) -> np.ndarray:
    """Load WAV bytes, convert to mono if needed, and resample to target_sr."""
    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav


def trim_silence(wav: np.ndarray, top_db: float = 25) -> np.ndarray:
    """Concatenate all voiced intervals determined by librosa.effects.split."""
    intervals = librosa.effects.split(wav, top_db=top_db)
    if len(intervals) == 0:
        return wav
    pieces = [wav[s:e] for s, e in intervals]
    return np.concatenate(pieces) if pieces else wav


## (rolled back VAD cleanup helper)


def chunk_windows(wav: np.ndarray, sr: int = SR, win_sec: float = WIN_SEC, stride_sec: float = STRIDE_SEC) -> list:
    """Return fixed-size windows with a fixed stride across the waveform."""
    win = int(win_sec * sr)
    stride = int(stride_sec * sr)
    if len(wav) < win:
        pad = np.zeros(win - len(wav), dtype=wav.dtype)
        wav = np.concatenate([wav, pad])
    starts = np.arange(0, max(1, len(wav) - win + 1), stride, dtype=int)
    return [wav[s:s + win] for s in starts]


def embed_wave(w: np.ndarray) -> np.ndarray:
    """Encode a waveform into a speaker embedding using SpeechBrain ECAPA."""
    t = torch.tensor(w, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        emb = model.encode_batch(t).squeeze().detach().cpu().numpy()
    return emb


def embed_mean_from_wave(wav: np.ndarray, pp: str = "none") -> np.ndarray:
    """Mirror /embed_mean behavior for an in-memory waveform.
    Applies optional preprocessing, trims silence, windows, embeds each window,
    and returns the mean embedding.
    """
    x = wav
    if pp != "none":
        try:
            x = apply_preprocess(x, pp)
        except Exception:
            pass
    x = trim_silence(x, top_db=25)
    windows = chunk_windows(x, SR, WIN_SEC, STRIDE_SEC)
    embs = [embed_wave(w) for w in windows]
    return np.mean(np.stack(embs, axis=0), axis=0)


def embed_single_from_wave(wav: np.ndarray, pp: str = "none") -> np.ndarray:
    """Single-shot embedding with optional preprocessing; no silence trim, no windowing."""
    x = wav
    if pp != "none":
        try:
            x = apply_preprocess(x, pp)
        except Exception:
            pass
    emb = embed_wave(x)
    return emb


 


def normalize_rms(wav: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
    """Normalize RMS level to target_dbfs and hard-clip to [-1, 1]."""
    if wav.size == 0:
        return wav
    rms = float(np.sqrt(np.mean(np.square(wav), dtype=np.float64)))
    if rms <= 1e-8:
        return wav
    target = 10.0 ** (target_dbfs / 20.0)  # e.g., -20 dBFS => ~0.1
    g = target / rms
    y = wav * g
    # Hard clip to [-1, 1]
    np.clip(y, -1.0, 1.0, out=y)
    return y.astype(np.float32, copy=False)


 


def highpass_butter(wav: np.ndarray, sr: int, cutoff: float = 100.0, order: int = 2) -> np.ndarray:
    """Apply a Butterworth high-pass filter to the waveform."""
    if wav.size == 0:
        return wav
    nyq = 0.5 * float(sr)
    wn = float(cutoff) / nyq
    wn = min(max(wn, 1e-4), 0.99)
    b, a = butter(order, wn, btype="highpass")
    y = filtfilt(b, a, wav, method="gust")
    return y.astype(np.float32, copy=False)


def normalize_lufs(wav: np.ndarray, sr: int, target_lufs: float = -20.0, true_peak_db: float = -1.0) -> np.ndarray:
    """Integrated loudness normalization (BS.1770 via pyloudnorm). Falls back to RMS if unavailable.
    Order: measure -> apply gain -> clip to true-peak ceiling.
    """
    try:
        import pyloudnorm as pyln  # type: ignore
    except Exception:
        return normalize_rms(wav, target_dbfs=target_lufs)
    if wav.size == 0:
        return wav
    meter = pyln.Meter(sr)
    try:
        loud = float(meter.integrated_loudness(wav.astype(np.float32)))
    except Exception:
        return normalize_rms(wav, target_dbfs=target_lufs)
    gain_db = target_lufs - loud
    gain = 10.0 ** (gain_db / 20.0)
    y = wav * gain
    # True-peak limit (approximate hard clip at -1 dBFS)
    tp_amp = 10.0 ** (true_peak_db / 20.0)
    np.clip(y, -tp_amp, tp_amp, out=y)
    return y.astype(np.float32, copy=False)


def apply_preprocess(wav: np.ndarray, mode: str) -> np.ndarray:
    """Run the configured preprocessing chain (HPF + normalization variants)."""
    m = (mode or "none").lower()
    if m == "none":
        return wav
    if m == "norm":
        return normalize_rms(wav)
    if m in ("hpf_norm", "hpfnorm"):
        # default 100 Hz HPF + RMS normalize
        return normalize_rms(highpass_butter(wav, SR, cutoff=100.0))
    if m == "hpf60_norm":
        # 60 Hz HPF + RMS normalize
        return normalize_rms(highpass_butter(wav, SR, cutoff=60.0))
    if m == "hpf400_norm":
        # 400 Hz HPF + RMS normalize
        return normalize_rms(highpass_butter(wav, SR, cutoff=400.0))
    if m in ("lufs_hpf", "hpf_lufs"):
        # HPF precedes LUFS (compressor removed)
        y = highpass_butter(wav, SR)
        return normalize_lufs(y, SR)
    return wav

def duration_threshold(d: float) -> float:
    """Duration-dependent cosine threshold (smooth saturating curve)."""
    t_min = 0.06
    t_max = 0.55
    tau = 2.8
    d = max(0.0, float(d))
    return float(t_min + (t_max - t_min) * (1.0 - np.exp(-d / tau)))

 

def embed_mean_bytes(data: bytes, pp: str = "none") -> np.ndarray:
    wav = load_resample_mono_bytes(data, SR)
    rms_in = float(np.sqrt(np.mean(np.square(wav), dtype=np.float64))) if wav.size > 0 else 0.0
    wav = apply_preprocess(wav, pp)
    rms_out = float(np.sqrt(np.mean(np.square(wav), dtype=np.float64))) if wav.size > 0 else 0.0
    try:
        print(f"pp={pp} rms_in={rms_in:.4f} rms_out={rms_out:.4f}")
    except Exception:
        pass
    wav = trim_silence(wav, top_db=25)
    windows = chunk_windows(wav, SR, WIN_SEC, STRIDE_SEC)
    embs = [embed_wave(w) for w in windows]
    return np.mean(np.stack(embs, axis=0), axis=0)


def embed_single_bytes(data: bytes, pp: str = "none") -> np.ndarray:
    wav = load_resample_mono_bytes(data, SR)
    # Optional logging of RMS change for parity with embed_mean_bytes
    rms_in = float(np.sqrt(np.mean(np.square(wav), dtype=np.float64))) if wav.size > 0 else 0.0
    if pp != "none":
        wav = apply_preprocess(wav, pp)
    rms_out = float(np.sqrt(np.mean(np.square(wav), dtype=np.float64))) if wav.size > 0 else 0.0
    try:
        print(f"embed_single pp={pp} rms_in={rms_in:.4f} rms_out={rms_out:.4f}")
    except Exception:
        pass
    return embed_wave(wav)


@app.get("/health")
def health():
    """Health probe for connectivity checks."""
    return {"ok": True}


def compute_sliding_segments(
    x_train: np.ndarray,
    x_test: np.ndarray,
    win_sec: float,
    hop_sec: float,
    min_keep: float,
    top_db: float,
    pp: str,
) -> list[list[float]]:
    """Compute sliding-window verification segments for a train/test pair.
    Steps:
      1) Enrollment: preprocess (optional) and optionally VAD-trim the train audio, then embed and L2-normalize.
      2) VAD test: librosa.effects.split finds voiced intervals ("islands"), with optional gap merge and micro-island drop.
      3) Anchor and step: for each island, align the first hop to onset (with optional pre-roll) and step hops within the island.
      4) Score and merge: score windows vs enrollment, merge consecutive accepts into segments, enforce min_keep.
      5) Refine: run tail and head score-driven drill-down using margin-aware criterion and optional virtual window.
    """
    # Training embedding: slider pipeline (optional pp, single-shot embed)
    if pp != "none":
        try:
            x_train = apply_preprocess(x_train, pp)
        except Exception:
            pass
    # Optionally trim training silence (VAD) before embedding
    if SLIDING_TRIM_TRAIN:
        try:
            dur_b = len(x_train) / SR
        except Exception:
            dur_b = 0.0
        x_train = trim_silence(x_train, top_db=float(SLIDING_TRAIN_TOP_DB))
        try:
            dur_a = len(x_train) / SR
            print(f"train_trim slider pp={pp} dur_before={dur_b:.3f}s dur_after={dur_a:.3f}s top_db={SLIDING_TRAIN_TOP_DB}")
        except Exception:
            pass
    emb_train = embed_wave(x_train)
    emb_train = emb_train / (np.linalg.norm(emb_train) + 1e-12)

    # VAD on test and optional interval conditioning
    raw_voiced = librosa.effects.split(x_test, top_db=float(top_db))
    if len(raw_voiced) == 0:
        return []
    # Merge short silences and drop micro voiced islands for anchoring
    voiced: list[tuple[int,int]] = []
    cur_s, cur_e = int(raw_voiced[0][0]), int(raw_voiced[0][1])
    min_gap = int(max(0.0, SLIDING_MIN_SILENCE_SEC) * SR)
    min_island = int(max(0.0, SLIDING_MIN_ISLAND_SEC) * SR)
    for (s, e) in raw_voiced[1:]:
        s = int(s); e = int(e)
        if s - cur_e <= min_gap:
            cur_e = max(cur_e, e)
        else:
            if cur_e - cur_s >= max(1, min_island):
                voiced.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    if cur_e - cur_s >= max(1, min_island):
        voiced.append((cur_s, cur_e))

    win = int(max(1, win_sec * SR))
    hop = int(max(1, hop_sec * SR))
    pre_roll = int(max(0.0, SLIDING_PRE_ROLL_SEC) * SR) if SLIDING_ALIGN_TO_VOICE else 0
    n = len(x_test)
    # Build segments directly using onset anchoring and hop stepping within islands
    segs: list[list[float]] = []
    for (vs, ve) in voiced:
        s = max(0, vs - pre_roll)
        in_run = False
        run_start_time_sec = 0.0
        last_accept_start = s
        last_accept_len = 0
        while s < ve and s < n:
            e = s + win
            if e > n:
                e = n
            # No tail refinement: score full window
            e_eff = e
            w = x_test[s:e_eff]
            if pp != "none":
                try:
                    w = apply_preprocess(w, pp)
                except Exception:
                    pass
            emb_w = embed_wave(w)
            emb_w = emb_w / (np.linalg.norm(emb_w) + 1e-12)
            sc = float(np.dot(emb_w, emb_train))
            thr = duration_threshold(max(1e-6, (e_eff - s) / SR))
            if sc >= thr:
                if not in_run:
                    in_run = True
                    run_start_time_sec = max(0.0, s / SR)
                last_accept_start = s
                last_accept_len = max(0, e_eff - s)
            else:
                if in_run:
                    end_time_sec = min(n, last_accept_start + last_accept_len) / SR
                    if end_time_sec - run_start_time_sec >= min_keep:
                        if segs and run_start_time_sec <= segs[-1][1] + 1e-6:
                            segs[-1][1] = max(segs[-1][1], end_time_sec)
                        else:
                            segs.append([float(run_start_time_sec), float(end_time_sec)])
                    in_run = False
            s += hop
        if in_run:
            end_time_sec = min(n, last_accept_start + last_accept_len) / SR
            if end_time_sec - run_start_time_sec >= min_keep:
                if segs and run_start_time_sec <= segs[-1][1] + 1e-6:
                    segs[-1][1] = max(segs[-1][1], end_time_sec)
                else:
                    segs.append([float(run_start_time_sec), float(end_time_sec)])
            in_run = False
    # Score-driven tail drill-down per segment (keep start; shorten end if score stays >= best)
    step = int(max(1, SLIDING_TAIL_DRILL_STEP_SEC * SR))
    min_len = int(max(1, SLIDING_TAIL_DRILL_MIN_LEN_SEC * SR))
    max_trim = int(max(0.0, SLIDING_TAIL_DRILL_MAX_SEC) * SR)
    if step > 0 and max_trim > 0 and len(segs) > 0:
        refined: list[list[float]] = []
        # Helper to score a slice with same pipeline as windows
        def score_slice(s0: int, e0: int) -> float:
            if e0 <= s0:
                return -1.0
            ww = x_test[s0:e0]
            if pp != "none":
                try:
                    ww = apply_preprocess(ww, pp)
                except Exception:
                    pass
            ew = embed_wave(ww)
            ew = ew / (np.linalg.norm(ew) + 1e-12)
            return float(np.dot(ew, emb_train))
        for ss, ee in segs:
            sS = int(max(0.0, ss) * SR)
            eS_initial = int(min(ee, n / SR) * SR)
            eS = eS_initial
            if eS - sS < min_len:
                refined.append([float(ss), float(ee)])
                continue
            # Virtual start anchored to original tail end minus vwin, capped by true start
            vwin = int(max(0.0, SLIDING_TAIL_DRILL_VWIN_SEC) * SR)
            vS = sS if (eS_initial - sS) <= vwin else max(sS, eS_initial - vwin)
            best_eS = eS
            best_score = score_slice(vS, eS)
            best_thr = duration_threshold((best_eS - vS) / SR)
            best_margin = best_score - best_thr
            if SLIDING_TAIL_DRILL_DEBUG:
                try:
                    print(f"tail-drill start: s={ss:.3f} e={ee:.3f} vS={(vS/SR):.3f} len={((best_eS - vS)/SR):.3f} score={best_score:.3f} thr={best_thr:.3f} margin={best_margin:.3f}")
                except Exception:
                    pass
            trimmed = 0
            while True:
                new_eS = max(vS + min_len, eS - step)
                if new_eS >= eS:
                    break
                if trimmed + (eS - new_eS) > max_trim:
                    break
                new_score = score_slice(vS, new_eS)
                new_thr = duration_threshold((new_eS - vS) / SR)
                new_margin = new_score - new_thr
                # Combined change criterion: (Δscore + Δmargin) must exceed configured minimum
                delta_score = new_score - best_score
                delta_margin = new_margin - best_margin
                combined = delta_score + delta_margin
                if combined >= SLIDING_TAIL_DRILL_DELTA_SM_MIN:
                    best_score = new_score
                    best_thr = new_thr
                    best_margin = new_margin
                    eS = new_eS
                    best_eS = new_eS
                    trimmed += step
                    if SLIDING_TAIL_DRILL_DEBUG:
                        try:
                            print(f"tail-drill keep:  new_e={best_eS/SR:.3f} len={((best_eS - vS)/SR):.3f} score={best_score:.3f} thr={best_thr:.3f} margin={best_margin:.3f} dS={delta_score:.3f} dM={delta_margin:.3f} sum={combined:.3f}")
                        except Exception:
                            pass
                    continue
                else:
                    if SLIDING_TAIL_DRILL_DEBUG:
                        try:
                            print(f"tail-drill stop: new_e={new_eS/SR:.3f} len={((new_eS - vS)/SR):.3f} score={new_score:.3f} thr={new_thr:.3f} margin={new_margin:.3f} dS={delta_score:.3f} dM={delta_margin:.3f} sum={combined:.3f}")
                        except Exception:
                            pass
                    break
            new_end_sec = best_eS / SR
            if new_end_sec - ss >= min_keep:
                refined.append([float(ss), float(new_end_sec)])
                if SLIDING_TAIL_DRILL_DEBUG:
                    try:
                        print(f"tail-drill final: s={ss:.3f} e={ee:.3f} vS={(vS/SR):.3f} len={(new_end_sec - (vS/SR)):.3f}")
                    except Exception:
                        pass
            # If too short after refinement, drop it (consistent with min_keep)
        segs = refined
    # Head drill-down pass (mirror logic), reusing same parameters
    if step > 0 and max_trim > 0 and len(segs) > 0:
        refined2: list[list[float]] = []
        for ss, ee in segs:
            sS_initial = int(max(0.0, ss) * SR)
            eS = int(min(ee, n / SR) * SR)
            sS = sS_initial
            if eS - sS < min_len:
                refined2.append([float(ss), float(ee)])
                continue
            vwin = int(max(0.0, SLIDING_TAIL_DRILL_VWIN_SEC) * SR)
            vE = eS if (eS - sS_initial) <= vwin else min(eS, sS_initial + vwin)
            best_sS = sS
            # score slice using [best_sS, vE]
            best_score = score_slice(best_sS, vE)
            best_thr = duration_threshold((vE - best_sS) / SR)
            best_margin = best_score - best_thr
            if SLIDING_TAIL_DRILL_DEBUG:
                try:
                    print(f"head-drill start: s={ss:.3f} e={ee:.3f} vE={(vE/SR):.3f} len={((vE - best_sS)/SR):.3f} score={best_score:.3f} thr={best_thr:.3f} margin={best_margin:.3f}")
                except Exception:
                    pass
            trimmed = 0
            while True:
                new_sS = min(vE - min_len, sS + step)
                if new_sS <= sS:
                    break
                if trimmed + (new_sS - sS) > max_trim:
                    break
                new_score = score_slice(new_sS, vE)
                new_thr = duration_threshold((vE - new_sS) / SR)
                new_margin = new_score - new_thr
                delta_score = new_score - best_score
                delta_margin = new_margin - best_margin
                combined = delta_score + delta_margin
                if combined >= SLIDING_HEAD_DRILL_DELTA_SM_MIN:
                    best_score = new_score
                    best_thr = new_thr
                    best_margin = new_margin
                    sS = new_sS
                    best_sS = new_sS
                    trimmed += step
                    if SLIDING_TAIL_DRILL_DEBUG:
                        try:
                            print(f"head-drill keep:  new_s={best_sS/SR:.3f} len={((vE - best_sS)/SR):.3f} score={best_score:.3f} thr={best_thr:.3f} margin={best_margin:.3f} dS={delta_score:.3f} dM={delta_margin:.3f} sum={combined:.3f}")
                        except Exception:
                            pass
                    continue
                else:
                    if SLIDING_TAIL_DRILL_DEBUG:
                        try:
                            print(f"head-drill stop: new_s={new_sS/SR:.3f} len={((vE - new_sS)/SR):.3f} score={new_score:.3f} thr={new_thr:.3f} margin={new_margin:.3f} dS={delta_score:.3f} dM={delta_margin:.3f} sum={combined:.3f}")
                        except Exception:
                            pass
                    break
            new_start_sec = best_sS / SR
            if ee - new_start_sec >= min_keep:
                refined2.append([float(new_start_sec), float(ee)])
                if SLIDING_TAIL_DRILL_DEBUG:
                    try:
                        print(f"head-drill final: s_final={new_start_sec:.3f} e={ee:.3f} vE={(vE/SR):.3f} len={( (vE/SR) - new_start_sec ):.3f}")
                    except Exception:
                        pass
        segs = refined2
    return segs


# (removed compute_sliding_segments_from_emb during rollback)


@app.get("/sliding_config")
def sliding_config():
    """Expose current sliding parameters and refinement toggles used by the app."""
    return {
        "win_sec": SLIDING_WIN_SEC,
        "hop_sec": SLIDING_HOP_SEC,
        "min_keep_sec": SLIDING_MIN_KEEP_SEC,
        "top_db": SLIDING_TOP_DB,
        "pp": SLIDING_PP,
        "end_hop_offset": SLIDING_END_HOP_OFFSET,
        "tail_trim_sec": SLIDING_TAIL_TRIM_SEC,
        "head_trim_sec": SLIDING_HEAD_TRIM_SEC,
        "align_to_voice": SLIDING_ALIGN_TO_VOICE,
        "pre_roll_sec": SLIDING_PRE_ROLL_SEC,
        "min_silence_sec": SLIDING_MIN_SILENCE_SEC,
        "min_island_sec": SLIDING_MIN_ISLAND_SEC,
        "tail_drill_step_sec": SLIDING_TAIL_DRILL_STEP_SEC,
        "tail_drill_min_len_sec": SLIDING_TAIL_DRILL_MIN_LEN_SEC,
        "tail_drill_max_sec": SLIDING_TAIL_DRILL_MAX_SEC,
        "tail_drill_eps": SLIDING_TAIL_DRILL_EPS,
        "tail_drill_vwin_sec": SLIDING_TAIL_DRILL_VWIN_SEC,
        "trim_train": SLIDING_TRIM_TRAIN,
        "train_top_db": SLIDING_TRAIN_TOP_DB,
    }


@app.post("/plot_from_decisions")
async def plot_from_decisions(
    test: UploadFile = File(...),
    decisions: str = Form(..., description="JSON array of DecisionItem from app"),
    meta: str | None = Form(None, description="Optional JSON of meta like mode/pp/head/tail"),
):
    """Render a plot directly from the app's saved decisions. No ML is performed here.
    Inputs:
      - test: WAV file
      - decisions: JSON array of objects with fields: index,start,end,duration,score,threshold,accepted
      - meta: optional JSON (e.g., {"mode":"sliding","pp":"hpf_norm","head_trim_sec":0.4,"tail_trim_sec":0.6})
    """
    import json as _json
    # Load audio
    test_bytes = await test.read()
    x_test = load_resample_mono_bytes(test_bytes, SR)
    total_sec = len(x_test) / SR
    # Parse decisions
    try:
        raw = _json.loads(decisions)
        items = [DecisionItem(**d) for d in raw]
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=f"Invalid decisions JSON: {e}")
    # Optional meta
    meta_text = ""
    if meta:
        try:
            m = _json.loads(meta)
            # Footer: general line + optional tail drill-down line
            gen_keys = ("mode","pp","win_sec","hop_sec","min_keep_sec","top_db")
            gen_parts = [f"{k}={m[k]}" for k in gen_keys if k in m]
            line1 = "  ".join(gen_parts)
            drill_map = [
                ("tail_drill_step_sec","Step_Sec"),
                ("tail_drill_min_len_sec","Min_Len_Sec"),
                ("tail_drill_max_sec","Max_Sec"),
                ("tail_drill_eps","Eps"),
                ("tail_drill_vwin_sec","Vwin_Sec"),
            ]
            drill_parts: list[str] = []
            for key, label in drill_map:
                if key in m:
                    drill_parts.append(f"{label}={m[key]}")
            if drill_parts:
                meta_text = line1 + "\n" + "Sliding Tail Drill Down: " + "  ".join(drill_parts)
            else:
                meta_text = line1
        except Exception:
            meta_text = meta[:120] + ("..." if len(meta) > 120 else "")

    # Draw
    t = np.arange(len(x_test)) / SR
    fig = plt.figure(figsize=(12, 3.6))
    ax = plt.gca()
    ax.plot(t, x_test, linewidth=0.3, color="#333")
    # Sort by start and render
    items.sort(key=lambda d: d.start)
    # Create a little headroom so labels don't collide with the plot title/top
    ymin, ymax = ax.get_ylim()
    if ymax - ymin > 0:
        headroom = 0.12 * (ymax - ymin)
        ax.set_ylim(ymin - 0.02 * (ymax - ymin), ymax + headroom)
    # Render segments with score/threshold labels
    for d in items:
        s = max(0.0, float(d.start))
        e = max(s, min(float(d.end), total_sec))
        if e - s <= 1e-3:
            continue
        color = "#4CAF50" if bool(d.accepted) else "#F44336"
        ax.axvspan(s, e, color=color, alpha=0.25, linewidth=0)
        ax.axvline(s, color="black", linewidth=0.5, alpha=0.7)
        ax.axvline(e, color="black", linewidth=0.5, alpha=0.7)
        mid = 0.5 * (s + e)
        yb, yt = ax.get_ylim()
        y_text = yt - 0.08 * (yt - yb)
        ax.text(mid, y_text, f"{float(d.score):.3f}/{float(d.threshold):.3f}",
                ha='center', va='center', fontsize=9, fontweight='semibold', zorder=5,
                bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1.0))
    ax.set_xlabel("Seconds")
    ax.set_title(test.filename if hasattr(test, 'filename') else "test.wav")
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    # Remove vertical grid to keep segment score labels clear
    ax.grid(False)
    # Reduce tick label padding to avoid footer collisions
    ax.tick_params(axis='x', pad=1)
    # Restore footer with meta only (no source tag)
    if meta_text:
        plt.figtext(0.01, 0.01, meta_text, ha='left', va='bottom', fontsize=8, family='monospace')
    # Increase bottom margin so footer doesn't overlap tick labels
    pad = 0.18 if ("\n" in meta_text) else 0.12
    plt.tight_layout(rect=[0, pad, 1, 1])
    import io as _io
    buf = _io.BytesIO(); fig.savefig(buf, format='png', dpi=150); plt.close(fig); buf.seek(0)
    from fastapi.responses import StreamingResponse
    return StreamingResponse(buf, media_type="image/png")

@app.post("/embed_mean", response_model=EmbedResponse)
async def embed_mean(wav: UploadFile = File(...), pp: str = Query("none", description="Preprocess: none|norm|hpf_norm|hpf60_norm|hpf400_norm|lufs_hpf")):
    """Return an embedding that is the mean of per-window embeddings after optional preprocess and silence trim."""
    data = await wav.read()
    emb = embed_mean_bytes(data, pp=pp)
    return EmbedResponse(embedding=emb.astype(float).tolist())


@app.post("/embed_single", response_model=EmbedResponse)
async def embed_single(wav: UploadFile = File(...), pp: str = Query("none", description="Preprocess for single-shot: none|norm|hpf_norm|hpf60_norm|hpf400_norm|lufs_hpf")):
    """Return a single-shot embedding of the full input (no silence trim/windowing)."""
    data = await wav.read()
    emb = embed_single_bytes(data, pp=pp)
    return EmbedResponse(embedding=emb.astype(float).tolist())


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity with L2 normalization applied to both vectors."""
    a = a.astype(np.float64); b = b.astype(np.float64)
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))


@app.post("/score_cosine", response_model=ScoreResponse)
async def score_cosine(a: UploadFile = File(...), b: UploadFile = File(...)):
    """Cosine similarity between mean-embeddings of two WAV files."""
    da = await a.read()
    db = await b.read()
    ea = embed_mean_bytes(da)
    eb = embed_mean_bytes(db)
    # Debug: print norms and first values to server log
    try:
        la = float(np.linalg.norm(ea)); lb = float(np.linalg.norm(eb))
        print(f"A: L2={la:.3f} first4={[float(x) for x in ea[:4]]}")
        print(f"B: L2={lb:.3f} first4={[float(x) for x in eb[:4]]}")
    except Exception:
        pass
    s = cosine(ea, eb)
    return ScoreResponse(score=s)


@app.post("/score_cosine_single", response_model=ScoreDetailed)
async def score_cosine_single(
    a: UploadFile = File(...),  # train
    b: UploadFile = File(...),  # test
    pp: str = Query("none"),
    dur_sec: float | None = Query(None, description="If not provided, duration of b is used"),
    log: int = Query(0),
):
    """Cosine similarity between single-shot embeddings (train vs test) plus duration-based threshold."""
    da = await a.read()
    db = await b.read()
    # Train emb cache (keyed by bytes+pp)
    ka = (_sha1(da), pp)
    ea = _cache_train.get(ka)
    if ea is None:
        # Build training embedding with optional preprocess and silence trim
        x_train = load_resample_mono_bytes(da, SR)
        if pp != "none":
            try:
                x_train = apply_preprocess(x_train, pp)
            except Exception:
                pass
        if SLIDING_TRIM_TRAIN:
            try:
                dur_b = len(x_train) / SR
            except Exception:
                dur_b = 0.0
            x_train = trim_silence(x_train, top_db=float(SLIDING_TRAIN_TOP_DB))
            try:
                dur_a = len(x_train) / SR
                print(f"train_trim score_single pp={pp} dur_before={dur_b:.3f}s dur_after={dur_a:.3f}s top_db={SLIDING_TRAIN_TOP_DB}")
            except Exception:
                pass
        ea = embed_wave(x_train)
        _cache_put(_cache_train, ka, ea)
    # Test emb (single call, skip cache since slice varies in other endpoint)
    eb = embed_single_bytes(db, pp=pp)
    # L2 normalize
    ea = ea / (np.linalg.norm(ea) + 1e-12)
    eb = eb / (np.linalg.norm(eb) + 1e-12)
    score = float(np.dot(ea, eb))
    # Determine duration
    if dur_sec is None:
        kb = _sha1(db)
        wav_b = _cache_test_wave.get(kb)
        if wav_b is None:
            wav_b = load_resample_mono_bytes(db, SR)
            _cache_put(_cache_test_wave, kb, wav_b)
        dur_sec = float(len(wav_b)) / float(SR)
    thr = duration_threshold(max(0.0, float(dur_sec)))
    margin = score - thr
    accepted = score >= thr
    if log:
        try:
            print(f"score_single pp={pp} dur={dur_sec:.3f} score={score:.3f} thr={thr:.3f} margin={margin:.3f} accepted={accepted}")
        except Exception:
            pass
    return ScoreDetailed(score=score, duration=float(dur_sec), threshold=thr, margin=margin, accepted=accepted)


@app.post("/score_cosine_slice", response_model=ScoreDetailed)
async def score_cosine_slice(
    train: UploadFile = File(...),
    test: UploadFile = File(...),
    s: float = Query(..., description="slice start (sec)"),
    e: float = Query(..., description="slice end (sec)"),
    pp: str = Query("none"),
    log: int = Query(0),
):
    """Cosine similarity between the train embedding and a test slice [s, e] with duration threshold."""
    # Load audio
    ta = await train.read()
    tb = await test.read()
    # Train embedding (cache)
    ka = (_sha1(ta), pp)
    ea_full = _cache_train.get(ka)
    if ea_full is None:
        x_train = load_resample_mono_bytes(ta, SR)
        if pp != "none":
            try:
                x_train = apply_preprocess(x_train, pp)
            except Exception:
                pass
        if SLIDING_TRIM_TRAIN:
            try:
                dur_b = len(x_train) / SR
            except Exception:
                dur_b = 0.0
            x_train = trim_silence(x_train, top_db=float(SLIDING_TRAIN_TOP_DB))
            try:
                dur_a = len(x_train) / SR
                print(f"train_trim score_slice pp={pp} dur_before={dur_b:.3f}s dur_after={dur_a:.3f}s top_db={SLIDING_TRAIN_TOP_DB}")
            except Exception:
                pass
        ea_full = embed_wave(x_train)
        _cache_put(_cache_train, ka, ea_full)
    # Test wave (cache full file load only)
    kb = _sha1(tb)
    x_test = _cache_test_wave.get(kb)
    if x_test is None:
        x_test = load_resample_mono_bytes(tb, SR)
        _cache_put(_cache_test_wave, kb, x_test)
    # Slice
    sS = int(max(0.0, s) * SR)
    sec_limit = min(e, len(x_test) / SR)
    eS = max(sS, int(sec_limit * SR))
    w = x_test[sS:eS]
    # Apply pp to slice
    if pp != "none":
        try:
            w = apply_preprocess(w, pp)
        except Exception:
            pass
    # Single-shot embeds
    ea = ea_full
    eb = embed_wave(w)
    ea = ea / (np.linalg.norm(ea) + 1e-12)
    eb = eb / (np.linalg.norm(eb) + 1e-12)
    score = float(np.dot(ea, eb))
    dur_sec = max(0.0, (eS - sS) / SR)
    thr = duration_threshold(dur_sec)
    margin = score - thr
    accepted = score >= thr
    if log:
        try:
            print(f"score_slice pp={pp} s={s:.3f} e={e:.3f} dur={dur_sec:.3f} score={score:.3f} thr={thr:.3f} margin={margin:.3f} accepted={accepted}")
        except Exception:
            pass
    return ScoreDetailed(score=score, duration=dur_sec, threshold=thr, margin=margin, accepted=accepted)


@app.post("/score_cosine_slice_batch")
async def score_cosine_slice_batch(
    train: UploadFile = File(...),
    test: UploadFile = File(...),
    slices: str = Form(..., description="JSON array of [start_sec, end_sec]"),
    pp: str = Query("none"),
    log: int = Query(0),
):
    """
    Batch single-shot scoring for many slices in one request.
    Request:
      - train: training WAV
      - test: test WAV
      - slices: JSON string '[ [s1,e1], [s2,e2], ... ]'
      - pp: preprocess mode
    Response: list of ScoreDetailed (one per slice, in order)
    """
    import json as _json
    ta = await train.read()
    tb = await test.read()
    # Train embedding (cache)
    ka = (_sha1(ta), pp)
    ea_full = _cache_train.get(ka)
    if ea_full is None:
        x_train = load_resample_mono_bytes(ta, SR)
        if pp != "none":
            try:
                x_train = apply_preprocess(x_train, pp)
            except Exception:
                pass
        if SLIDING_TRIM_TRAIN:
            try:
                dur_b = len(x_train) / SR
            except Exception:
                dur_b = 0.0
            x_train = trim_silence(x_train, top_db=float(SLIDING_TRAIN_TOP_DB))
            try:
                dur_a = len(x_train) / SR
                print(f"train_trim score_batch pp={pp} dur_before={dur_b:.3f}s dur_after={dur_a:.3f}s top_db={SLIDING_TRAIN_TOP_DB}")
            except Exception:
                pass
        ea_full = embed_wave(x_train)
        _cache_put(_cache_train, ka, ea_full)
    ea = ea_full / (np.linalg.norm(ea_full) + 1e-12)
    # Test wave (cache)
    kb = _sha1(tb)
    x_test = _cache_test_wave.get(kb)
    if x_test is None:
        x_test = load_resample_mono_bytes(tb, SR)
        _cache_put(_cache_test_wave, kb, x_test)
    # Parse slices
    try:
        pairs = _json.loads(slices)
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=f"Invalid slices JSON: {e}")
    out: list[dict] = []
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        s = float(pair[0]); e = float(pair[1])
        sS = int(max(0.0, s) * SR)
        sec_limit = min(e, len(x_test) / SR)
        eS = max(sS, int(sec_limit * SR))
        w = x_test[sS:eS]
        if pp != "none":
            try:
                w = apply_preprocess(w, pp)
            except Exception:
                pass
        eb = embed_wave(w)
        eb = eb / (np.linalg.norm(eb) + 1e-12)
        score = float(np.dot(ea, eb))
        dur_sec = max(0.0, (eS - sS) / SR)
        thr = duration_threshold(dur_sec)
        margin = score - thr
        accepted = score >= thr
        if log:
            try:
                print(f"score_batch slice s={s:.3f} e={e:.3f} dur={dur_sec:.3f} sc={score:.3f} thr={thr:.3f} acc={accepted}")
            except Exception:
                pass
        out.append({
            "score": score,
            "duration": dur_sec,
            "threshold": float(thr),
            "margin": float(margin),
            "accepted": bool(accepted),
        })
    return JSONResponse(out)

@app.post("/segments", response_model=SegmentsResponse)
async def segments(wav: UploadFile = File(...)):
    """Return raw VAD segments (start/end in seconds) from librosa.effects.split."""
    data = await wav.read()
    # Load at native SR for precise timings
    x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    # Use librosa energy-based VAD to split voiced regions
    intervals = librosa.effects.split(x, top_db=25)
    segs: List[List[float]] = []
    for s, e in intervals:
        segs.append([float(s) / float(sr), float(e) / float(sr)])
    return SegmentsResponse(segments=segs)

@app.post("/plot_pair_sliding")
async def plot_pair_sliding(
    train: UploadFile = File(...),
    test: UploadFile = File(...),
    win_sec: float = Query(SLIDING_WIN_SEC),
    hop_sec: float = Query(SLIDING_HOP_SEC),
    min_keep: float = Query(SLIDING_MIN_KEEP_SEC),
    top_db: float = Query(SLIDING_TOP_DB),
    pp: str = Query(SLIDING_PP),
):
    """Legacy placeholder that renders only the test waveform (ML disabled)."""
    # Legacy endpoint kept for compatibility but disabled to avoid model work.
    # Redirect to render-only path using empty decisions (flat waveform only).
    from fastapi.responses import StreamingResponse
    import io as _io
    test_bytes = await test.read()
    x_test = load_resample_mono_bytes(test_bytes, SR)
    t = np.arange(len(x_test)) / SR
    fig = plt.figure(figsize=(12,3)); ax = plt.gca()
    ax.plot(t, x_test, linewidth=0.3, color="#333")
    ax.set_xlabel("Seconds"); ax.set_title(test.filename if hasattr(test,'filename') else "test.wav")
    ax.xaxis.set_major_locator(MultipleLocator(1.0)); ax.grid(True, axis='x', linestyle='--', linewidth=0.4, alpha=0.25)
    plt.figtext(0.01, 0.01, "source=disabled_plot_pair_sliding", ha='left', va='bottom', fontsize=8, family='monospace')
    plt.tight_layout(rect=[0,0.04,1,1])
    buf = _io.BytesIO(); fig.savefig(buf, format='png', dpi=150); plt.close(fig); buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
    

@app.post("/plot_pair")
async def plot_pair(
    train: UploadFile = File(...),
    test: UploadFile = File(...),
    pp: str = Query("none"),
    smooth: int = Query(1),  # apply client-like smoothing for AMP by default
    debug: int = Query(0),
):
    """Return a PNG visualization with accept/reject segments for a train/test pair.
    - Amplitude VAD segmentation (with optional smoothing to mirror app)
    - Embeddings computed server-side (mean-embedding), cosine vs training embedding
    - Threshold from duration_threshold(d)
    """
    # Legacy endpoint kept for compatibility but disabled to avoid model work.
    from fastapi.responses import StreamingResponse
    import io as _io
    test_bytes = await test.read()
    x_test = load_resample_mono_bytes(test_bytes, SR)
    t = np.arange(len(x_test)) / SR
    fig = plt.figure(figsize=(12, 3))
    ax = plt.gca(); ax.plot(t, x_test, linewidth=0.3, color="#333")
    plt.figtext(0.01, 0.01, "source=disabled_plot_pair", ha='left', va='bottom', fontsize=8, family='monospace')
    plt.tight_layout(rect=[0,0.04,1,1])
    buf = _io.BytesIO(); fig.savefig(buf, format='png', dpi=150); plt.close(fig); buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.post("/segments_sliding", response_model=SegmentsResponse)
async def segments_sliding(
    train: UploadFile = File(...),
    test: UploadFile = File(...),
    win_sec: float = Query(SLIDING_WIN_SEC),
    hop_sec: float = Query(SLIDING_HOP_SEC),
    min_keep: float = Query(SLIDING_MIN_KEEP_SEC, description="Drop segments shorter than this (s)"),
    top_db: float = Query(SLIDING_TOP_DB, description="VAD threshold for silence trimming"),
    pp: str = Query(SLIDING_PP),
):
    """Sliding-window verification segmentation via shared implementation."""
    # Load audio
    train_bytes = await train.read()
    test_bytes = await test.read()
    x_train = load_resample_mono_bytes(train_bytes, SR)
    x_test = load_resample_mono_bytes(test_bytes, SR)
    # Delegate to shared implementation (onset-anchored hop stepping)
    segs = compute_sliding_segments(
        x_train=x_train,
        x_test=x_test,
        win_sec=win_sec,
        hop_sec=hop_sec,
        min_keep=min_keep,
        top_db=top_db,
        pp=pp,
    )
    return SegmentsResponse(segments=segs)


# (removed segments_sliding_emb during rollback)

# Run: uvicorn IOS.tools.embed_server:app --reload --port 8000

#cd /Users/bclh/Dropbox/Github/Voice-Recognition
#/opt/anaconda3/envs/VoiceRecognition/bin/python -m uvicorn IOS.tools.embed_server:app --host 0.0.0.0 --port 8000