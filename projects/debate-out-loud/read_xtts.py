#!/usr/bin/env python3
"""Read aloud via Coqui XTTS-v2 — fully offline neural TTS.

Single-page web UI: paste text, pick voice, click Read. Speech is synthesized
on the local GPU (CUDA if available, else CPU) and returned as a WAV the
browser plays through standard `<audio>` — no WebRTC, no API key.

First synthesis loads the model (~10s, ~2 GB download the very first time).
Subsequent requests reuse the warm model.
"""
from __future__ import annotations

import io
import json
import os
import re
import secrets
import struct
import sys
import threading
import time
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs


# Accept Coqui's model license non-interactively.
os.environ.setdefault("COQUI_TOS_AGREED", "1")


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(Path(__file__).resolve().parent / ".env")


# ---------------------------------------------------------------------------
# Passkey auth — gates every route except /login and /auth. The browser holds
# a random per-process session token in a cookie; the actual passkey never
# leaves the server. Restart invalidates all sessions.
# ---------------------------------------------------------------------------
PASSKEY = os.environ.get("READ_XTTS_PASSKEY", "1200gulf10129trails")
SESSION_TOKEN = secrets.token_urlsafe(32)
AUTH_COOKIE = "rxauth"

LOGIN_HTML = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#0a0a0a">
<title>Read aloud — sign in</title>
<style>
body{background:#0a0a0a;color:#e6e6e6;font:14px system-ui,sans-serif;
     display:flex;align-items:center;justify-content:center;
     min-height:100vh;margin:0;padding:16px;box-sizing:border-box;
     -webkit-text-size-adjust:100%}
form{background:#161616;padding:24px 24px;border-radius:10px;
     border:1px solid #2a2a2a;width:100%;max-width:340px;box-sizing:border-box}
h1{margin:0 0 16px;font-size:16px;font-weight:600}
input[type=password]{width:100%;box-sizing:border-box;background:#0a0a0a;
     color:#e6e6e6;border:1px solid #333;border-radius:6px;padding:12px;
     font:16px monospace}
button{margin-top:14px;width:100%;padding:12px;background:#2a5fb4;color:#fff;
     border:none;border-radius:6px;font:16px sans-serif;cursor:pointer;
     min-height:44px}
.err{color:#ff7a7a;margin-top:10px;font-size:12px;min-height:1em}
</style></head><body>
<form method="POST" action="/auth">
  <h1>Read aloud — sign in</h1>
  <input name="passkey" type="password" autofocus placeholder="passkey" autocomplete="current-password">
  <button>Unlock</button>
  <div class="err">__ERROR__</div>
</form></body></html>
"""


# ---------------------------------------------------------------------------
# Voice ratings — bridge from the browser's localStorage to a disk file so
# other tools (debate.py) can pick top-rated voices. The rating UI saves to
# localStorage; the main page POSTs that to /scores; we persist it here.
# ---------------------------------------------------------------------------

SCORES_PATH = Path(__file__).resolve().parent / "voice_scores.json"
VOICES_DIR = Path(__file__).resolve().parent / "voices"
HISTORY_DIR = Path(__file__).resolve().parent / "history"
HISTORY_MAX = 100  # rolling cap; oldest pruned after each successful save
_scores_lock = threading.Lock()
_history_lock = threading.Lock()


def list_custom_voices() -> list[dict]:
    """Discover cloning reference WAVs in voices/. Each entry: {name, file}."""
    if not VOICES_DIR.is_dir():
        return []
    return [{"name": p.stem, "file": p.name}
            for p in sorted(VOICES_DIR.glob("*.wav"))]


def resolve_custom_voice(filename: str) -> Path | None:
    """Return absolute path for `filename` iff it's a real WAV under voices/.

    Rejects path traversal — only bare filenames are accepted; the resolved
    path must still sit directly under VOICES_DIR.
    """
    if not filename or "/" in filename or "\\" in filename or filename.startswith("."):
        return None
    candidate = (VOICES_DIR / filename).resolve()
    if candidate.parent != VOICES_DIR.resolve():
        return None
    if not candidate.is_file():
        return None
    return candidate


def load_scores() -> dict:
    """Return {'xtts': {voice: {score,t}}, 'gemini': {...}} from disk."""
    try:
        with _scores_lock:
            if SCORES_PATH.exists():
                return json.loads(SCORES_PATH.read_text() or "{}")
    except Exception:
        pass
    return {"xtts": {}, "gemini": {}}


def merge_scores(incoming: dict) -> dict:
    """Merge posted scores into the on-disk file (newer timestamp wins).

    Per-voice record can carry two independent ratings, each with its own
    timestamp so the two channels don't fight each other:
      score / t          — sampler's 1-9 audition rating (/sample page)
      stars / stars_t    — main page's 1-5 post-generation rating
    """
    with _scores_lock:
        current = {"xtts": {}, "gemini": {}}
        if SCORES_PATH.exists():
            try:
                current = json.loads(SCORES_PATH.read_text() or "{}")
            except Exception:
                pass
        for engine in ("xtts", "gemini"):
            cur = current.setdefault(engine, {})
            for voice, rec in (incoming.get(engine) or {}).items():
                if not isinstance(rec, dict):
                    continue
                slot = cur.setdefault(voice, {})
                if "score" in rec:
                    t = rec.get("t", 0)
                    if t >= slot.get("t", 0):
                        slot["score"] = rec["score"]
                        slot["t"] = t
                if "stars" in rec:
                    t = rec.get("stars_t", 0)
                    if t >= slot.get("stars_t", 0):
                        try:
                            n = int(rec["stars"])
                        except (TypeError, ValueError):
                            continue
                        if 1 <= n <= 5:
                            slot["stars"] = n
                            slot["stars_t"] = t
        SCORES_PATH.write_text(json.dumps(current, indent=2))
        return current


# ---------------------------------------------------------------------------
# Generation history — every successful synth is written to history/ as a
# .wav plus a .json sidecar. The web UI reads /history to render a playable
# list. Bounded by HISTORY_MAX so the directory doesn't grow unbounded.
# ---------------------------------------------------------------------------

_HIST_ID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}_[0-9a-f]{8}$")
HISTORY_PREVIEW_CHARS = 200

# In-memory cache of history sidecars, newest-first. Populated lazily on
# first use; mutated in lockstep with disk by save_history / prune. Avoids
# globbing + 100 file opens on every GET /history (which fires after every
# generation via refreshHistory()).
_history_index: list[dict] = []
_history_loaded = False


def _history_id_safe(uid: str) -> bool:
    return bool(_HIST_ID_RE.match(uid or ""))


def _ensure_history_loaded_locked() -> None:
    """Populate _history_index from disk if not already. Caller holds lock."""
    global _history_loaded
    if _history_loaded:
        return
    _history_index.clear()
    if HISTORY_DIR.is_dir():
        paths = sorted(HISTORY_DIR.glob("*.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)[:HISTORY_MAX]
        for jp in paths:
            try:
                _history_index.append(json.loads(jp.read_text()))
            except Exception:
                continue
    _history_loaded = True


def save_history(*, text: str, engine: str, voice: str, speed: float,
                 language: str, wav_bytes: bytes, sample_rate: int,
                 duration_seconds: float) -> str:
    """Write a WAV + sidecar JSON describing one generation; return its id."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    now = time.time()
    base = (time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
            + "_" + uuid.uuid4().hex[:8])
    meta = {
        "id": base,
        "created_at": now,
        "engine": engine,
        "voice": voice or "",
        "speed": float(speed),
        "language": language or "",
        "text": text,
        "sample_rate": int(sample_rate),
        "duration_seconds": float(duration_seconds),
        "bytes": len(wav_bytes),
    }
    with _history_lock:
        (HISTORY_DIR / f"{base}.wav").write_bytes(wav_bytes)
        (HISTORY_DIR / f"{base}.json").write_text(json.dumps(meta))
        _ensure_history_loaded_locked()
        _history_index.insert(0, meta)
        _prune_history_locked()
    return base


def _prune_history_locked() -> None:
    """Trim disk + in-memory to HISTORY_MAX entries. Caller holds lock.

    Assumes _history_index is already populated and ordered newest-first.
    Trims the in-memory tail, deletes matching disk files. We don't re-scan
    the directory — anything not in the index isn't our problem.
    """
    while len(_history_index) > HISTORY_MAX:
        old = _history_index.pop()
        old_id = (old.get("id") or "")
        if not _history_id_safe(old_id):
            continue
        for p in (HISTORY_DIR / f"{old_id}.wav",
                  HISTORY_DIR / f"{old_id}.json"):
            try:
                p.unlink()
            except FileNotFoundError:
                pass


def _entry_for_list(meta: dict) -> dict:
    """Slim copy of a history entry for /history responses.

    Strips the full text body (can be many KB) and substitutes a fixed-size
    preview so the listing payload stays small regardless of how long the
    underlying generations were. Full text remains on disk in the sidecar.
    """
    text = meta.get("text") or ""
    if len(text) > HISTORY_PREVIEW_CHARS:
        preview = text[:HISTORY_PREVIEW_CHARS] + "…"
    else:
        preview = text
    out = {k: v for k, v in meta.items() if k != "text"}
    out["text_preview"] = preview
    return out


def list_history() -> list[dict]:
    """Return history listing newest-first, capped at HISTORY_MAX entries.

    Each entry has `text_preview` instead of `text`; the full text stays in
    the on-disk sidecar.
    """
    with _history_lock:
        _ensure_history_loaded_locked()
        return [_entry_for_list(m) for m in _history_index]


# ---------------------------------------------------------------------------
# Lazy XTTS-v2 loader. Loading takes ~10s; we keep a single instance warm.
# ---------------------------------------------------------------------------

_tts_lock = threading.Lock()
_tts = None
_tts_device: str = "cpu"
_tts_speakers: list[str] = []


def _patch_xtts_audio_loader() -> None:
    """Swap XTTS's torchaudio-based audio loader for a soundfile-based one.

    torchaudio 2.11 routes file loads through torchcodec, which requires
    FFmpeg shared libs the system doesn't fully ship. We only ever feed XTTS
    pre-cleaned WAVs from voices/, so a soundfile + torch.functional.resample
    path is sufficient and avoids the FFmpeg dependency entirely.
    """
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio.functional as taf
    from TTS.tts.models import xtts as _xtts

    def _load_audio(audiopath, sampling_rate):
        data, lsr = sf.read(str(audiopath), always_2d=True)
        audio = torch.from_numpy(data.T.astype(np.float32))
        if audio.size(0) != 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        if lsr != sampling_rate:
            audio = taf.resample(audio, lsr, sampling_rate)
        audio.clip_(-1, 1)
        return audio

    _xtts.load_audio = _load_audio


def get_tts():
    global _tts, _tts_device, _tts_speakers
    if _tts is not None:
        return _tts
    with _tts_lock:
        if _tts is not None:
            return _tts
        import torch
        from TTS.api import TTS
        _tts_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[xtts] loading model on {_tts_device}…", file=sys.stderr)
        t0 = time.time()
        _tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(_tts_device)
        _patch_xtts_audio_loader()
        try:
            _tts_speakers = list(_tts.synthesizer.tts_model.speaker_manager.name_to_id.keys())
        except Exception:
            _tts_speakers = list(getattr(_tts, "speakers", []) or [])
        print(f"[xtts] loaded in {time.time()-t0:.1f}s; "
              f"{len(_tts_speakers)} built-in speakers", file=sys.stderr)
    return _tts


def synth_wav(text: str, speaker: str | None, language: str = "en",
              speed: float = 1.0, *, speaker_wav: str | None = None) -> bytes:
    """Synthesize `text`, return WAV bytes (mono 24 kHz int16).

    Pass either `speaker` (built-in XTTS speaker name) or `speaker_wav` (path
    to a reference WAV for zero-shot cloning). `speed` is clamped to 0.5–2.0.
    """
    tts = get_tts()
    import numpy as np
    s = max(0.5, min(2.0, float(speed)))
    if speaker_wav:
        cache_dir = VOICES_DIR / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        wav = tts.tts(text=text,
                      speaker=Path(speaker_wav).stem,
                      speaker_wav=speaker_wav,
                      voice_dir=str(cache_dir),
                      language=language, speed=s)
    else:
        wav = tts.tts(text=text, speaker=speaker, language=language, speed=s)
    arr = np.asarray(wav, dtype=np.float32)
    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr * 32767.0).astype(np.int16).tobytes()
    sr = int(tts.synthesizer.output_sample_rate or 24000)
    return _wav_wrap(pcm, n_channels=1, sample_width=2, sample_rate=sr)


def _wav_wrap(pcm: bytes, n_channels: int, sample_width: int, sample_rate: int) -> bytes:
    byte_rate = sample_rate * n_channels * sample_width
    block_align = n_channels * sample_width
    bits = sample_width * 8
    data_size = len(pcm)
    riff_size = 36 + data_size
    header = (
        b"RIFF" + struct.pack("<I", riff_size) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, n_channels, sample_rate,
                                byte_rate, block_align, bits)
        + b"data" + struct.pack("<I", data_size)
    )
    return header + pcm


# ---------------------------------------------------------------------------
# Streaming synth: split text into sentence-sized chunks, emit a WAV header
# with an "unknown length" data chunk up front, then write raw PCM to the
# socket as each chunk finishes. Browsers play <audio src> progressively.
# ---------------------------------------------------------------------------

# Serialize GPU inference across concurrent stream handlers — XTTS isn't
# safe to call from multiple threads against the same model instance.
_infer_lock = threading.Lock()


def split_for_synth(text: str) -> list[str]:
    """Split text into ~80–220 char chunks on sentence/clause boundaries."""
    text = (text or "").strip()
    if not text:
        return []
    parts = [p for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    merged: list[str] = []
    for p in parts:
        if merged and len(merged[-1]) < 80 and len(p) < 60:
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    out: list[str] = []
    for c in merged:
        if len(c) <= 220:
            out.append(c)
            continue
        sub = re.split(r"(?<=[,;:])\s+", c)
        buf = ""
        for s in sub:
            if not buf:
                buf = s
            elif len(buf) + 1 + len(s) <= 200:
                buf = buf + " " + s
            else:
                out.append(buf)
                buf = s
        if buf:
            out.append(buf)
    return out


def _wav_header_unknown_length(n_channels: int, sample_width: int,
                                sample_rate: int) -> bytes:
    """WAV header with max-int data/riff sizes so browsers stream it."""
    byte_rate = sample_rate * n_channels * sample_width
    block_align = n_channels * sample_width
    bits = sample_width * 8
    return (
        b"RIFF" + struct.pack("<I", 0xFFFFFFFE) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, n_channels, sample_rate,
                                byte_rate, block_align, bits)
        + b"data" + struct.pack("<I", 0xFFFFFFFF)
    )


# Short-lived job table: POST /tts/prepare stashes params here under a uuid,
# then the browser's <audio src="/tts/stream?id=..."> picks them up. We do
# this two-step dance because <audio> can only GET, and we don't want long
# text in a query string.
_jobs_lock = threading.Lock()
_jobs: dict[str, dict] = {}
_JOB_TTL_SEC = 120.0


def _stash_job(payload: dict) -> str:
    uid = uuid.uuid4().hex
    now = time.time()
    with _jobs_lock:
        stale = [k for k, v in _jobs.items() if now - v["t"] > _JOB_TTL_SEC]
        for k in stale:
            _jobs.pop(k, None)
        _jobs[uid] = {"payload": payload, "t": now}
    return uid


def _pop_job(uid: str) -> dict | None:
    with _jobs_lock:
        return _jobs.pop(uid, None)


# A curated subset of XTTS-v2 speakers that sound clean in English.
# Full list is exposed at GET /voices once the model is loaded.
RECOMMENDED = [
    "Claribel Dervla",
    "Daisy Studious",
    "Gracie Wise",
    "Tammie Ema",
    "Alison Dietlinde",
    "Ana Florence",
    "Damien Black",
    "Ferran Simen",
    "Viktor Eka",
    "Filip Traverse",
    "Andrew Chipper",
]


# ---------------------------------------------------------------------------
# Gemini Flash TTS backend (cloud, requires GEMINI_API_KEY)
# ---------------------------------------------------------------------------

GEMINI_MODEL = os.environ.get("GEMINI_TTS_MODEL", "gemini-3.1-flash-tts-preview")

# Google's prebuilt voices with the one-word style label they ship for each.
# https://ai.google.dev/gemini-api/docs/speech-generation
GEMINI_VOICES: list[tuple[str, str]] = [
    ("Zephyr",         "bright"),
    ("Puck",           "upbeat"),
    ("Charon",         "informative"),
    ("Kore",           "firm"),
    ("Fenrir",         "excitable"),
    ("Leda",           "youthful"),
    ("Orus",           "firm"),
    ("Aoede",          "breezy"),
    ("Callirrhoe",     "easy-going"),
    ("Autonoe",        "bright"),
    ("Enceladus",      "breathy"),
    ("Iapetus",        "clear"),
    ("Umbriel",        "easy-going"),
    ("Algieba",        "smooth"),
    ("Despina",        "smooth"),
    ("Erinome",        "clear"),
    ("Algenib",        "gravelly"),
    ("Rasalgethi",     "informative"),
    ("Laomedeia",      "upbeat"),
    ("Achernar",       "soft"),
    ("Alnilam",        "firm"),
    ("Schedar",        "even"),
    ("Gacrux",         "mature"),
    ("Pulcherrima",    "forward"),
    ("Achird",         "friendly"),
    ("Zubenelgenubi",  "casual"),
    ("Vindemiatrix",   "gentle"),
    ("Sadachbia",      "lively"),
    ("Sadaltager",     "knowledgeable"),
    ("Sulafat",        "warm"),
]


def gemini_available() -> bool:
    if not os.environ.get("GEMINI_API_KEY"):
        return False
    try:
        import google.genai  # noqa: F401
        return True
    except Exception:
        return False


def _gemini_synth_chunk(client, types_mod, text: str, voice: str) -> bytes:
    """Call Gemini TTS once and return raw 24 kHz mono int16 PCM bytes."""
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=text,
        config=types_mod.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types_mod.SpeechConfig(
                voice_config=types_mod.VoiceConfig(
                    prebuilt_voice_config=types_mod.PrebuiltVoiceConfig(
                        voice_name=voice,
                    ),
                ),
            ),
        ),
    )
    parts = resp.candidates[0].content.parts
    for p in parts:
        d = getattr(p, "inline_data", None)
        if d and d.data:
            return d.data if isinstance(d.data, (bytes, bytearray)) else bytes(d.data)
    raise RuntimeError("Gemini response had no inline audio data")


GEMINI_PARALLEL_WORKERS = int(os.environ.get("READ_XTTS_GEMINI_WORKERS", "4"))


def synth_gemini(text: str, voice: str = "Aoede") -> bytes:
    """Synthesize via Gemini TTS, returning a single WAV (24 kHz mono int16).

    Always chunks via `split_for_synth` and concatenates raw PCM, even for
    short input. Single-shot calls hit Gemini's per-request audio-output
    budget on long text — chunking dodges both the truncation and the
    documented "quality drifts after a few minutes" caveat.

    Chunks are fanned out concurrently up to GEMINI_PARALLEL_WORKERS, then
    reassembled in submission order so audio plays in the right sequence.
    Fail-fast: first chunk to error aborts the rest.
    """
    from google import genai
    from google.genai import types
    chunks = split_for_synth(text)
    if not chunks:
        raise RuntimeError("empty text")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    total = len(chunks)

    def synth_one(idx: int, chunk: str) -> bytes:
        t0 = time.time()
        last_err: Exception | None = None
        for attempt in (1, 2):
            try:
                pcm = _gemini_synth_chunk(client, types, chunk, voice)
                last_err = None
                break
            except Exception as e:
                last_err = e
                if attempt == 1:
                    print(f"[gemini-tts] chunk {idx}/{total} attempt 1 "
                          f"failed ({type(e).__name__}: {e}); retrying…",
                          file=sys.stderr)
                    time.sleep(1.0)
        if last_err is not None:
            raise RuntimeError(
                f"Gemini chunk {idx}/{total} failed after retry: "
                f"{type(last_err).__name__}: {last_err}"
            ) from last_err
        dt = time.time() - t0
        audio_ms = (len(pcm) / (2 * 24000)) * 1000.0
        print(f"[gemini-tts] chunk {idx}/{total}: {len(chunk)} chars → "
              f"{audio_ms:.0f}ms audio in {dt:.2f}s", file=sys.stderr)
        return pcm

    workers = max(1, min(GEMINI_PARALLEL_WORKERS, total))
    pcm_parts: list[bytes | None] = [None] * total
    if workers == 1:
        for i, c in enumerate(chunks):
            pcm_parts[i] = synth_one(i + 1, c)
    else:
        t_pool0 = time.time()
        with ThreadPoolExecutor(max_workers=workers,
                                thread_name_prefix="gemini-tts") as ex:
            futures = {ex.submit(synth_one, i + 1, c): i
                       for i, c in enumerate(chunks)}
            try:
                for fut in as_completed(futures):
                    pcm_parts[futures[fut]] = fut.result()
            except Exception:
                # Cancel anything still queued; in-flight calls will finish
                # on their own as the executor exits.
                for f in futures:
                    f.cancel()
                raise
        print(f"[gemini-tts] {total} chunks via {workers} workers in "
              f"{time.time() - t_pool0:.2f}s", file=sys.stderr)

    return _wav_wrap(b"".join(pcm_parts), n_channels=1, sample_width=2,
                     sample_rate=24000)


# ---------------------------------------------------------------------------
# HTML/JS UI — paste, pick a voice, hit Read.
# ---------------------------------------------------------------------------

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#0e0e12">
<title>Read aloud (XTTS-v2 local)</title>
<style>
  :root {
    --bg: #0e0e12;
    --panel: #16161d;
    --fg: #e6e6ee;
    --dim: #8a8aa0;
    --accent: #7fff7f;
    --rule: #2a2a36;
    --bad: #ff7f7f;
  }
  html, body { margin: 0; padding: 0; background: var(--bg); color: var(--fg);
    font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
    -webkit-text-size-adjust: 100%; }
  body { max-width: 980px; margin: 0 auto; padding: 24px;
    padding-left: max(24px, env(safe-area-inset-left));
    padding-right: max(24px, env(safe-area-inset-right)); }
  h1 { margin: 0 0 4px; font-size: 22px; font-weight: 600; line-height: 1.25; }
  .sub { color: var(--dim); font-size: 13px; margin-bottom: 16px; }
  /* 16px font on textarea/inputs avoids iOS Safari's zoom-on-focus. */
  textarea { width: 100%; min-height: 180px; resize: vertical;
    background: var(--panel); color: var(--fg); border: 1px solid var(--rule);
    border-radius: 10px; padding: 12px 14px; font: inherit; font-size: 16px;
    line-height: 1.4; box-sizing: border-box; }
  textarea:focus { outline: 2px solid var(--accent); border-color: transparent; }
  .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
    margin-top: 12px; }
  select, button, input[type=range] { font-size: 16px; }
  select, button { background: var(--panel); color: var(--fg);
    border: 1px solid var(--rule); border-radius: 8px; padding: 9px 14px;
    font: inherit; min-height: 40px; }
  button.primary { background: var(--accent); color: #0e1a0e; border: 0;
    font-weight: 600; cursor: pointer; }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  .status { color: var(--dim); font-size: 13px; margin-left: 8px; }
  .status.ok { color: var(--accent); }
  .status.err { color: var(--bad); }
  audio { width: 100%; max-width: 600px; margin-top: 14px; display: block; }
  footer { color: var(--dim); font-size: 12px; margin-top: 18px; }

  .rating { display: flex; align-items: center; gap: 10px; margin-top: 14px;
    flex-wrap: wrap; }
  .rating-label { color: var(--dim); font-size: 13px; }
  .rating-hint { color: var(--dim); font-size: 12px; }
  .stars { display: inline-flex; gap: 2px; }
  /* Each star is a button. Default to "dim" outline; flip to accent when
     the row is showing a rating up to N. Hover preview uses :hover on the
     row + ~ sibling state so trailing stars stay dim until pointed at. */
  .star { background: transparent; border: 0; padding: 2px 4px;
    font-size: 22px; line-height: 1; cursor: pointer; color: var(--rule);
    min-height: 0; transition: color 0.08s; }
  .star:hover, .star:focus { color: var(--fg); outline: none; }
  .stars.filled-1 .star:nth-child(-n+1),
  .stars.filled-2 .star:nth-child(-n+2),
  .stars.filled-3 .star:nth-child(-n+3),
  .stars.filled-4 .star:nth-child(-n+4),
  .stars.filled-5 .star:nth-child(-n+5) { color: var(--accent); }

  details.history { margin-top: 18px; }
  details.history > summary { color: var(--dim); font-size: 13px;
    cursor: pointer; padding: 6px 0; user-select: none; list-style: none; }
  details.history > summary::-webkit-details-marker { display: none; }
  details.history > summary::before { content: "▸ "; display: inline-block;
    width: 14px; transition: transform 0.15s; }
  details.history[open] > summary::before { transform: rotate(90deg); }
  details.history > summary:hover { color: var(--fg); }
  .hist-count { color: var(--dim); margin-left: 4px; }
  .hist-list { display: flex; flex-direction: column; gap: 8px;
    margin-top: 8px; max-height: 360px; overflow: auto; padding-right: 4px; }
  .hist-entry { display: flex; align-items: center; gap: 10px;
    padding: 8px 10px; background: var(--panel); border: 1px solid var(--rule);
    border-radius: 8px; }
  .hist-play { flex: 0 0 36px; height: 36px; min-height: 0; padding: 0;
    background: var(--accent); color: #0e1a0e; border: 0; border-radius: 6px;
    font-weight: 700; cursor: pointer; font-size: 14px; }
  .hist-play:disabled { opacity: 0.45; cursor: not-allowed; }
  .hist-meta { flex: 1 1 auto; min-width: 0; }
  .hist-line1 { font-size: 12px; color: var(--dim);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .hist-line2 { font-size: 13px; color: var(--fg); margin-top: 2px;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
    overflow: hidden; line-height: 1.35; }
  .hist-dl { color: var(--dim); text-decoration: none; font-size: 18px;
    padding: 4px 8px; flex: 0 0 auto; line-height: 1; }
  .hist-dl:hover { color: var(--accent); }
  .hist-empty { color: var(--dim); font-size: 13px; padding: 8px 4px; }

  @media (max-width: 640px) {
    body { padding: 14px;
      padding-left: max(14px, env(safe-area-inset-left));
      padding-right: max(14px, env(safe-area-inset-right)); }
    h1 { font-size: 20px; }
    .row { gap: 8px; margin-top: 10px; }
    /* Stack the controls into rows that breathe; main buttons go full width
       so they're easy to hit with a thumb. */
    #voice { flex: 1 1 100%; min-width: 0; }
    #read, #random, #stop, #save, #clear { flex: 1 1 calc(50% - 4px); }
    #read { order: -1; }
    .row label[for=voice] { display: none; }
    .row label.status { flex: 1 1 100%; margin-left: 0; justify-content: space-between; }
    .row label.status input[type=range] { flex: 1; width: auto; }
    #status { flex: 1 1 100%; margin-left: 0; }
    footer { font-size: 11px; }
  }
</style>
</head>
<body>
  <h1>Read aloud · Coqui XTTS-v2 (local)</h1>
  <div class="sub">Fully offline neural TTS running on this machine. Paste text, pick a voice, click Read.</div>
  <textarea id="text" placeholder="Paste text here…"></textarea>
  <div class="row">
    <label for="voice" class="status">Voice:</label>
    <select id="voice"></select>
    <button id="read" class="primary">Read</button>
    <button id="random" title="Pick a random voice with no star rating yet and Read">🎲 Try</button>
    <button id="stop">Stop</button>
    <button id="save" disabled title="Save the last generated audio as a WAV file">Save</button>
    <button id="clear" title="Clear the text; voice stays as selected">Clear</button>
    <label class="status" style="display:flex; align-items:center; gap:6px;">
      Speed
      <input type="range" id="speed" min="0.6" max="1.5" step="0.05" value="1.0" style="width:120px;">
      <span id="speed_val" style="min-width:36px;">1.00×</span>
    </label>
    <span class="status" id="status">ready</span>
  </div>
  <audio id="out" controls preload="auto"></audio>

  <div class="rating" id="rating" hidden>
    <span class="rating-label" id="ratingLabel">Rate this voice:</span>
    <div class="stars" id="stars" role="radiogroup" aria-label="rating">
      <button class="star" type="button" data-n="1" aria-label="1 star">★</button>
      <button class="star" type="button" data-n="2" aria-label="2 stars">★</button>
      <button class="star" type="button" data-n="3" aria-label="3 stars">★</button>
      <button class="star" type="button" data-n="4" aria-label="4 stars">★</button>
      <button class="star" type="button" data-n="5" aria-label="5 stars">★</button>
    </div>
    <span class="rating-hint" id="ratingHint"></span>
  </div>

  <details class="history" id="history">
    <summary>History <span id="hist_count" class="hist-count"></span></summary>
    <div class="hist-list" id="hist_list">
      <div class="hist-empty">No saved audio yet.</div>
    </div>
  </details>

  <footer>
    <span id="meta">device: …</span>
    &nbsp;·&nbsp; first synth loads the model (~10s); subsequent ones are fast.
  </footer>

<script>
const $text   = document.getElementById('text');
const $voice  = document.getElementById('voice');
const $read   = document.getElementById('read');
const $stop   = document.getElementById('stop');
const $status = document.getElementById('status');
const $out    = document.getElementById('out');
const $meta   = document.getElementById('meta');

// State for the Web Audio streaming player (XTTS path). Tracked so Stop
// can cancel an in-flight stream and silence queued AudioBufferSourceNodes.
let activeStream = null;

// Last finished audio, ready to be downloaded as a single WAV file.
// {blob, voice, engine} once a generation completes.
let lastAudio = null;

// One long-lived AudioContext, primed inside a user gesture before any
// `await`. iOS Safari gates Web Audio on user activation — a context first
// touched after an awaited fetch stays `suspended` forever, so streaming
// plays silently. Calling resume() must also happen synchronously inside
// the gesture; awaiting it does not count.
let audioCtx = null;
function getAudioCtx() {
  if (!audioCtx) {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    audioCtx = new Ctx();
  }
  if (audioCtx.state === 'suspended') {
    audioCtx.resume().catch(() => {});
  }
  return audioCtx;
}

function setStatus(msg, cls='') {
  $status.className = 'status' + (cls ? ' ' + cls : '');
  $status.textContent = msg;
}

function makeOption(engine, name, label) {
  const o = document.createElement('option');
  o.value = engine + '::' + name;
  o.dataset.engine = engine;
  o.dataset.voice = name;
  o.textContent = label;
  return o;
}

function loadScores(engine) {
  try {
    return JSON.parse(localStorage.getItem('voiceScores.' + engine) || '{}');
  } catch (e) { return {}; }
}

// Sort an engine's voices: star-rated highest (5★ → 1★), then sampler-rated
// (audition score high → low), then recommended (XTTS only), then original
// list order. Stars come from serverScores (server-authoritative, 1-5);
// sampler `score` comes from localStorage / sampler page (1-9).
function sortVoices(items, scores, stars, recSet) {
  const augmented = items.map((it, idx) => ({
    ...it,
    score: (scores[it.name] && typeof scores[it.name].score === 'number')
              ? scores[it.name].score : null,
    stars: (stars && stars[it.name] && Number.isInteger(stars[it.name].stars))
              ? stars[it.name].stars : null,
    rec: recSet ? recSet.has(it.name) : false,
    origIdx: idx,
  }));
  augmented.sort((a, b) => {
    const sta = a.stars == null ? -Infinity : a.stars;
    const stb = b.stars == null ? -Infinity : b.stars;
    if (stb !== sta) return stb - sta;
    const sa = a.score == null ? -Infinity : a.score;
    const sb = b.score == null ? -Infinity : b.score;
    if (sb !== sa) return sb - sa;
    if (a.rec !== b.rec) return a.rec ? -1 : 1;
    return a.origIdx - b.origIdx;
  });
  return augmented;
}

function labelFor(item) {
  // Stars win over sampler score so post-generation ratings (the real verdict)
  // take precedence in the label.
  if (item.stars != null) {
    return `${item.label} · ` + '★'.repeat(item.stars)
                              + '☆'.repeat(5 - item.stars);
  }
  if (item.score != null) return `${item.label} · ★${item.score}`;
  if (item.rec)           return `${item.label} · ☆`;
  return item.label;
}

// Push whatever ratings are in localStorage up to the server so disk-backed
// tools (debate.py) can read them. Runs once on page load; cheap and idempotent.
async function syncScoresToServer() {
  try {
    const payload = {
      xtts: loadScores('xtts'),
      gemini: loadScores('gemini'),
    };
    const hasAny = Object.keys(payload.xtts).length || Object.keys(payload.gemini).length;
    if (!hasAny) return;
    await fetch('/scores', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  } catch (e) { /* non-fatal */ }
}

async function loadVoices() {
  try {
    syncScoresToServer();   // fire-and-forget; persists ratings to disk
    const r = await fetch('/voices');
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    const totalXtts = (data.xtts.all || []).length;
    $meta.textContent =
      `xtts: ${data.device} · ${totalXtts} speakers   `
      + `gemini: ${data.gemini.enabled ? data.gemini.model : 'disabled (' + (data.gemini.missing||'') + ')'}`;
    $voice.innerHTML = '';

    const xttsScores = loadScores('xtts');
    const geminiScores = loadScores('gemini');
    const xttsStars = serverScores.xtts || {};
    const geminiStars = serverScores.gemini || {};
    let topXtts = null, topGemini = null, topCustom = null;

    // Stamp baseLabel + rec onto each option so we can recompute the label
    // (e.g. after a star rating change) without rebuilding the dropdown.
    function appendOpt(group, engine, it, extras) {
      const o = makeOption(engine, it.name, labelFor(it));
      o.dataset.baseLabel = it.label;
      o.dataset.rec = it.rec ? '1' : '0';
      if (extras) Object.assign(o.dataset, extras);
      group.appendChild(o);
      return o;
    }

    // Custom cloned voices first (most prominent). Score namespace is shared
    // with XTTS so a rated clone competes against the built-ins.
    if (data.xtts.custom && data.xtts.custom.length) {
      const og = document.createElement('optgroup');
      og.label = 'Custom (cloned)';
      const items = data.xtts.custom.map(c => ({
        name: c.name, label: c.name, file: c.file,
      }));
      const sorted = sortVoices(items, xttsScores, xttsStars, null);
      for (const it of sorted) {
        appendOpt(og, 'xtts', it, { speakerWav: it.file });
      }
      $voice.appendChild(og);
      topCustom = sorted[0] || null;
    }

    if (data.gemini.enabled) {
      const og = document.createElement('optgroup');
      og.label = 'Gemini Flash TTS (cloud)';
      const items = data.gemini.voices.map(v => ({
        name: v.name, label: `${v.name} · ${v.style}`,
      }));
      const sorted = sortVoices(items, geminiScores, geminiStars, null);
      for (const it of sorted) {
        appendOpt(og, 'gemini', it, null);
      }
      $voice.appendChild(og);
      topGemini = sorted[0] || null;
    }

    if (data.xtts.all && data.xtts.all.length) {
      const og = document.createElement('optgroup');
      og.label = 'XTTS-v2 local';
      const items = data.xtts.all.map(n => ({ name: n, label: n }));
      const recSet = new Set(data.xtts.recommended || []);
      const sorted = sortVoices(items, xttsScores, xttsStars, recSet);
      for (const it of sorted) {
        appendOpt(og, 'xtts', it, null);
      }
      $voice.appendChild(og);
      topXtts = sorted[0] || null;
    }

    if (data.xtts.error) setStatus('xtts: ' + data.xtts.error, 'err');

    // Default selection: highest-rated voice across engines. On ties (e.g.
    // a fresh install with no ratings), prefer custom clones, then XTTS
    // built-ins, then Gemini — so first-time users land on the voice they
    // just added.
    const candidates = [];
    if (topCustom) candidates.push({ kind: 'custom', engine: 'xtts',   it: topCustom });
    if (topXtts)   candidates.push({ kind: 'xtts',   engine: 'xtts',   it: topXtts });
    if (topGemini) candidates.push({ kind: 'gemini', engine: 'gemini', it: topGemini });
    const kindRank = { custom: 0, xtts: 1, gemini: 2 };
    candidates.sort((a, b) => {
      const sta = a.it.stars == null ? -Infinity : a.it.stars;
      const stb = b.it.stars == null ? -Infinity : b.it.stars;
      if (stb !== sta) return stb - sta;
      const sa = a.it.score == null ? -Infinity : a.it.score;
      const sb = b.it.score == null ? -Infinity : b.it.score;
      if (sb !== sa) return sb - sa;
      return kindRank[a.kind] - kindRank[b.kind];
    });
    if (candidates.length) {
      $voice.value = candidates[0].engine + '::' + candidates[0].it.name;
    }
  } catch (e) {
    setStatus('voice list failed: ' + e.message, 'err');
  }
}

function stopActiveStream() {
  if (!activeStream) return;
  activeStream.aborted = true;
  try { activeStream.abortCtrl.abort(); } catch (e) {}
  for (const src of activeStream.sources) {
    try { src.stop(); } catch (e) {}
    try { src.disconnect(); } catch (e) {}
  }
  activeStream.sources.length = 0;
  activeStream = null;
}

// Parse a WAV header out of the leading bytes of the stream. Returns
// {sampleRate, channels, dataOffset} or null if we don't have enough bytes
// yet to be sure.
function parseWavHeader(buf) {
  if (buf.length < 44) return null;
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  // "RIFF" == 0x52494646 big-endian
  if (dv.getUint32(0, false) !== 0x52494646) return null;
  if (dv.getUint32(8, false) !== 0x57415645) return null; // "WAVE"
  const channels = dv.getUint16(22, true);
  const sampleRate = dv.getUint32(24, true);
  // Walk chunks looking for "data"
  let off = 12;
  while (off + 8 <= buf.length) {
    const id = String.fromCharCode(buf[off], buf[off+1], buf[off+2], buf[off+3]);
    const sz = dv.getUint32(off+4, true);
    if (id === 'data') return { sampleRate, channels, dataOffset: off + 8 };
    if (sz === 0xFFFFFFFF) return null; // shouldn't happen except for data
    off += 8 + sz;
  }
  return null;
}

function concatU8(a, b) {
  const out = new Uint8Array(a.length + b.length);
  out.set(a, 0); out.set(b, a.length);
  return out;
}

// Stream XTTS PCM into a Web Audio graph. Schedules each chunk as an
// AudioBufferSourceNode at the precise time the previous one ends, so the
// resulting playback is gapless. Returns when the server closes the stream
// (audio may still be playing out afterwards).
async function playXttsStream(url, t0) {
  // Assumes getAudioCtx() was already called in the click handler. Reusing
  // the singleton keeps the iOS gesture chain intact across awaits.
  const ctx = getAudioCtx();
  const abortCtrl = new AbortController();
  const state = { ctx, sources: [], aborted: false, abortCtrl };
  activeStream = state;

  let resp;
  try {
    resp = await fetch(url, { signal: abortCtrl.signal });
  } catch (e) {
    if (state.aborted) return;
    throw e;
  }
  if (!resp.ok) throw new Error(await resp.text());
  if (!resp.body) throw new Error('streaming fetch unsupported in this browser');

  const reader = resp.body.getReader();
  let header = null;
  let buf = new Uint8Array(0);
  let nextStartTime = 0;
  let firstAudioSignalled = false;
  // Accumulate all PCM bytes so we can build a downloadable WAV at the end.
  const pcmChunks = [];
  let pcmTotalBytes = 0;

  while (true) {
    if (state.aborted) { try { reader.cancel(); } catch (e) {} break; }
    const { done, value } = await reader.read();
    if (done) break;
    buf = concatU8(buf, value);
    if (!header) {
      header = parseWavHeader(buf);
      if (!header) continue;
      buf = buf.subarray(header.dataOffset);
      nextStartTime = ctx.currentTime + 0.05;
    }
    // Need an even byte count for Int16Array; keep any straggler byte.
    const evenLen = buf.length - (buf.length % 2);
    if (evenLen < 2) continue;
    const pcm = buf.subarray(0, evenLen);
    buf = buf.subarray(evenLen);
    const ab = pcm.buffer.slice(pcm.byteOffset, pcm.byteOffset + evenLen);
    pcmChunks.push(new Uint8Array(ab));
    pcmTotalBytes += ab.byteLength;
    const samples = new Int16Array(ab);
    const floats = new Float32Array(samples.length);
    for (let i = 0; i < samples.length; i++) floats[i] = samples[i] / 32768;
    if (state.aborted) break;
    const abuf = ctx.createBuffer(header.channels || 1, floats.length,
                                  header.sampleRate);
    abuf.getChannelData(0).set(floats);
    const src = ctx.createBufferSource();
    src.buffer = abuf;
    src.connect(ctx.destination);
    const startAt = Math.max(ctx.currentTime + 0.01, nextStartTime);
    src.start(startAt);
    state.sources.push(src);
    nextStartTime = startAt + abuf.duration;
    if (!firstAudioSignalled) {
      firstAudioSignalled = true;
      const dt = (performance.now() - t0) / 1000;
      setStatus(`first audio at ${dt.toFixed(2)}s — streaming…`, 'ok');
    }
  }
  if (state.aborted) return;
  if (header && pcmTotalBytes > 0) {
    const wav = buildWavBlob(header.sampleRate, header.channels || 1,
                             pcmChunks, pcmTotalBytes);
    lastAudio = { blob: wav, engine: 'xtts' };
    enableSaveButton(true);
  }
  const totalDt = (performance.now() - t0) / 1000;
  const audioEnd = nextStartTime - ctx.currentTime;
  setStatus(`stream done at ${totalDt.toFixed(2)}s · ${audioEnd > 0 ? audioEnd.toFixed(1)+'s audio queued' : 'playing'}`, 'ok');
}

// Build a complete WAV blob from accumulated PCM chunks (mono int16).
function buildWavBlob(sampleRate, channels, pcmChunks, totalBytes) {
  const byteRate = sampleRate * channels * 2;
  const blockAlign = channels * 2;
  const header = new Uint8Array(44);
  const dv = new DataView(header.buffer);
  // "RIFF"
  dv.setUint8(0, 0x52); dv.setUint8(1, 0x49); dv.setUint8(2, 0x46); dv.setUint8(3, 0x46);
  dv.setUint32(4, 36 + totalBytes, true);
  // "WAVE"
  dv.setUint8(8, 0x57); dv.setUint8(9, 0x41); dv.setUint8(10, 0x56); dv.setUint8(11, 0x45);
  // "fmt "
  dv.setUint8(12, 0x66); dv.setUint8(13, 0x6D); dv.setUint8(14, 0x74); dv.setUint8(15, 0x20);
  dv.setUint32(16, 16, true);
  dv.setUint16(20, 1, true);           // PCM
  dv.setUint16(22, channels, true);
  dv.setUint32(24, sampleRate, true);
  dv.setUint32(28, byteRate, true);
  dv.setUint16(32, blockAlign, true);
  dv.setUint16(34, 16, true);          // bits per sample
  // "data"
  dv.setUint8(36, 0x64); dv.setUint8(37, 0x61); dv.setUint8(38, 0x74); dv.setUint8(39, 0x61);
  dv.setUint32(40, totalBytes, true);
  return new Blob([header, ...pcmChunks], { type: 'audio/wav' });
}

function enableSaveButton(on) {
  const btn = document.getElementById('save');
  if (btn) btn.disabled = !on;
}

// --- Star rating -----------------------------------------------------------
// Server-side voice_scores.json mirrored here so the rating bar can show the
// current star count without re-fetching on every Read.
const serverScores = { xtts: {}, gemini: {} };
let currentRating = null;   // { engine, voice } of the most recent Read

async function loadServerScores() {
  try {
    const r = await fetch('/scores');
    if (!r.ok) return;
    const data = await r.json();
    for (const eng of ['xtts', 'gemini']) {
      Object.assign(serverScores[eng] = serverScores[eng] || {}, data[eng] || {});
    }
  } catch (e) {
    // Non-fatal — rating bar will just start unhighlighted.
  }
}

const $rating     = document.getElementById('rating');
const $stars      = document.getElementById('stars');
const $ratingLabel = document.getElementById('ratingLabel');
const $ratingHint = document.getElementById('ratingHint');

function currentStarsFor(engine, voice) {
  const rec = (serverScores[engine] || {})[voice];
  return rec && Number.isInteger(rec.stars) ? rec.stars : 0;
}

function setStarsFilled(n) {
  $stars.classList.remove('filled-1','filled-2','filled-3','filled-4','filled-5');
  if (n >= 1 && n <= 5) $stars.classList.add('filled-' + n);
}

function showRatingFor(engine, voice) {
  if (!voice) { $rating.hidden = true; currentRating = null; return; }
  currentRating = { engine, voice };
  $ratingLabel.textContent = `Rate ${voice}:`;
  const n = currentStarsFor(engine, voice);
  setStarsFilled(n);
  $ratingHint.textContent = n ? `${n}/5` : '';
  $rating.hidden = false;
}

async function postRating(engine, voice, stars) {
  const stars_t = Date.now();
  const payload = { [engine]: { [voice]: { stars, stars_t } } };
  // Update local cache immediately for snappy UI; server merge will agree.
  const slot = (serverScores[engine] = serverScores[engine] || {});
  slot[voice] = Object.assign({}, slot[voice] || {}, { stars, stars_t });
  refreshVoiceLabel(engine, voice);
  try {
    await fetch('/scores', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  } catch (e) {
    setStatus('rating save failed: ' + e.message, 'err');
  }
}

// Re-render one option's label in place using current serverScores + the
// stamped baseLabel/rec. We deliberately do NOT re-sort the dropdown — that
// would jump items around mid-session and surprise the user. Sort is fresh
// on the next page load.
function refreshVoiceLabel(engine, voice) {
  for (const o of $voice.options) {
    if (o.dataset.engine !== engine || o.dataset.voice !== voice) continue;
    const rec = (serverScores[engine] || {})[voice] || {};
    const localRec = (loadScores(engine) || {})[voice] || {};
    o.textContent = labelFor({
      label: o.dataset.baseLabel || o.textContent,
      stars: Number.isInteger(rec.stars) ? rec.stars : null,
      score: typeof localRec.score === 'number' ? localRec.score
             : (typeof rec.score === 'number' ? rec.score : null),
      rec: o.dataset.rec === '1',
    });
    return;
  }
}

$stars.addEventListener('click', (e) => {
  const btn = e.target.closest('.star');
  if (!btn || !currentRating) return;
  const n = parseInt(btn.dataset.n, 10);
  if (!(n >= 1 && n <= 5)) return;
  setStarsFilled(n);
  $ratingHint.textContent = `${n}/5 · saved`;
  postRating(currentRating.engine, currentRating.voice, n);
});

// --- History ---------------------------------------------------------------

const $histList  = document.getElementById('hist_list');
const $histCount = document.getElementById('hist_count');

function fmtHistTime(epoch) {
  const d = new Date(epoch * 1000);
  const now = new Date();
  const t = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  if (d.toDateString() === now.toDateString()) return t;
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' }) + ' ' + t;
}

function previewText(s, max=140) {
  s = (s || '').trim().replace(/\s+/g, ' ');
  return s.length <= max ? s : s.slice(0, max - 1) + '…';
}

function playHistory(item) {
  // A click handler — gesture chain is fresh, so $out.play() satisfies iOS.
  stopActiveStream();
  $out.pause();
  $out.src = '/history/' + item.id + '.wav';
  $out.playbackRate = parseFloat(document.getElementById('speed').value) || 1.0;
  $out.play().catch(e => setStatus('play failed: ' + e.message, 'err'));
  lastAudio = null;       // history entry isn't the "last generated" blob
  enableSaveButton(false);
  setStatus('history · ' + previewText(item.text_preview || '', 60), 'ok');
}

async function refreshHistory() {
  try {
    const r = await fetch('/history');
    if (!r.ok) return;
    const { items } = await r.json();
    $histList.innerHTML = '';
    if (!items || !items.length) {
      const empty = document.createElement('div');
      empty.className = 'hist-empty';
      empty.textContent = 'No saved audio yet.';
      $histList.appendChild(empty);
      $histCount.textContent = '';
      return;
    }
    $histCount.textContent = '(' + items.length + ')';
    for (const it of items) {
      const row = document.createElement('div');
      row.className = 'hist-entry';

      const btn = document.createElement('button');
      btn.className = 'hist-play';
      btn.textContent = '▶';
      btn.title = 'Play';
      btn.addEventListener('click', () => playHistory(it));

      const meta = document.createElement('div');
      meta.className = 'hist-meta';
      const l1 = document.createElement('div');
      l1.className = 'hist-line1';
      const dur = (it.duration_seconds || 0).toFixed(1) + 's';
      l1.textContent = `${fmtHistTime(it.created_at)} · ${it.engine} · ${it.voice || '—'} · ${dur}`;
      const l2 = document.createElement('div');
      l2.className = 'hist-line2';
      // Server already trimmed to HISTORY_PREVIEW_CHARS; previewText caps
      // further at the UI's 2-line clamp width.
      l2.textContent = previewText(it.text_preview || '');
      meta.appendChild(l1); meta.appendChild(l2);

      const dl = document.createElement('a');
      dl.className = 'hist-dl';
      dl.href = '/history/' + it.id + '.wav';
      const safeVoice = (it.voice || 'voice').replace(/[^\w-]+/g, '_');
      dl.download = `read_${safeVoice}_${it.id}.wav`;
      dl.textContent = '↓';
      dl.title = 'Download';

      row.appendChild(btn); row.appendChild(meta); row.appendChild(dl);
      $histList.appendChild(row);
    }
  } catch (e) {
    // Non-fatal — leave the previous list in place.
  }
}

async function read() {
  const text = $text.value.trim();
  if (!text) { setStatus('paste text first', 'err'); return; }
  // Prime the AudioContext while we're still inside the click handler.
  // Doing this after any `await` would lose the iOS user-gesture grant.
  getAudioCtx();
  stopActiveStream();
  $out.pause();
  lastAudio = null;
  enableSaveButton(false);
  $rating.hidden = true;
  $read.disabled = true;
  setStatus('synthesizing…');
  const t0 = performance.now();
  try {
    const opt = $voice.selectedOptions[0];
    const engine = opt ? opt.dataset.engine : 'xtts';
    const speaker = opt ? opt.dataset.voice : $voice.value;
    const speakerWav = opt ? (opt.dataset.speakerWav || '') : '';
    const speed = parseFloat(document.getElementById('speed').value);

    if (engine === 'xtts') {
      const body = { text, engine, speaker, speed, language: 'en' };
      if (speakerWav) body.speaker_wav = speakerWav;
      const pr = await fetch('/tts/prepare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!pr.ok) throw new Error(await pr.text());
      const { url } = await pr.json();
      await playXttsStream(url, t0);
      refreshHistory();
      showRatingFor('xtts', speaker);
    } else {
      // Gemini: single-shot blob (cloud API returns one complete audio).
      const r = await fetch('/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, engine, speaker, speed }),
      });
      if (!r.ok) throw new Error(await r.text());
      const blob = await r.blob();
      lastAudio = { blob, engine: 'gemini' };
      enableSaveButton(true);
      const url = URL.createObjectURL(blob);
      $out.src = url;
      $out.playbackRate = speed; // Gemini doesn't bake in rate
      await $out.play();
      const dt = (performance.now() - t0) / 1000;
      setStatus(`synthesized in ${dt.toFixed(1)}s (${(blob.size/1024).toFixed(0)} KB)`, 'ok');
      refreshHistory();
      showRatingFor('gemini', speaker);
    }
  } catch (e) {
    setStatus('synth failed: ' + e.message, 'err');
  } finally {
    $read.disabled = false;
  }
}

// Pick a random voice in the dropdown that the user hasn't given any star
// rating yet. Considers every option (custom clones, Gemini, full XTTS list)
// so "unrated" reflects the whole catalog, not just what's currently in view.
function pickUnratedOption() {
  const opts = Array.from($voice.options).filter(o =>
    o.dataset && o.dataset.engine && o.dataset.voice);
  const unrated = opts.filter(o => {
    const rec = (serverScores[o.dataset.engine] || {})[o.dataset.voice];
    return !rec || !Number.isInteger(rec.stars);
  });
  if (!unrated.length) return null;
  return unrated[Math.floor(Math.random() * unrated.length)];
}

document.getElementById('random').addEventListener('click', () => {
  if (!$text.value.trim()) {
    setStatus('paste text first', 'err');
    return;
  }
  const pick = pickUnratedOption();
  if (!pick) {
    setStatus('every voice already rated — try clearing some stars', 'ok');
    return;
  }
  $voice.value = pick.value;
  setStatus(`trying ${pick.dataset.voice} (${pick.dataset.engine})…`);
  read();
});

$read.addEventListener('click', read);
$stop.addEventListener('click', () => {
  stopActiveStream();
  $out.pause(); $out.currentTime = 0;
  $rating.hidden = true;
  setStatus('stopped');
});

document.getElementById('save').addEventListener('click', () => {
  if (!lastAudio) return;
  const ts = new Date().toISOString().replace(/[:T]/g,'-').replace(/\..*/,'');
  const opt = $voice.selectedOptions[0];
  const voiceTag = opt ? opt.dataset.voice.replace(/\s+/g,'_') : 'voice';
  const name = `read_${voiceTag}_${ts}.wav`;
  const url = URL.createObjectURL(lastAudio.blob);
  const a = document.createElement('a');
  a.href = url; a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  setStatus(`saved ${name} (${(lastAudio.blob.size/1024).toFixed(0)} KB)`, 'ok');
});
document.getElementById('clear').addEventListener('click', () => {
  $text.value = '';
  $text.focus();
  // intentionally do not touch $voice — speaker selection persists
  $rating.hidden = true;
  setStatus('cleared');
});

const $speed = document.getElementById('speed');
const $speedVal = document.getElementById('speed_val');
function updateSpeedDisplay() {
  const v = parseFloat($speed.value);
  $speedVal.textContent = v.toFixed(2) + '×';
  // Live-adjust currently playing audio (works regardless of engine; for
  // XTTS-generated audio this compounds with the baked-in speed, but only
  // until the next Read which will re-synth at the new rate).
  if (!$out.paused) $out.playbackRate = v;
}
$speed.addEventListener('input', updateSpeedDisplay);
updateSpeedDisplay();

// Serial: serverScores must arrive before loadVoices so the dropdown sorts
// and labels by star rating. syncScoresToServer (called from inside
// loadVoices) handles its own async work fire-and-forget.
loadServerScores().then(loadVoices);
refreshHistory();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Voice sampler — walk the RECOMMENDED list, each voice says its name and
# a short sentence; user rates with 1–9 via keypress.
# ---------------------------------------------------------------------------

SAMPLER_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#0e0e12">
<title>XTTS voice sampler</title>
<style>
  :root {
    --bg: #0e0e12; --panel: #16161d; --fg: #e6e6ee; --dim: #8a8aa0;
    --accent: #7fff7f; --rule: #2a2a36; --bad: #ff7f7f;
  }
  html, body { margin: 0; padding: 0; background: var(--bg); color: var(--fg);
    font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
    -webkit-text-size-adjust: 100%; }
  body { max-width: 760px; margin: 0 auto; padding: 24px;
    padding-left: max(24px, env(safe-area-inset-left));
    padding-right: max(24px, env(safe-area-inset-right)); }
  h1 { margin: 0 0 4px; font-size: 22px; font-weight: 600; line-height: 1.25; }
  .sub { color: var(--dim); font-size: 13px; margin-bottom: 20px; }
  .stage { background: var(--panel); border: 1px solid var(--rule);
    border-radius: 14px; padding: 28px 24px; }
  .progress { color: var(--dim); font-size: 13px; }
  .voice-name { font-size: 32px; font-weight: 600; margin: 12px 0 4px;
    overflow-wrap: anywhere; }
  .voice-state { color: var(--dim); font-size: 13px; min-height: 18px; }
  .keys { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 22px; }
  .key { background: #0e0e12; border: 1px solid var(--rule); border-radius: 8px;
    padding: 6px 10px; font: 13px ui-monospace, monospace; color: var(--dim); }
  .key b { color: var(--fg); margin-right: 4px; }
  table { width: 100%; border-collapse: collapse; margin-top: 24px; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--rule);
    font-size: 14px; }
  th { color: var(--dim); font-weight: 500; }
  td.score { font-variant-numeric: tabular-nums; }
  .bar { display: inline-block; height: 10px; background: var(--accent);
    border-radius: 3px; vertical-align: middle; margin-left: 8px; }
  .done { color: var(--accent); font-size: 16px; margin-top: 16px; }
  button { background: var(--panel); color: var(--fg); border: 1px solid var(--rule);
    border-radius: 8px; padding: 8px 14px; font: inherit; cursor: pointer;
    min-height: 40px; font-size: 16px; }
  button.primary { background: var(--accent); color: #0e1a0e; border: 0;
    font-weight: 600; }
  .err { color: var(--bad); font-size: 13px; }
  .start-overlay { text-align: center; padding: 30px 0; }

  @media (max-width: 640px) {
    body { padding: 14px;
      padding-left: max(14px, env(safe-area-inset-left));
      padding-right: max(14px, env(safe-area-inset-right)); }
    h1 { font-size: 20px; }
    .stage { padding: 20px 16px; border-radius: 12px; }
    .voice-name { font-size: 26px; }
    /* On phones the numeric keypad isn't visible, so swap key chips for
       large tap targets — two rows of digit buttons. */
    .keys { display: none; }
    table { font-size: 13px; }
    th, td { padding: 6px 4px; }
  }
</style>
</head>
<body>
  <h1 id="headline">XTTS voice sampler</h1>
  <div class="sub" id="subline">Each voice says its name and a short sentence. Rate it 1–9; results land in the table below.</div>
  <div class="sub" id="switchLink" style="margin-top:-12px;"></div>

  <div class="stage" id="stage">
    <div class="start-overlay" id="overlay">
      <button class="primary" id="start">Start sampling</button>
      <div class="sub" style="margin-top:10px;">A click is required so the browser will let audio play.</div>
    </div>

    <div id="active" style="display:none;">
      <div class="progress" id="progress">Voice 0 of 0</div>
      <div class="voice-name" id="voiceName">…</div>
      <div class="voice-state" id="voiceState">loading…</div>
      <div class="keys">
        <span class="key"><b>1–9</b>rate</span>
        <span class="key"><b>Space</b>replay</span>
        <span class="key"><b>N</b>skip</span>
        <span class="key"><b>Esc</b>stop</span>
      </div>
    </div>
  </div>

  <table id="resultsTable" style="display:none;">
    <thead><tr><th>#</th><th>Voice</th><th>Score</th></tr></thead>
    <tbody id="resultsBody"></tbody>
  </table>
  <div id="doneMsg"></div>
  <div class="err" id="errMsg"></div>

<script>
const $overlay    = document.getElementById('overlay');
const $active     = document.getElementById('active');
const $progress   = document.getElementById('progress');
const $voiceName  = document.getElementById('voiceName');
const $voiceState = document.getElementById('voiceState');
const $table      = document.getElementById('resultsTable');
const $body       = document.getElementById('resultsBody');
const $doneMsg    = document.getElementById('doneMsg');
const $errMsg     = document.getElementById('errMsg');

const ENGINE = (new URLSearchParams(window.location.search).get('engine') || 'xtts').toLowerCase();

// Header / switch-engine link
document.getElementById('headline').textContent =
  ENGINE === 'gemini' ? 'Gemini TTS voice sampler' : 'XTTS voice sampler';
document.getElementById('subline').textContent =
  ENGINE === 'gemini'
    ? 'Each Gemini Flash TTS voice says its name and a short sentence. ⚠ each rating round is ~30 cloud API calls against your Gemini key.'
    : 'Each voice says its name and a short sentence. Rate it 1–9; results land in the table below.';
const $sw = document.getElementById('switchLink');
const otherEngine = ENGINE === 'gemini' ? 'xtts' : 'gemini';
$sw.innerHTML = `→ <a href="/sample?engine=${otherEngine}" style="color:#7fff7f;text-decoration:none;">rank ${otherEngine === 'gemini' ? 'Gemini' : 'XTTS'} voices instead</a>`;
document.title = (ENGINE === 'gemini' ? 'Gemini' : 'XTTS') + ' voice sampler';

function sentenceFor(name) {
  return `Hi, my name is ${name}. I can read your text out loud, clearly and at your pace.`;
}

function concatU8(a, b) {
  const out = new Uint8Array(a.length + b.length);
  out.set(a, 0); out.set(b, a.length); return out;
}

function parseWavHeader(buf) {
  if (buf.length < 44) return null;
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  if (dv.getUint32(0, false) !== 0x52494646) return null;
  if (dv.getUint32(8, false) !== 0x57415645) return null;
  const sampleRate = dv.getUint32(24, true);
  let off = 12;
  while (off + 8 <= buf.length) {
    const id = String.fromCharCode(buf[off], buf[off+1], buf[off+2], buf[off+3]);
    const sz = dv.getUint32(off+4, true);
    if (id === 'data') return { sampleRate, dataOffset: off + 8 };
    if (sz === 0xFFFFFFFF) return null;
    off += 8 + sz;
  }
  return null;
}

async function streamAndPlay(ctx, url, abortCtrl) {
  const r = await fetch(url, { signal: abortCtrl.signal });
  if (!r.ok) throw new Error(await r.text());
  const reader = r.body.getReader();
  let header = null, buf = new Uint8Array(0), nextStart = 0, lastSrc = null;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf = concatU8(buf, value);
    if (!header) {
      header = parseWavHeader(buf);
      if (!header) continue;
      buf = buf.subarray(header.dataOffset);
      nextStart = ctx.currentTime + 0.05;
    }
    const evenLen = buf.length - (buf.length % 2);
    if (evenLen < 2) continue;
    const pcm = buf.subarray(0, evenLen);
    buf = buf.subarray(evenLen);
    const ab = pcm.buffer.slice(pcm.byteOffset, pcm.byteOffset + evenLen);
    const samples = new Int16Array(ab);
    const floats = new Float32Array(samples.length);
    for (let i = 0; i < samples.length; i++) floats[i] = samples[i] / 32768;
    const abuf = ctx.createBuffer(1, floats.length, header.sampleRate);
    abuf.getChannelData(0).set(floats);
    const src = ctx.createBufferSource();
    src.buffer = abuf;
    src.connect(ctx.destination);
    const startAt = Math.max(ctx.currentTime + 0.01, nextStart);
    src.start(startAt);
    nextStart = startAt + abuf.duration;
    lastSrc = src;
  }
  if (lastSrc) {
    await new Promise(res => lastSrc.addEventListener('ended', res));
  }
}

function waitForKey() {
  return new Promise(resolve => {
    const handler = (e) => {
      const k = e.key;
      if (/^[1-9]$/.test(k) || k === ' ' || k === 'Escape'
          || k === 'n' || k === 'N') {
        e.preventDefault();
        document.removeEventListener('keydown', handler, true);
        resolve(k);
      }
    };
    document.addEventListener('keydown', handler, true);
  });
}

function renderResults(results, finalized) {
  $table.style.display = '';
  $body.innerHTML = '';
  const sorted = [...results];
  if (finalized) sorted.sort((a, b) => (b.score ?? -1) - (a.score ?? -1));
  for (const r of sorted) {
    const tr = document.createElement('tr');
    const idx = document.createElement('td');
    idx.textContent = results.indexOf(r) + 1;
    const name = document.createElement('td');
    name.textContent = r.voice;
    const score = document.createElement('td');
    score.className = 'score';
    if (r.score === null || r.score === undefined) {
      score.textContent = '—';
    } else {
      score.textContent = r.score;
      const bar = document.createElement('span');
      bar.className = 'bar';
      bar.style.width = (r.score * 12) + 'px';
      score.appendChild(bar);
    }
    tr.appendChild(idx); tr.appendChild(name); tr.appendChild(score);
    $body.appendChild(tr);
  }
}

async function singleShotAndPlay(ctx, body, abortCtrl) {
  // Gemini: POST /tts returns a complete WAV blob in one go.
  const r = await fetch('/tts', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal: abortCtrl.signal,
  });
  if (!r.ok) throw new Error(await r.text());
  const arrayBuf = await r.arrayBuffer();
  // decodeAudioData mutates the input buffer in some browsers; clone to be safe.
  const decoded = await ctx.decodeAudioData(arrayBuf.slice(0));
  const src = ctx.createBufferSource();
  src.buffer = decoded;
  src.connect(ctx.destination);
  src.start();
  await new Promise(res => src.addEventListener('ended', res));
}

function loadVoiceList() {
  return fetch('/voices').then(r => r.json()).then(v => {
    if (ENGINE === 'gemini') {
      if (!v.gemini || !v.gemini.enabled) {
        throw new Error('Gemini backend not enabled (set GEMINI_API_KEY in .env)');
      }
      // Each entry: {name, style}
      return v.gemini.voices.map(x => ({ name: x.name, label: `${x.name} · ${x.style}` }));
    }
    const xtts = (v.xtts && v.xtts.recommended) || [];
    if (!xtts.length) throw new Error('no XTTS recommended voices available');
    return xtts.map(n => ({ name: n, label: n }));
  });
}

async function run(ctx) {
  let voices;
  try {
    voices = await loadVoiceList();
  } catch (e) {
    $errMsg.textContent = 'failed to load voice list: ' + e.message;
    return;
  }

  const results = [];
  for (let i = 0; i < voices.length; i++) {
    const v = voices[i];
    $progress.textContent = `Voice ${i+1} of ${voices.length}`;
    $voiceName.textContent = v.label;
    let entry = { voice: v.name, label: v.label, score: null };
    results.push(entry);
    renderResults(results, false);

    let scored = false;
    while (!scored) {
      $voiceState.textContent = ENGINE === 'gemini' ? 'calling Gemini…' : 'synthesizing…';
      const abortCtrl = new AbortController();
      try {
        if (ENGINE === 'gemini') {
          await singleShotAndPlay(ctx, {
            text: sentenceFor(v.name), engine: 'gemini',
            speaker: v.name, speed: 1.0,
          }, abortCtrl);
        } else {
          const pr = await fetch('/tts/prepare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              text: sentenceFor(v.name), engine: 'xtts',
              speaker: v.name, speed: 1.0, language: 'en',
            }),
          });
          if (!pr.ok) throw new Error(await pr.text());
          const { url } = await pr.json();
          $voiceState.textContent = 'playing… rate 1–9, Space to replay';
          await streamAndPlay(ctx, url, abortCtrl);
        }
      } catch (e) {
        $voiceState.textContent = 'error: ' + e.message;
      }
      $voiceState.textContent = 'rate 1–9 · Space replay · N skip · Esc stop';
      const key = await waitForKey();
      if (/^[1-9]$/.test(key)) {
        entry.score = parseInt(key, 10);
        scored = true;
      } else if (key === 'n' || key === 'N') {
        entry.score = null;
        scored = true;
      } else if (key === 'Escape') {
        saveScores(results);
        renderResults(results, true);
        $doneMsg.className = 'done';
        $doneMsg.textContent = `stopped after ${results.length} voice(s) — scores saved.`;
        $active.style.display = 'none';
        return;
      }
      // Space falls through → replay
      renderResults(results, false);
    }
  }

  $active.style.display = 'none';
  saveScores(results);
  renderResults(results, true);
  $doneMsg.className = 'done';
  $doneMsg.textContent = `Done — ${results.filter(r=>r.score!=null).length} of ${results.length} rated. Scores saved.`;
}

function saveScores(results) {
  // Persist to localStorage so the main page can sort voices by score later.
  try {
    const key = ENGINE === 'gemini' ? 'voiceScores.gemini' : 'voiceScores.xtts';
    const existing = JSON.parse(localStorage.getItem(key) || '{}');
    for (const r of results) {
      if (r.score == null) continue;
      existing[r.voice] = { score: r.score, t: Date.now() };
    }
    localStorage.setItem(key, JSON.stringify(existing));
  } catch (e) {
    console.warn('saveScores failed:', e);
  }
}

document.getElementById('start').addEventListener('click', () => {
  // Create + resume the AudioContext synchronously here so iOS Safari sees
  // it as a user-gesture-initiated context. Awaiting before this point
  // would leave it suspended and produce silent playback on iPhone.
  const Ctx = window.AudioContext || window.webkitAudioContext;
  const ctx = new Ctx();
  if (ctx.state === 'suspended') ctx.resume().catch(() => {});
  $overlay.style.display = 'none';
  $active.style.display = '';
  run(ctx).catch(e => { $errMsg.textContent = 'aborted: ' + e.message; });
});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    # HTTP/1.1 so we can use Transfer-Encoding: chunked for streamed audio.
    # All other routes already send Content-Length, so keep-alive is fine.
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):  # quiet
        pass

    def _write_chunk(self, data: bytes) -> None:
        if not data:
            return
        self.wfile.write(f"{len(data):x}\r\n".encode("ascii"))
        self.wfile.write(data)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    def _end_chunked(self) -> None:
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _is_authed(self) -> bool:
        for part in (self.headers.get("Cookie", "") or "").split(";"):
            k, _, v = part.strip().partition("=")
            if k == AUTH_COOKIE and v == SESSION_TOKEN:
                return True
        return False

    def _send_login(self, error: str = "", status: int = 200) -> None:
        body = LOGIN_HTML.replace("__ERROR__", error).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/login":
            self._send_login()
            return
        if not self._is_authed():
            self._send_login(status=401)
            return
        if self.path.startswith("/tts/stream"):
            self._handle_stream()
            return
        if self.path == "/history":
            try:
                body = json.dumps({"items": list_history()}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                self.send_error(500, f"{type(e).__name__}: {e}")
            return
        if self.path.startswith("/history/"):
            # Only serve <id>.wav; reject anything else (incl. path traversal).
            tail = self.path[len("/history/"):]
            if not tail.endswith(".wav"):
                self.send_error(404)
                return
            uid = tail[:-4]
            if not _history_id_safe(uid):
                self.send_error(400, "bad history id")
                return
            wav_path = HISTORY_DIR / f"{uid}.wav"
            if not wav_path.is_file():
                self.send_error(404, "history entry not found")
                return
            try:
                data = wav_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Accept-Ranges", "none")
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:
                self.send_error(500, f"{type(e).__name__}: {e}")
            return
        if self.path == "/" or self.path.startswith("/index"):
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/sample" or self.path.startswith("/sample?"):
            body = SAMPLER_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/scores":
            try:
                body = json.dumps(load_scores()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                self.send_error(500, f"{type(e).__name__}: {e}")
            return
        if self.path == "/voices":
            try:
                # Load XTTS lazily so first launch isn't slow if user only wants Gemini.
                xtts_recommended: list[str] = []
                xtts_all: list[str] = []
                xtts_err: str | None = None
                try:
                    get_tts()
                    xtts_recommended = [s for s in RECOMMENDED if s in _tts_speakers]
                    xtts_all = _tts_speakers
                except Exception as e:
                    xtts_err = f"{type(e).__name__}: {e}"
                payload = {
                    "device": _tts_device,
                    "xtts": {
                        "recommended": xtts_recommended,
                        "all": xtts_all,
                        "custom": list_custom_voices(),
                        "error": xtts_err,
                    },
                    "gemini": {
                        "enabled": gemini_available(),
                        "model": GEMINI_MODEL,
                        "voices": [{"name": n, "style": s} for n, s in GEMINI_VOICES],
                        "missing": (None if gemini_available() else (
                            "google-genai SDK missing" if "google" not in sys.modules
                            else "GEMINI_API_KEY not set in .env"
                        )),
                    },
                }
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                self.send_error(500, f"{type(e).__name__}: {e}")
            return
        self.send_error(404)

    def _handle_stream(self):
        try:
            qs = parse_qs(urlsplit(self.path).query)
            uid = (qs.get("id") or [""])[0]
            job = _pop_job(uid) if uid else None
            if not job:
                self.send_error(404, "unknown or expired job id")
                return
            payload = job["payload"]
            text = (payload.get("text") or "").strip()
            speaker = (payload.get("speaker")
                       or (RECOMMENDED[0] if RECOMMENDED else ""))
            speaker_wav_path = payload.get("_speaker_wav_path")
            language = payload.get("language") or "en"
            speed = max(0.5, min(2.0, float(payload.get("speed") or 1.0)))
            chunks = split_for_synth(text)
            if not chunks:
                self.send_error(400, "empty text")
                return

            try:
                tts = get_tts()
            except Exception as e:
                self.send_error(500, f"xtts load failed: {e}")
                return
            sr = int(tts.synthesizer.output_sample_rate or 24000)

            import numpy as np
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "close")
            self.end_headers()
            self._write_chunk(_wav_header_unknown_length(1, 2, sr))

            # Buffer the raw PCM so we can persist a proper WAV to history
            # after the stream completes successfully. A 5-minute clip at
            # 24 kHz mono int16 is ~14 MB — safe to hold in memory.
            pcm_parts: list[bytes] = []
            stream_completed = False
            for idx, chunk in enumerate(chunks):
                with _infer_lock:
                    t0 = time.time()
                    try:
                        if speaker_wav_path:
                            cache_dir = VOICES_DIR / ".cache"
                            cache_dir.mkdir(parents=True, exist_ok=True)
                            wav = tts.tts(
                                text=chunk,
                                speaker=Path(speaker_wav_path).stem,
                                speaker_wav=speaker_wav_path,
                                voice_dir=str(cache_dir),
                                language=language, speed=speed,
                            )
                        else:
                            wav = tts.tts(text=chunk, speaker=speaker,
                                          language=language, speed=speed)
                    except Exception as e:
                        print(f"[xtts-stream] chunk {idx} failed: {e}",
                              file=sys.stderr)
                        try:
                            self._end_chunked()
                        except Exception:
                            pass
                        return
                    arr = np.asarray(wav, dtype=np.float32)
                    arr = np.clip(arr, -1.0, 1.0)
                    pcm = (arr * 32767.0).astype(np.int16).tobytes()
                    dt = time.time() - t0
                audio_ms = (len(pcm) / (2 * sr)) * 1000.0
                print(f"[xtts-stream] chunk {idx}: {len(chunk)} chars → "
                      f"{audio_ms:.0f}ms audio in {dt:.2f}s",
                      file=sys.stderr)
                pcm_parts.append(pcm)
                try:
                    self._write_chunk(pcm)
                except (BrokenPipeError, ConnectionResetError):
                    return
            try:
                self._end_chunked()
                stream_completed = True
            except (BrokenPipeError, ConnectionResetError):
                pass

            if stream_completed and pcm_parts:
                try:
                    pcm_all = b"".join(pcm_parts)
                    wav_bytes = _wav_wrap(pcm_all, n_channels=1,
                                          sample_width=2, sample_rate=sr)
                    duration_s = len(pcm_all) / (2 * sr)
                    voice_label = (speaker if not speaker_wav_path
                                   else f"clone:{Path(speaker_wav_path).stem}")
                    save_history(
                        text=text, engine="xtts", voice=voice_label,
                        speed=speed, language=language,
                        wav_bytes=wav_bytes, sample_rate=sr,
                        duration_seconds=duration_s,
                    )
                except Exception as e:
                    print(f"[xtts-stream] history save failed: "
                          f"{type(e).__name__}: {e}", file=sys.stderr)
        except Exception as e:
            # Headers may already be sent — log and bail.
            print(f"[xtts-stream] aborted: {type(e).__name__}: {e}",
                  file=sys.stderr)

    def do_POST(self):
        if self.path == "/auth":
            n = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(n).decode("utf-8", errors="replace")
            fields = parse_qs(body)
            pk = (fields.get("passkey") or [""])[0]
            if secrets.compare_digest(pk, PASSKEY):
                self.send_response(303)
                self.send_header("Location", "/")
                self.send_header(
                    "Set-Cookie",
                    f"{AUTH_COOKIE}={SESSION_TOKEN}; HttpOnly; SameSite=Lax; "
                    f"Path=/; Max-Age=31536000",
                )
                self.send_header("Content-Length", "0")
                self.end_headers()
            else:
                self._send_login(error="invalid passkey", status=401)
            return
        if not self._is_authed():
            self.send_error(401, "auth required")
            return
        if self.path == "/scores":
            try:
                n = int(self.headers.get("Content-Length", "0") or "0")
                incoming = json.loads(self.rfile.read(n) or b"{}")
                merged = merge_scores(incoming)
                body = json.dumps(merged).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                self.send_error(500, f"{type(e).__name__}: {e}")
            return
        if self.path == "/tts/prepare":
            try:
                n = int(self.headers.get("Content-Length", "0") or "0")
                payload = json.loads(self.rfile.read(n) or b"{}")
                if not (payload.get("text") or "").strip():
                    self.send_error(400, "missing 'text'")
                    return
                sw_name = (payload.get("speaker_wav") or "").strip()
                if sw_name:
                    sw_path = resolve_custom_voice(sw_name)
                    if not sw_path:
                        self.send_error(400, f"unknown custom voice: {sw_name}")
                        return
                    payload["_speaker_wav_path"] = str(sw_path)
                uid = _stash_job(payload)
                body = json.dumps({"id": uid,
                                   "url": f"/tts/stream?id={uid}"}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                self.send_error(500, f"{type(e).__name__}: {e}")
            return
        if self.path == "/tts":
            try:
                n = int(self.headers.get("Content-Length", "0") or "0")
                payload = json.loads(self.rfile.read(n) or b"{}")
                text = (payload.get("text") or "").strip()
                engine = payload.get("engine") or "xtts"
                speaker = payload.get("speaker") or ""
                sw_name = (payload.get("speaker_wav") or "").strip()
                language = payload.get("language") or "en"
                speed = float(payload.get("speed") or 1.0)
                if not text:
                    self.send_error(400, "missing 'text'"); return
                voice_label = speaker
                hist_sr = 24000
                if engine == "gemini":
                    if not gemini_available():
                        self.send_error(400, "Gemini backend unavailable "
                                             "(set GEMINI_API_KEY in .env).")
                        return
                    # Gemini Flash TTS doesn't expose a rate param yet; the
                    # browser applies `speed` to <audio>.playbackRate instead.
                    wav = synth_gemini(text, voice=speaker or "Aoede")
                    voice_label = speaker or "Aoede"
                else:
                    sw_path: str | None = None
                    if sw_name:
                        rp = resolve_custom_voice(sw_name)
                        if not rp:
                            self.send_error(400, f"unknown custom voice: {sw_name}")
                            return
                        sw_path = str(rp)
                    wav = synth_wav(
                        text,
                        speaker=(speaker or (RECOMMENDED[0] if RECOMMENDED else None)),
                        language=language,
                        speed=speed,
                        speaker_wav=sw_path,
                    )
                    if sw_path:
                        voice_label = f"clone:{Path(sw_path).stem}"
                    else:
                        voice_label = (speaker
                                       or (RECOMMENDED[0] if RECOMMENDED else ""))
                    try:
                        hist_sr = int(get_tts().synthesizer.output_sample_rate
                                      or 24000)
                    except Exception:
                        hist_sr = 24000
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(wav)))
                self.end_headers()
                self.wfile.write(wav)
                try:
                    # WAV header is 44 bytes; the rest is int16 mono PCM.
                    pcm_len = max(0, len(wav) - 44)
                    duration_s = pcm_len / (2 * hist_sr) if hist_sr else 0.0
                    save_history(
                        text=text, engine=engine, voice=voice_label,
                        speed=speed, language=language,
                        wav_bytes=wav, sample_rate=hist_sr,
                        duration_seconds=duration_s,
                    )
                except Exception as e:
                    print(f"[tts] history save failed: "
                          f"{type(e).__name__}: {e}", file=sys.stderr)
            except Exception as e:
                self.send_error(500, f"{type(e).__name__}: {e}")
            return
        self.send_error(404)


def _pick_top_xtts_voice() -> str | None:
    """Highest-rated XTTS voice in voice_scores.json, or None if no ratings."""
    scores = (load_scores().get("xtts") or {})
    rated = [(v, rec.get("score", 0)) for v, rec in scores.items()
             if isinstance(rec, dict) and isinstance(rec.get("score"), (int, float))]
    if not rated:
        return None
    rated.sort(key=lambda kv: kv[1], reverse=True)
    return rated[0][0]


def cmd_batch(argv: list[str]) -> int:
    """Read a whole .txt file with one XTTS voice into a single .wav file.

    Splits text into the same sentence-sized chunks the streaming path uses,
    synthesizes each, and appends raw int16 PCM to a wave.Wave_write so the
    output is one continuous file regardless of input length.
    """
    import argparse, wave
    p = argparse.ArgumentParser(
        prog="read_xtts.py batch",
        description="Synthesize a whole text file to a single WAV (XTTS-v2).",
    )
    p.add_argument("-i", "--input", required=True, type=Path,
                   help="Input text file (UTF-8).")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output WAV (default: <input stem>.wav alongside input).")
    p.add_argument("-v", "--voice", default=None,
                   help="XTTS speaker name. Default: highest-rated, else "
                        f"{RECOMMENDED[0]!r}.")
    p.add_argument("--language", default="en")
    p.add_argument("--speed", type=float, default=1.0,
                   help="XTTS rate multiplier, 0.5–2.0 (default 1.0).")
    p.add_argument("--list-voices", action="store_true",
                   help="Print rated + recommended voices and exit.")
    args = p.parse_args(argv)

    if args.list_voices:
        scores = (load_scores().get("xtts") or {})
        rated = sorted(((v, r.get("score", 0)) for v, r in scores.items()
                        if isinstance(r, dict)), key=lambda kv: kv[1], reverse=True)
        if rated:
            print("rated voices (high → low):")
            for v, s in rated:
                print(f"  ★{s}  {v}")
        else:
            print("no rated voices yet — run the web UI's /sample to rate some.")
        print("\nrecommended (curated):")
        for v in RECOMMENDED:
            print(f"  ☆   {v}")
        return 0

    if not args.input.exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2
    text = args.input.read_text(encoding="utf-8")
    chunks = split_for_synth(text)
    if not chunks:
        print("error: no synthesizable text in input", file=sys.stderr)
        return 2

    voice = (args.voice
             or _pick_top_xtts_voice()
             or (RECOMMENDED[0] if RECOMMENDED else None))
    if not voice:
        print("error: no voice specified and no recommended fallback",
              file=sys.stderr)
        return 2

    out_path: Path = args.output or args.input.with_suffix(".wav")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tts = get_tts()
    if voice not in _tts_speakers:
        print(f"error: voice {voice!r} not in this XTTS model. Try "
              f"--list-voices.", file=sys.stderr)
        return 2
    sr = int(tts.synthesizer.output_sample_rate or 24000)
    speed = max(0.5, min(2.0, float(args.speed)))

    import numpy as np
    total_chars = sum(len(c) for c in chunks)
    print(f"[batch] voice={voice!r} · {len(chunks)} chunks · "
          f"{total_chars} chars · sr={sr} Hz · → {out_path}",
          file=sys.stderr)

    t_start = time.time()
    total_audio_s = 0.0
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        for idx, chunk in enumerate(chunks, start=1):
            t0 = time.time()
            try:
                wav = tts.tts(text=chunk, speaker=voice,
                              language=args.language, speed=speed)
            except Exception as e:
                print(f"[batch] chunk {idx}/{len(chunks)} failed: "
                      f"{type(e).__name__}: {e}", file=sys.stderr)
                return 1
            arr = np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0)
            pcm = (arr * 32767.0).astype(np.int16).tobytes()
            wf.writeframes(pcm)
            audio_s = len(pcm) / (2 * sr)
            total_audio_s += audio_s
            dt = max(time.time() - t0, 1e-6)
            print(f"[batch] {idx:4d}/{len(chunks)}  "
                  f"{len(chunk):4d} chars → {audio_s:5.1f}s audio "
                  f"in {dt:4.1f}s ({audio_s/dt:4.1f}× rt)",
                  file=sys.stderr)

    elapsed = max(time.time() - t_start, 1e-6)
    print(f"[batch] done: {total_audio_s/60:.1f} min audio in "
          f"{elapsed/60:.1f} min ({total_audio_s/elapsed:.1f}× rt) → "
          f"{out_path}", file=sys.stderr)
    return 0


def main() -> int:
    # Subcommand split: `read_xtts.py batch ...` runs offline file→wav;
    # everything else (including no args) runs the web server.
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        return cmd_batch(sys.argv[2:])

    import argparse, socket
    p = argparse.ArgumentParser(description="Local XTTS-v2 read-aloud web app.")
    p.add_argument("--network", action=argparse.BooleanOptionalAction, default=True,
                   help="Bind to 0.0.0.0 (default). Use --no-network for localhost only.")
    p.add_argument("--port", type=int,
                   default=int(os.environ.get("READ_XTTS_PORT", "8888")))
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--preload", action=argparse.BooleanOptionalAction, default=True,
                   help="Load the XTTS-v2 model at startup (default).")
    args = p.parse_args()

    bind = "0.0.0.0" if args.network else "127.0.0.1"
    srv = ThreadingHTTPServer((bind, args.port), Handler)
    port = srv.server_address[1]
    url_local = f"http://127.0.0.1:{port}/"
    print(f"Read aloud (XTTS-v2) running.")
    if args.network:
        try:
            ip = next((info[4][0] for info in socket.getaddrinfo(socket.gethostname(), port, socket.AF_INET)
                       if not info[4][0].startswith("127.")), None)
        except Exception:
            ip = None
        print(f"  local: {url_local}")
        if ip:
            print(f"  LAN:   http://{ip}:{port}/")
        print("  \033[33m⚠ bound to 0.0.0.0 — anyone on the network can synthesize.\033[0m")
    else:
        print(f"  url:   {url_local}  (localhost only; use --network for LAN)")

    if args.preload:
        threading.Thread(target=get_tts, daemon=True).start()

    if not args.no_browser:
        try:
            webbrowser.open(url_local)
        except Exception:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        srv.shutdown()
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
