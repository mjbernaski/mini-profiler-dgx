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
import struct
import sys
import threading
import time
import uuid
import webbrowser
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
# Lazy XTTS-v2 loader. Loading takes ~10s; we keep a single instance warm.
# ---------------------------------------------------------------------------

_tts_lock = threading.Lock()
_tts = None
_tts_device: str = "cpu"
_tts_speakers: list[str] = []


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
        try:
            _tts_speakers = list(_tts.synthesizer.tts_model.speaker_manager.name_to_id.keys())
        except Exception:
            _tts_speakers = list(getattr(_tts, "speakers", []) or [])
        print(f"[xtts] loaded in {time.time()-t0:.1f}s; "
              f"{len(_tts_speakers)} built-in speakers", file=sys.stderr)
    return _tts


def synth_wav(text: str, speaker: str, language: str = "en",
              speed: float = 1.0) -> bytes:
    """Synthesize `text` with `speaker`, return WAV bytes (mono 24 kHz int16).

    `speed` is XTTS-v2's native rate multiplier (1.0 = default). Clamped to
    a sensible range so we don't ship garbled audio.
    """
    tts = get_tts()
    import numpy as np
    s = max(0.5, min(2.0, float(speed)))
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


def synth_gemini(text: str, voice: str = "Aoede") -> bytes:
    """Synthesize via gemini-3.1-flash-tts-preview. Returns WAV bytes."""
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice,
                    ),
                ),
            ),
        ),
    )
    # The audio comes back as raw PCM in inline_data. Gemini TTS returns
    # mono 24 kHz signed 16-bit PCM as of writing.
    parts = resp.candidates[0].content.parts
    pcm: bytes | None = None
    for p in parts:
        d = getattr(p, "inline_data", None)
        if d and d.data:
            pcm = d.data if isinstance(d.data, (bytes, bytearray)) else bytes(d.data)
            break
    if pcm is None:
        raise RuntimeError("Gemini response had no inline audio data")
    return _wav_wrap(pcm, n_channels=1, sample_width=2, sample_rate=24000)


# ---------------------------------------------------------------------------
# HTML/JS UI — paste, pick a voice, hit Read.
# ---------------------------------------------------------------------------

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
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
    font-family: -apple-system, "Segoe UI", system-ui, sans-serif; }
  body { max-width: 980px; margin: 0 auto; padding: 24px; }
  h1 { margin: 0 0 4px; font-size: 22px; font-weight: 600; }
  .sub { color: var(--dim); font-size: 13px; margin-bottom: 16px; }
  textarea { width: 100%; min-height: 180px; resize: vertical;
    background: var(--panel); color: var(--fg); border: 1px solid var(--rule);
    border-radius: 10px; padding: 12px 14px; font: inherit; font-size: 15px;
    line-height: 1.4; box-sizing: border-box; }
  textarea:focus { outline: 2px solid var(--accent); border-color: transparent; }
  .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
    margin-top: 12px; }
  select, button { background: var(--panel); color: var(--fg);
    border: 1px solid var(--rule); border-radius: 8px; padding: 9px 14px;
    font: inherit; }
  button.primary { background: var(--accent); color: #0e1a0e; border: 0;
    font-weight: 600; cursor: pointer; }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  .status { color: var(--dim); font-size: 13px; margin-left: 8px; }
  .status.ok { color: var(--accent); }
  .status.err { color: var(--bad); }
  audio { width: 100%; max-width: 600px; margin-top: 14px; display: block; }
  footer { color: var(--dim); font-size: 12px; margin-top: 18px; }
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

// Sort an engine's voices: rated (high → low), then recommended (XTTS only),
// then original list order. Returns augmented items with .score / .rec.
function sortVoices(items, scores, recSet) {
  const augmented = items.map((it, idx) => ({
    ...it,
    score: (scores[it.name] && typeof scores[it.name].score === 'number')
              ? scores[it.name].score : null,
    rec: recSet ? recSet.has(it.name) : false,
    origIdx: idx,
  }));
  augmented.sort((a, b) => {
    const sa = a.score == null ? -Infinity : a.score;
    const sb = b.score == null ? -Infinity : b.score;
    if (sb !== sa) return sb - sa;
    if (a.rec !== b.rec) return a.rec ? -1 : 1;
    return a.origIdx - b.origIdx;
  });
  return augmented;
}

function labelFor(item) {
  if (item.score != null) return `${item.label} · ★${item.score}`;
  if (item.rec)           return `${item.label} · ☆`;
  return item.label;
}

async function loadVoices() {
  try {
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
    let topXtts = null, topGemini = null;

    if (data.gemini.enabled) {
      const og = document.createElement('optgroup');
      og.label = 'Gemini Flash TTS (cloud)';
      const items = data.gemini.voices.map(v => ({
        name: v.name, label: `${v.name} · ${v.style}`,
      }));
      const sorted = sortVoices(items, geminiScores, null);
      for (const it of sorted) {
        og.appendChild(makeOption('gemini', it.name, labelFor(it)));
      }
      $voice.appendChild(og);
      topGemini = sorted[0] || null;
    }

    if (data.xtts.all && data.xtts.all.length) {
      const og = document.createElement('optgroup');
      og.label = 'XTTS-v2 local';
      const items = data.xtts.all.map(n => ({ name: n, label: n }));
      const recSet = new Set(data.xtts.recommended || []);
      const sorted = sortVoices(items, xttsScores, recSet);
      for (const it of sorted) {
        og.appendChild(makeOption('xtts', it.name, labelFor(it)));
      }
      $voice.appendChild(og);
      topXtts = sorted[0] || null;
    }

    if (data.xtts.error) setStatus('xtts: ' + data.xtts.error, 'err');

    // Default selection: highest-rated voice across engines, tiebreaker XTTS
    // (local, free). With no ratings yet, both scores tie at -Infinity and
    // XTTS wins, so first-time users land on an XTTS voice as before.
    const candidates = [];
    if (topXtts)   candidates.push({ engine: 'xtts',   it: topXtts });
    if (topGemini) candidates.push({ engine: 'gemini', it: topGemini });
    candidates.sort((a, b) => {
      const sa = a.it.score == null ? -Infinity : a.it.score;
      const sb = b.it.score == null ? -Infinity : b.it.score;
      if (sb !== sa) return sb - sa;
      if (a.engine !== b.engine) return a.engine === 'xtts' ? -1 : 1;
      return 0;
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
  const Ctx = window.AudioContext || window.webkitAudioContext;
  const ctx = new Ctx();
  if (ctx.state === 'suspended') {
    try { await ctx.resume(); } catch (e) {}
  }
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

async function read() {
  const text = $text.value.trim();
  if (!text) { setStatus('paste text first', 'err'); return; }
  stopActiveStream();
  $out.pause();
  lastAudio = null;
  enableSaveButton(false);
  $read.disabled = true;
  setStatus('synthesizing…');
  const t0 = performance.now();
  try {
    const opt = $voice.selectedOptions[0];
    const engine = opt ? opt.dataset.engine : 'xtts';
    const speaker = opt ? opt.dataset.voice : $voice.value;
    const speed = parseFloat(document.getElementById('speed').value);

    if (engine === 'xtts') {
      const pr = await fetch('/tts/prepare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, engine, speaker, speed, language: 'en' }),
      });
      if (!pr.ok) throw new Error(await pr.text());
      const { url } = await pr.json();
      await playXttsStream(url, t0);
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
    }
  } catch (e) {
    setStatus('synth failed: ' + e.message, 'err');
  } finally {
    $read.disabled = false;
  }
}

$read.addEventListener('click', read);
$stop.addEventListener('click', () => {
  stopActiveStream();
  $out.pause(); $out.currentTime = 0;
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

loadVoices();
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
<title>XTTS voice sampler</title>
<style>
  :root {
    --bg: #0e0e12; --panel: #16161d; --fg: #e6e6ee; --dim: #8a8aa0;
    --accent: #7fff7f; --rule: #2a2a36; --bad: #ff7f7f;
  }
  html, body { margin: 0; padding: 0; background: var(--bg); color: var(--fg);
    font-family: -apple-system, "Segoe UI", system-ui, sans-serif; }
  body { max-width: 760px; margin: 0 auto; padding: 24px; }
  h1 { margin: 0 0 4px; font-size: 22px; font-weight: 600; }
  .sub { color: var(--dim); font-size: 13px; margin-bottom: 20px; }
  .stage { background: var(--panel); border: 1px solid var(--rule);
    border-radius: 14px; padding: 28px 24px; }
  .progress { color: var(--dim); font-size: 13px; }
  .voice-name { font-size: 32px; font-weight: 600; margin: 12px 0 4px; }
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
    border-radius: 8px; padding: 8px 14px; font: inherit; cursor: pointer; }
  button.primary { background: var(--accent); color: #0e1a0e; border: 0;
    font-weight: 600; }
  .err { color: var(--bad); font-size: 13px; }
  .start-overlay { text-align: center; padding: 30px 0; }
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

async function run() {
  let voices;
  try {
    voices = await loadVoiceList();
  } catch (e) {
    $errMsg.textContent = 'failed to load voice list: ' + e.message;
    return;
  }

  const Ctx = window.AudioContext || window.webkitAudioContext;
  const ctx = new Ctx();
  if (ctx.state === 'suspended') {
    try { await ctx.resume(); } catch (e) {}
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
  $overlay.style.display = 'none';
  $active.style.display = '';
  run().catch(e => { $errMsg.textContent = 'aborted: ' + e.message; });
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

    def do_GET(self):
        if self.path.startswith("/tts/stream"):
            self._handle_stream()
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

            for idx, chunk in enumerate(chunks):
                with _infer_lock:
                    t0 = time.time()
                    try:
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
                try:
                    self._write_chunk(pcm)
                except (BrokenPipeError, ConnectionResetError):
                    return
            try:
                self._end_chunked()
            except (BrokenPipeError, ConnectionResetError):
                pass
        except Exception as e:
            # Headers may already be sent — log and bail.
            print(f"[xtts-stream] aborted: {type(e).__name__}: {e}",
                  file=sys.stderr)

    def do_POST(self):
        if self.path == "/tts/prepare":
            try:
                n = int(self.headers.get("Content-Length", "0") or "0")
                payload = json.loads(self.rfile.read(n) or b"{}")
                if not (payload.get("text") or "").strip():
                    self.send_error(400, "missing 'text'")
                    return
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
                language = payload.get("language") or "en"
                speed = float(payload.get("speed") or 1.0)
                if not text:
                    self.send_error(400, "missing 'text'"); return
                if engine == "gemini":
                    if not gemini_available():
                        self.send_error(400, "Gemini backend unavailable "
                                             "(set GEMINI_API_KEY in .env).")
                        return
                    # Gemini Flash TTS doesn't expose a rate param yet; the
                    # browser applies `speed` to <audio>.playbackRate instead.
                    wav = synth_gemini(text, voice=speaker or "Aoede")
                else:
                    wav = synth_wav(text,
                                    speaker=(speaker or (RECOMMENDED[0] if RECOMMENDED else None)),
                                    language=language,
                                    speed=speed)
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(wav)))
                self.end_headers()
                self.wfile.write(wav)
            except Exception as e:
                self.send_error(500, f"{type(e).__name__}: {e}")
            return
        self.send_error(404)


def main() -> int:
    import argparse, socket
    p = argparse.ArgumentParser(description="Local XTTS-v2 read-aloud web app.")
    p.add_argument("--network", action="store_true",
                   help="Bind to 0.0.0.0 (reachable on LAN).")
    p.add_argument("--port", type=int, default=int(os.environ.get("READ_XTTS_PORT", "0")))
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--preload", action="store_true",
                   help="Load the XTTS-v2 model at startup (instead of on first request).")
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
