#!/usr/bin/env python3
"""Narrate each Greece photo caption to a WAV with one pleasant XTTS voice.

Reuses the local Coqui XTTS-v2 model from the sibling debate-out-loud project
(loaded once, in-process) and the photo ordering from build_gallery, so each
audio file lines up with its thumbnail/caption by stem.

Output: captions/audio/<stem>.wav, one per captioned photo. Re-runnable:
existing audio is skipped, so an interrupted run continues where it left off.
Run with the XTTS venv:

  /home/mjbernaski/projects/debate-out-loud/.venv/bin/python3 narrate_captions.py

Stop the read-xtts service first so two copies of the model don't fight for
the GPU.
"""
from __future__ import annotations

import sys
import wave
from pathlib import Path

# Pull in the XTTS synthesis machinery from the sibling project.
XTTS_DIR = Path("/home/mjbernaski/projects/debate-out-loud")
sys.path.insert(0, str(XTTS_DIR))

import photos as photo_src  # shared photo ordering across all folders
import read_xtts as rx       # get_tts / split_for_synth / voice helpers

OUT_DIR = Path("/home/mjbernaski/projects/omni-play/captions")
AUDIO_DIR = OUT_DIR / "audio"


def pick_voice(requested: str | None) -> str:
    """Resolve the single narration voice: CLI arg → top-rated → recommended."""
    voice = (requested
             or rx._pick_top_xtts_voice()
             or (rx.RECOMMENDED[0] if rx.RECOMMENDED else None))
    if not voice:
        sys.exit("no voice available (no rating, no RECOMMENDED fallback)")
    return voice


def synth_to_wav(tts, voice, text, out_path, sr, language="en"):
    """Synthesize `text` (split into sentence chunks) into one mono WAV.

    Writes to a .part file and renames on success so an interrupt never leaves
    a truncated WAV that a later run would skip as done.
    """
    import numpy as np

    chunks = rx.split_for_synth(text)
    if not chunks:
        return False
    tmp = out_path.with_suffix(".wav.part")
    with wave.open(str(tmp), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        for chunk in chunks:
            wav = tts.tts(text=chunk, speaker=voice, language=language)
            arr = np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0)
            wf.writeframes((arr * 32767.0).astype(np.int16).tobytes())
    tmp.replace(out_path)
    return True


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-v", "--voice", default=None,
                    help="XTTS speaker (default: highest-rated, else "
                         f"{rx.RECOMMENDED[0]!r}).")
    ap.add_argument("--language", default="en")
    args = ap.parse_args()

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    voice = pick_voice(args.voice)

    photos = photo_src.list_photos()
    tts = rx.get_tts()
    if voice not in rx._tts_speakers:
        sys.exit(f"voice {voice!r} not in this XTTS model "
                 f"(see read_xtts.py batch --list-voices)")
    sr = int(tts.synthesizer.output_sample_rate or 24000)
    print(f"voice: {voice!r} · sr={sr} Hz · → {AUDIO_DIR}")

    made = skipped = nocap = 0
    for i, rec in enumerate(photos, 1):
        stem = rec["stem"]
        out_path = AUDIO_DIR / (stem + ".wav")
        if out_path.exists():
            skipped += 1
            continue
        cap_file = OUT_DIR / (stem + ".txt")
        text = cap_file.read_text().strip() if cap_file.exists() else ""
        if not text:
            nocap += 1
            continue
        try:
            if synth_to_wav(tts, voice, text, out_path, sr, args.language):
                made += 1
                print(f"  [{i}/{len(photos)}] {stem}.wav")
        except KeyboardInterrupt:
            print("\ninterrupted — re-run to continue")
            break
        except Exception as e:
            print(f"  ! {stem}: {type(e).__name__}: {e}")

    print(f"\ndone: {made} made, {skipped} already present, "
          f"{nocap} without caption")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
