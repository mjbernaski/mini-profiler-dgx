#!/usr/bin/env python3
"""Caption the Greece trip photos with Nemotron-3-Nano-Omni.

Pipeline: enumerate JPEGs -> downscale -> base64 -> POST to the local Omni
server (OpenAI-compatible) -> keep the answer paragraph (drop the <think>
trace) -> write one .txt per image + a captions.md + a metrics.json.

Test run:   python3 caption_greece.py --limit 5 --shuffle --seed 1
Full run:   python3 caption_greece.py
"""
import argparse
import base64
import io
import json
import re
import sys
import time
from pathlib import Path

from PIL import Image, ImageOps
from openai import OpenAI

import photos as photo_src

OUT_DIR = Path("/home/mjbernaski/projects/omni-play/captions")
ENDPOINT = "http://localhost:8355/v1"

# Trip context steers the model away from generic guesses toward the real places.
SYSTEM_PROMPT = (
    "You are writing captions for a personal travel photo album. The photos are "
    "from a trip to Greece: Attica (including Athens), the Peloponnese, and the "
    "island of Rhodes. Sites likely include the Acropolis, ancient ruins, "
    "coastal villages, medieval Rhodes Old Town, and Mediterranean landscapes."
)
USER_PROMPT = (
    "Write one vivid, evocative paragraph (4-6 sentences) describing this photo "
    "for the album. Describe what is actually visible -- the setting, light, "
    "architecture or landscape, people, and mood. If you recognize a specific "
    "Greek landmark or place, name it; otherwise do not invent specifics. "
    "Output only the paragraph, no preamble or title."
)


def downscale_to_b64(path, max_side, quality):
    """Load, EXIF-rotate, downscale longest side to max_side, re-encode JPEG."""
    t0 = time.time()
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        orig = im.size
        im.thumbnail((max_side, max_side), Image.LANCZOS)
        new = im.size
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality)
    raw = buf.getvalue()
    b64 = base64.b64encode(raw).decode()
    return b64, {
        "orig_px": orig,
        "sent_px": new,
        "sent_kb": round(len(raw) / 1024, 1),
        "resize_s": round(time.time() - t0, 2),
    }


def caption_one(client, model, path, args):
    b64, meta = downscale_to_b64(path, args.max_side, args.quality)
    data_uri = f"data:image/jpeg;base64,{b64}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": USER_PROMPT},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ]},
    ]

    t0 = time.time()
    ttft = None
    reasoning_chars = 0
    answer = []
    usage = None
    finish = None
    stream = client.chat.completions.create(
        model=model, messages=messages, stream=True,
        stream_options={"include_usage": True},
        temperature=args.temperature, top_p=0.95, max_tokens=args.max_tokens,
    )
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if not chunk.choices:
            continue
        if chunk.choices[0].finish_reason:
            finish = chunk.choices[0].finish_reason
        delta = chunk.choices[0].delta
        rc = getattr(delta, "reasoning_content", None)
        if rc:
            if ttft is None:
                ttft = time.time() - t0
            reasoning_chars += len(rc)
        if delta.content:
            if ttft is None:
                ttft = time.time() - t0
            answer.append(delta.content)
    dt = time.time() - t0

    text = "".join(answer).strip()
    comp_tok = getattr(usage, "completion_tokens", None) if usage else None
    prompt_tok = getattr(usage, "prompt_tokens", None) if usage else None
    metrics = {
        **meta,
        "ttft_s": round(ttft, 2) if ttft else None,
        "total_s": round(dt, 2),
        "prompt_tokens": prompt_tok,
        "completion_tokens": comp_tok,
        "reasoning_chars": reasoning_chars,
        "tok_per_s": round(comp_tok / dt, 1) if comp_tok and dt else None,
        "finish_reason": finish,
        "truncated": finish == "length" or not text,
    }
    return text, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap number of images (0 = all)")
    ap.add_argument("--shuffle", action="store_true", help="random sample instead of in-order")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-side", type=int, default=1568, help="longest image edge sent to model")
    ap.add_argument("--quality", type=int, default=88)
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="must cover the <think> trace AND the answer for this reasoning model")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--endpoint", default=ENDPOINT)
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--force", action="store_true",
                    help="re-caption photos that already have a .txt")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_photos = [rec["path"] for rec in photo_src.list_photos()]
    # Resumable by default: skip photos that already have a non-empty caption.
    if args.force:
        photos = all_photos
    else:
        photos = [p for p in all_photos
                  if not (out_dir / (p.stem + ".txt")).exists()
                  or not (out_dir / (p.stem + ".txt")).read_text().strip()]
    skipped = len(all_photos) - len(photos)
    if skipped:
        print(f"skipping {skipped} already-captioned photos "
              f"(use --force to redo)")
    if args.shuffle:
        import random
        random.Random(args.seed).shuffle(photos)
    if args.limit:
        photos = photos[:args.limit]

    client = OpenAI(base_url=args.endpoint, api_key="not-needed")
    try:
        model = client.models.list().data[0].id
    except Exception as e:
        sys.exit(f"Cannot reach Omni server at {args.endpoint}: {e}\n"
                 f"Start it with ~/projects/trtllm-spark/serve-nemotron-omni.sh")

    print(f"model    : {model}")
    print(f"images   : {len(photos)}  (max_side={args.max_side}, max_tokens={args.max_tokens})")
    print(f"out dir  : {out_dir}\n")

    results = []
    md_lines = ["# Greece 2026 — Photo Captions\n"]
    wall0 = time.time()
    for i, path in enumerate(photos, 1):
        print(f"[{i}/{len(photos)}] {path.name} ...", end=" ", flush=True)
        try:
            text, m = caption_one(client, model, path, args)
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({"file": path.name, "error": str(e)})
            continue
        (out_dir / (path.stem + ".txt")).write_text(text + "\n")
        md_lines.append(f"## {path.name}\n\n{text}\n")
        m = {"file": path.name, **m}
        results.append(m)
        flag = "  !!TRUNCATED/EMPTY" if m["truncated"] else ""
        print(f"{m['total_s']}s  ttft {m['ttft_s']}s  "
              f"{m['completion_tokens']}tok  {m['tok_per_s']}tok/s  "
              f"({m['orig_px'][0]}x{m['orig_px'][1]}->{m['sent_px'][0]}x{m['sent_px'][1]}, {m['sent_kb']}KB){flag}")
    wall = time.time() - wall0

    # Rebuild captions.md from every caption on disk (not just this run's),
    # so it stays a complete album even when most photos were skipped.
    md = ["# Greece 2026 — Photo Captions\n"]
    for p in all_photos:
        cf = out_dir / (p.stem + ".txt")
        if cf.exists() and cf.read_text().strip():
            md.append(f"## {p.name}\n\n{cf.read_text().strip()}\n")
    (out_dir / "captions.md").write_text("\n".join(md))

    # Aggregate metrics over successful captions.
    ok = [r for r in results if "error" not in r]
    def total(k): return sum(r[k] for r in ok if r.get(k))
    def avg(k):
        vals = [r[k] for r in ok if r.get(k)]
        return round(sum(vals) / len(vals), 2) if vals else None
    summary = {
        "model": model,
        "images_attempted": len(photos),
        "images_ok": len(ok),
        "images_failed": len(photos) - len(ok),
        "wall_clock_s": round(wall, 1),
        "total_completion_tokens": total("completion_tokens"),
        "total_prompt_tokens": total("prompt_tokens"),
        "overall_tok_per_s": round(total("completion_tokens") / wall, 1) if wall else None,
        "avg_ttft_s": avg("ttft_s"),
        "avg_total_s": avg("total_s"),
        "avg_tok_per_s": avg("tok_per_s"),
        "avg_completion_tokens": avg("completion_tokens"),
        "settings": {"max_side": args.max_side, "quality": args.quality,
                     "max_tokens": args.max_tokens, "temperature": args.temperature},
        "per_image": results,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    print("\n=== summary ===")
    for k in ("images_ok", "images_failed", "wall_clock_s", "total_completion_tokens",
              "overall_tok_per_s", "avg_ttft_s", "avg_total_s", "avg_tok_per_s",
              "avg_completion_tokens"):
        print(f"  {k:24} {summary[k]}")
    print(f"\nwrote: {out_dir}/captions.md, metrics.json, and {len(ok)} .txt files")


if __name__ == "__main__":
    main()
