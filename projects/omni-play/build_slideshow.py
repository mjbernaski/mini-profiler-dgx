#!/usr/bin/env python3
"""Build a self-running slideshow of the Greece photos, duplicates removed.

Each slide shows the high-resolution photo on the left and its caption on the
right, then plays the narrated WAV; when the audio ends the show advances to
the next photo on its own. (Browsers block autoplay until a user gesture, so
the page opens with a Start button — after that one click, narration and
advancing run hands-free.)

Pipeline reuse:
  * photos.py            -> the canonical photo set + trip order + group labels
  * captions/<stem>.txt  -> caption text (built by caption_greece.py)
  * captions/thumbs/     -> existing web thumbnails, hashed for dedup
  * captions/audio/      -> narration WAVs (built by narrate_captions.py)

Duplicate removal: a difference-hash (dHash) is computed from each thumbnail;
photos whose hash is within --max-distance bits of one already kept are dropped
(first occurrence in trip order wins). DSC_0270/DSC_0270_1 etc. fall out here.

High-res display images are rendered from the raw photos into
captions/slideshow_img/ (longest edge --img-max, default 2200px) only for the
surviving slides. Re-runnable: existing display images are skipped.

Output: captions/slideshow.html — self-contained, served as-is by serve.py.

  python3 build_slideshow.py                 # uses repo's PIL
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from PIL import Image, ImageOps

import photos as photo_src

OUT_DIR = Path("/home/mjbernaski/projects/omni-play/captions")
THUMB_DIR = OUT_DIR / "thumbs"
AUDIO_DIR = OUT_DIR / "audio"
IMG_DIR = OUT_DIR / "slideshow_img"          # high-res display images
MODEL_NAME = "Nemotron-3-Nano-Omni"          # caption author, shown in footer


# ---------------------------------------------------------------- dedup -----
def dhash(path: Path, size: int = 8) -> int:
    """64-bit difference hash: compares each pixel to its right neighbour on a
    (size+1)x size grayscale downscale. Robust to re-encoding/resizing, so it
    catches genuine duplicates without flagging merely similar burst shots."""
    with Image.open(path) as im:
        im = im.convert("L").resize((size + 1, size), Image.LANCZOS)
        px = list(im.getdata())
    bits = 0
    for row in range(size):
        base = row * (size + 1)
        for col in range(size):
            bits = (bits << 1) | int(px[base + col] < px[base + col + 1])
    return bits


def is_dupe(h: int, kept: list[int], max_distance: int) -> bool:
    return any((h ^ k).bit_count() <= max_distance for k in kept)


# ------------------------------------------------------------ hi-res img ----
def make_display(src: Path, dst: Path, img_max: int) -> None:
    """Render a high-res display JPEG, skipping if already present. Writes to a
    .part file and renames so an interrupt never leaves a truncated image."""
    if dst.exists():
        return
    tmp = dst.with_suffix(dst.suffix + ".part")
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((img_max, img_max), Image.LANCZOS)
        im.save(tmp, format="JPEG", quality=88)
    tmp.replace(dst)


# --------------------------------------------------------------- page -------
PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Greece 2026 — Slideshow</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  html, body {{ height: 100%; margin: 0; }}
  body {{ background: #0e0f13; color: #e6e7ea; overflow: hidden;
         font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
  #stage {{ display: flex; height: 100vh; }}
  #imgwrap {{ flex: 0 0 70%; min-width: 0; background: #000;
             display: flex; align-items: center; justify-content: center; }}
  #imgwrap img {{ max-width: 100%; max-height: 100vh; display: block; }}
  #side {{ flex: 0 0 30%; display: flex; flex-direction: column;
          border-left: 1px solid #23252c; background: #15171d; }}
  #side .group {{ padding: 26px 30px 0; font-size: 12px; letter-spacing: .14em;
                 text-transform: uppercase; color: #8b93a3; }}
  #side .num {{ padding: 6px 30px 0; font-size: 12px; letter-spacing: .12em;
               text-transform: uppercase; color: #6b717c; }}
  #caption {{ flex: 1 1 auto; min-height: 0; overflow: hidden; padding: 18px 30px 24px;
             line-height: 1.45; color: #d6d9df; display: flex; flex-direction: column;
             justify-content: center; }}
  #controls {{ flex: 0 0 auto; display: flex; align-items: center; gap: 14px;
              padding: 16px 30px 22px; border-top: 1px solid #23252c; }}
  #controls button {{ appearance: none; border: 1px solid #2f3340; background: #1b1e26;
              color: #cdd2dc; height: 40px; min-width: 44px; padding: 0 14px;
              border-radius: 9px; font-size: 16px; cursor: pointer;
              transition: background .15s, border-color .15s; }}
  #controls button:hover {{ background: #262a35; border-color: #3a4150; }}
  #controls button.on {{ background: #2d6cdf; border-color: #2d6cdf; color: #fff; }}
  #bar {{ position: fixed; left: 0; bottom: 0; height: 3px; background: #2d6cdf;
         width: 0; transition: width .2s linear; z-index: 5; }}
  /* start overlay */
  #start {{ position: fixed; inset: 0; background: rgba(8,9,12,.94); z-index: 9;
           display: flex; flex-direction: column; align-items: center; justify-content: center;
           gap: 22px; text-align: center; padding: 24px; }}
  #start h1 {{ margin: 0; font-size: 34px; font-weight: 650; }}
  #start p {{ margin: 0; color: #9aa0ab; }}
  #start button {{ appearance: none; border: none; background: #2d6cdf; color: #fff;
           font-size: 18px; padding: 14px 34px; border-radius: 11px; cursor: pointer; }}
  #start button:hover {{ background: #3b78ea; }}
  #start a {{ color: #9aa0ab; text-decoration: underline; cursor: pointer; font-size: 15px; }}
  #start a:hover {{ color: #cdd2dc; }}
  /* index overlay */
  #index-overlay {{ position: fixed; inset: 0; background: rgba(8,9,12,.97); z-index: 12;
                   display: none; flex-direction: column; }}
  #index-overlay.open {{ display: flex; }}
  #index-head {{ flex: 0 0 auto; display: flex; align-items: center; justify-content: space-between;
                padding: 16px 26px; border-bottom: 1px solid #23252c; }}
  #index-head h2 {{ margin: 0; font-size: 18px; font-weight: 600; }}
  #index-head button {{ appearance: none; border: 1px solid #2f3340; background: #1b1e26;
                color: #cdd2dc; height: 38px; padding: 0 16px; border-radius: 9px;
                font-size: 15px; cursor: pointer; }}
  #index-head button:hover {{ background: #262a35; border-color: #3a4150; }}
  #grid {{ flex: 1 1 auto; overflow-y: auto; padding: 20px 26px 40px;
          display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 14px; }}
  #grid figure {{ margin: 0; cursor: pointer; border: 2px solid transparent; border-radius: 10px;
                 overflow: hidden; background: #15171d; transition: border-color .15s, transform .1s; }}
  #grid figure:hover {{ border-color: #2d6cdf; transform: translateY(-2px); }}
  #grid figure.active {{ border-color: #2d6cdf; }}
  #grid img {{ width: 100%; aspect-ratio: 4/3; object-fit: cover; display: block; background: #000; }}
  #grid figcaption {{ padding: 6px 9px 8px; font-size: 11px; color: #9aa0ab;
                     white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  #grid figcaption b {{ color: #cdd2dc; font-weight: 600; }}
  @media (max-width: 760px) {{
    #stage {{ flex-direction: column; }}
    #side {{ flex: 1 1 auto; max-width: none; border-left: none; border-top: 1px solid #23252c; }}
    #imgwrap {{ flex: 0 0 46vh; }}
    #imgwrap img {{ max-height: 46vh; }}
  }}
</style>
</head><body>
<div id="stage">
  <div id="imgwrap"><img id="photo" alt=""></div>
  <div id="side">
    <div class="group" id="group"></div>
    <div class="num" id="num"></div>
    <div id="caption"></div>
    <div id="controls">
      <button id="prev" title="Previous (←)">⏮</button>
      <button id="toggle" title="Play / pause (space)">⏸</button>
      <button id="next" title="Next (→)">⏭</button>
      <button id="shuffle" title="Shuffle — play in random order (s)">🔀</button>
      <button id="index" title="Index — jump to a photo (i)">☰</button>
      <button id="mute" title="Mute narration">🔊</button>
    </div>
  </div>
</div>
<div id="bar"></div>
<div id="index-overlay">
  <div id="index-head">
    <h2>All photos</h2>
    <button id="index-close" title="Close (Esc)">✕ Close</button>
  </div>
  <div id="grid"></div>
</div>
<div id="start">
  <h1>Greece 2026</h1>
  <p>{n} photographs · captions &amp; narration by {model}<br>
     Plays each photo's narration, then advances on its own.</p>
  <button id="go">▶ Start slideshow</button>
  <a id="browse">… or pick a photo to start from</a>
</div>
<script>
const SLIDES = {data};
const ADVANCE_NO_AUDIO_MS = 9000;   // dwell time for slides with no narration
const GAP_MS = 700;                 // pause between slides

const photo = document.getElementById('photo');
const group = document.getElementById('group');
const num   = document.getElementById('num');
const cap   = document.getElementById('caption');
const bar   = document.getElementById('bar');
const toggle= document.getElementById('toggle');
const mute  = document.getElementById('mute');
const shuffleBtn = document.getElementById('shuffle');
const audio = new Audio();
// `order` is the play sequence (a permutation of slide indices); `pos` is where
// we are within it. In normal mode order is the identity [0,1,2,…]; shuffle mode
// replaces it with a random permutation so prev/next/progress all follow suit.
let order = SLIDES.map((_, n) => n);
let pos = 0, paused = false, timer = null, shuffled = false;
const cur = () => order[pos];   // current slide index into SLIDES

function clearTimer() {{ if (timer) {{ clearTimeout(timer); timer = null; }} }}

function schedule(ms) {{
  clearTimer();
  if (paused) return;
  timer = setTimeout(() => advance(1), ms);
}}

function render() {{
  const s = SLIDES[cur()];
  photo.src = s.img;
  photo.alt = s.stem;
  group.textContent = s.group;
  num.textContent = (pos + 1) + ' / ' + SLIDES.length;
  cap.textContent = s.caption;
  fitCaption();
  bar.style.width = ((pos + 1) / SLIDES.length * 100) + '%';
  if (gridBuilt) markActive();
}}

// Grow the caption to the largest font size that fills the side panel without
// overflowing (no scrolling). Binary-searches the font-size; line-height is
// relative so it scales along. Re-run on resize.
function fitCaption() {{
  let lo = 12, hi = 160;
  for (let k = 0; k < 14; k++) {{
    const mid = (lo + hi) / 2;
    cap.style.fontSize = mid + 'px';
    if (cap.scrollHeight <= cap.clientHeight) lo = mid; else hi = mid;
  }}
  cap.style.fontSize = Math.floor(lo) + 'px';
}}
let fitRAF = null;
window.addEventListener('resize', () => {{
  if (fitRAF) cancelAnimationFrame(fitRAF);
  fitRAF = requestAnimationFrame(fitCaption);
}});

function play() {{
  const s = SLIDES[cur()];
  audio.pause();
  clearTimer();
  if (s.audio && !paused) {{
    audio.src = s.audio;
    audio.muted = mute.dataset.muted === '1';
    audio.play().catch(() => schedule(ADVANCE_NO_AUDIO_MS));
  }} else if (!s.audio) {{
    schedule(ADVANCE_NO_AUDIO_MS);
  }}
}}

function goto(p) {{ pos = (p + SLIDES.length) % SLIDES.length; render(); play(); }}
function show(n) {{ goto(order.indexOf(n)); }}   // n is a slide index
function advance(d) {{ clearTimer(); audio.pause(); setTimeout(() => goto(pos + d), GAP_MS); }}

audio.addEventListener('ended', () => schedule(GAP_MS));

function setPaused(p) {{
  paused = p;
  toggle.textContent = p ? '▶' : '⏸';
  if (p) {{ audio.pause(); clearTimer(); }}
  else {{ play(); }}
}}

document.getElementById('prev').onclick = () => advance(-1);
document.getElementById('next').onclick = () => advance(1);
toggle.onclick = () => setPaused(!paused);

// Shuffle: rebuild `order` as a random permutation (or restore identity),
// keeping the current photo on screen so narration isn't interrupted.
function setShuffle(on) {{
  const current = cur();
  shuffled = on;
  shuffleBtn.classList.toggle('on', on);
  if (on) {{
    for (let k = order.length - 1; k > 0; k--) {{   // Fisher–Yates
      const j = Math.floor(Math.random() * (k + 1));
      [order[k], order[j]] = [order[j], order[k]];
    }}
    const at = order.indexOf(current);              // move current to the front
    [order[0], order[at]] = [order[at], order[0]];
    pos = 0;
  }} else {{
    order = SLIDES.map((_, n) => n);
    pos = current;                                  // resume in sequence
  }}
  render();   // current slide is unchanged, so leave audio playing
}}
shuffleBtn.onclick = () => setShuffle(!shuffled);
mute.onclick = () => {{
  const m = mute.dataset.muted === '1' ? '0' : '1';
  mute.dataset.muted = m;
  mute.textContent = m === '1' ? '🔇' : '🔊';
  audio.muted = m === '1';
}};
document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowRight') advance(1);
  else if (e.key === 'ArrowLeft') advance(-1);
  else if (e.key === ' ') {{ e.preventDefault(); setPaused(!paused); }}
  else if (e.key === 's' || e.key === 'S') setShuffle(!shuffled);
}});

document.getElementById('go').onclick = () => {{
  document.getElementById('start').remove();
  show(0);
}};

// ---- index overlay: jump to / start from any photo --------------------------
const grid = document.getElementById('grid');
const indexOverlay = document.getElementById('index-overlay');
let gridBuilt = false;

function buildGrid() {{
  if (gridBuilt) return;
  gridBuilt = true;
  SLIDES.forEach((s, n) => {{
    const fig = document.createElement('figure');
    const im = document.createElement('img');
    im.loading = 'lazy'; im.src = s.thumb; im.alt = '';
    const fc = document.createElement('figcaption');
    const b = document.createElement('b'); b.textContent = n + 1;
    fc.appendChild(b);
    fc.appendChild(document.createTextNode(' · ' + s.group));
    fig.appendChild(im); fig.appendChild(fc);
    fig.onclick = () => jumpTo(n);
    grid.appendChild(fig);
  }});
}}

function markActive() {{
  const active = cur();
  for (let n = 0; n < grid.children.length; n++)
    grid.children[n].classList.toggle('active', n === active);
}}

function openIndex() {{
  buildGrid();
  markActive();
  indexOverlay.classList.add('open');
  const a = grid.children[cur()];
  if (a) a.scrollIntoView({{ block: 'center' }});
}}

function closeIndex() {{ indexOverlay.classList.remove('open'); }}

function jumpTo(n) {{
  closeIndex();
  const start = document.getElementById('start');
  if (start) start.remove();   // first interaction also dismisses the splash
  show(n);
}}

document.getElementById('index').onclick = openIndex;
document.getElementById('index-close').onclick = closeIndex;
document.getElementById('browse').onclick = openIndex;
indexOverlay.addEventListener('click', e => {{ if (e.target === indexOverlay) closeIndex(); }});
document.addEventListener('keydown', e => {{
  if (e.key === 'i' || e.key === 'I') openIndex();
  else if (e.key === 'Escape') closeIndex();
}});
</script>
</body></html>
"""


def build(max_distance: int, img_max: int) -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    photos = photo_src.list_photos()
    if not photos:
        raise SystemExit("no source photos found; refusing to overwrite slideshow.html")

    slides, kept_hashes = [], []
    dropped = no_cap = no_thumb = 0
    for rec in photos:
        stem = rec["stem"]
        thumb = THUMB_DIR / (stem + ".jpg")
        cap_file = OUT_DIR / (stem + ".txt")
        if not thumb.exists():
            no_thumb += 1
            continue
        text = cap_file.read_text().strip() if cap_file.exists() else ""
        if not text:
            no_cap += 1
            continue
        h = dhash(thumb)
        if is_dupe(h, kept_hashes, max_distance):
            dropped += 1
            continue
        kept_hashes.append(h)
        slides.append({"stem": stem, "group": rec["group"],
                       "caption": text, "raw": rec["path"]})

    print(f"{len(photos)} photos · {dropped} duplicates removed · "
          f"{no_cap} without caption · {no_thumb} without thumbnail")
    print(f"rendering {len(slides)} high-res images (≤{img_max}px) → {IMG_DIR}")

    data = []
    for n, s in enumerate(slides, 1):
        dst = IMG_DIR / (s["stem"] + ".jpg")
        try:
            make_display(s["raw"], dst, img_max)
        except Exception as e:
            print(f"  ! {s['stem']}: {type(e).__name__}: {e}")
            continue
        wav = AUDIO_DIR / (s["stem"] + ".wav")
        data.append({
            "stem": s["stem"],
            "group": s["group"],
            "caption": s["caption"],
            "img": f"slideshow_img/{dst.name}",
            "thumb": f"thumbs/{s['stem']}.jpg",
            "audio": f"audio/{wav.name}" if wav.exists() else None,
        })
        if n % 25 == 0:
            print(f"  {n}/{len(slides)}")

    out = OUT_DIR / "slideshow.html"
    out.write_text(PAGE.format(n=len(data), model=html.escape(MODEL_NAME),
                               data=json.dumps(data, ensure_ascii=False)))
    with_audio = sum(1 for d in data if d["audio"])
    print(f"\nwrote {out}")
    print(f"slides: {len(data)} ({with_audio} with narration)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-distance", type=int, default=5,
                    help="dHash Hamming distance under which photos count as "
                         "duplicates (default 5; 0 = byte-exact only).")
    ap.add_argument("--img-max", type=int, default=2200,
                    help="longest edge of the high-res display image (px).")
    args = ap.parse_args()
    build(args.max_distance, args.img_max)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
