#!/usr/bin/env python3
"""Build a self-contained HTML gallery of the Greece photos + their captions.

Generates web-sized thumbnails (raw photos are 12-75MB each) into
captions/thumbs/, reads each caption from captions/<stem>.txt, and writes
captions/index.html with the photos in trip order. Re-runnable: existing
thumbnails are skipped, captions are re-read fresh each time.
"""
import html
from pathlib import Path

from PIL import Image, ImageOps

import photos as photo_src

OUT_DIR = Path("/home/mjbernaski/projects/omni-play/captions")
THUMB_DIR = OUT_DIR / "thumbs"
AUDIO_DIR = OUT_DIR / "audio"  # narrated captions, one <stem>.wav per photo
THUMB_MAX = 1400  # longest edge of gallery image
MODEL_NAME = "Nemotron-3-Nano-Omni"  # caption author, shown per-photo and in footer


def make_thumb(src, dst):
    if dst.exists():
        return
    # Write to a temp file first so an interrupt mid-save never leaves a
    # truncated thumbnail that a later run would mistake for "done".
    tmp = dst.with_suffix(dst.suffix + ".part")
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((THUMB_MAX, THUMB_MAX), Image.LANCZOS)
        im.save(tmp, format="JPEG", quality=85)
    tmp.replace(dst)


PAGE_HEAD = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Greece 2026 — Attica, the Peloponnese &amp; Rhodes</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; background: #0e0f13; color: #e6e7ea;
         font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  header { padding: 48px 24px 24px; text-align: center; border-bottom: 1px solid #23252c; }
  header h1 { margin: 0 0 8px; font-size: 30px; font-weight: 650; letter-spacing: .2px; }
  header p { margin: 0; color: #9aa0ab; }
  main { max-width: 1100px; margin: 0 auto; padding: 32px 20px 80px; }
  h2.section { margin: 8px 0 28px; padding-bottom: 10px; font-size: 14px;
               font-weight: 600; letter-spacing: .14em; text-transform: uppercase;
               color: #8b93a3; border-bottom: 1px solid #23252c; }
  h2.section:not(:first-child) { margin-top: 40px; }
  figure { margin: 0 0 56px; background: #15171d; border: 1px solid #23252c;
           border-radius: 14px; overflow: hidden; }
  figure img { display: block; width: 100%; height: auto; background: #000; cursor: zoom-in; }
  figcaption { padding: 20px 24px 24px; }
  figcaption .num { font-size: 12px; letter-spacing: .12em; text-transform: uppercase;
                    color: #7d8492; margin-bottom: 8px;
                    display: flex; align-items: center; gap: 12px; }
  .play { appearance: none; border: 1px solid #2f3340; background: #1b1e26;
          color: #cdd2dc; width: 30px; height: 30px; border-radius: 50%;
          font-size: 12px; line-height: 1; cursor: pointer; padding: 0;
          display: inline-flex; align-items: center; justify-content: center;
          transition: background .15s, border-color .15s; }
  .play:hover { background: #262a35; border-color: #3a4150; }
  .play.playing { background: #2d6cdf; border-color: #2d6cdf; color: #fff; }
  figcaption p { margin: 0; color: #d6d9df; }
  .pending { color: #7d8492; font-style: italic; }
  figcaption .credit { color: #7d8492; font-size: 13px; }
  footer { text-align: center; color: #6b717c; padding: 0 0 60px; font-size: 14px; }
  /* lightbox */
  #lb { position: fixed; inset: 0; background: rgba(0,0,0,.92); display: none;
        align-items: center; justify-content: center; cursor: zoom-out; z-index: 9; }
  #lb img { max-width: 96vw; max-height: 96vh; }
</style>
</head><body>
<header>
  <h1>Greece 2026</h1>
  <p>Attica · the Peloponnese · Rhodes</p>
</header>
<main>
"""

PAGE_TAIL = """</main>
<footer>{n} photographs · captions by {model}</footer>
<div id="lb"><img alt=""></div>
<script>
  const lb = document.getElementById('lb'), lbi = lb.querySelector('img');
  document.querySelectorAll('figure img').forEach(img => {{
    img.addEventListener('click', () => {{ lbi.src = img.src; lb.style.display = 'flex'; }});
  }});
  lb.addEventListener('click', () => {{ lb.style.display = 'none'; lbi.src = ''; }});
  document.addEventListener('keydown', e => {{ if (e.key === 'Escape') lb.click(); }});

  // Narration playback: one shared <audio>; clicking a button toggles it and
  // stops whatever was playing before.
  const player = new Audio();
  let current = null;
  function stopCurrent() {{
    if (current) {{ current.classList.remove('playing'); current.textContent = '▶'; current = null; }}
  }}
  player.addEventListener('ended', stopCurrent);
  document.querySelectorAll('button.play').forEach(btn => {{
    btn.addEventListener('click', () => {{
      if (current === btn) {{ player.pause(); stopCurrent(); return; }}
      stopCurrent();
      player.src = btn.dataset.src;
      player.play();
      current = btn;
      btn.classList.add('playing');
      btn.textContent = '⏸';
    }});
  }});
</script>
</body></html>
"""


def render_index(photos):
    """Build index.html from whatever thumbs/captions exist right now.

    Cheap (no image work), so it's safe to call repeatedly to checkpoint
    progress. A figure whose thumbnail isn't ready yet is marked pending.
    Returns (captions_present, thumbs_present).
    """
    parts = [PAGE_HEAD]
    n = len(photos)
    caps_done = thumbs_done = 0
    cur_group = None
    for i, rec in enumerate(photos, 1):
        stem, name, group = rec["stem"], rec["path"].name, rec["group"]
        if group != cur_group:
            cur_group = group
            parts.append(f'<h2 class="section">{html.escape(group)}</h2>\n')
        thumb = THUMB_DIR / (stem + ".jpg")
        cap_file = OUT_DIR / (stem + ".txt")
        if cap_file.exists() and cap_file.read_text().strip():
            text = html.escape(cap_file.read_text().strip())
            credit = (f' <span class="credit">(written by: '
                      f'{html.escape(MODEL_NAME)})</span>')
            cap = f"<p>{text}{credit}</p>"
            caps_done += 1
        else:
            cap = '<p class="pending">(caption pending)</p>'
        if thumb.exists():
            thumbs_done += 1
            img = (f'<img loading="lazy" src="thumbs/{thumb.name}" '
                   f'alt="{html.escape(name)}">')
        else:
            img = ('<div class="pending" style="padding:24px">'
                   '(thumbnail pending)</div>')
        audio_file = AUDIO_DIR / (stem + ".wav")
        if audio_file.exists():
            play = (f'<button class="play" type="button" '
                    f'data-src="audio/{audio_file.name}" '
                    f'aria-label="Play narration" title="Play narration">'
                    f'▶</button>')
        else:
            play = ''
        parts.append(
            f'<figure id="p{i}">{img}'
            f'<figcaption><div class="num">{i} / {n}{play}</div>{cap}</figcaption>'
            f'</figure>\n'
        )
    parts.append(PAGE_TAIL.format(n=n, model=html.escape(MODEL_NAME)))
    (OUT_DIR / "index.html").write_text("".join(parts))
    return caps_done, thumbs_done


def main():
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    photos = photo_src.list_photos()
    interrupted = False
    try:
        for i, rec in enumerate(photos, 1):
            thumb = THUMB_DIR / (rec["stem"] + ".jpg")
            try:
                make_thumb(rec["path"], thumb)
            except Exception as e:
                print(f"  ! skipping {rec['path'].name}: {e}")
            # Checkpoint the gallery periodically so an interrupt leaves a
            # valid, viewable index.html with all completed work.
            if i % 25 == 0:
                caps, thumbs = render_index(photos)
                print(f"  progress: {i}/{len(photos)} "
                      f"(thumbs {thumbs}, captions {caps})")
    except KeyboardInterrupt:
        interrupted = True
        print("\ninterrupted — saving progress so far")

    caps, thumbs = render_index(photos)
    print(f"\nwrote {OUT_DIR/'index.html'}")
    print(f"thumbnails present: {thumbs}/{len(photos)}")
    print(f"captions present:   {caps}/{len(photos)}")
    if interrupted:
        print("re-run to continue where it left off")


if __name__ == "__main__":
    main()
