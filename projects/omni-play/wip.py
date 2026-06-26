#!/usr/bin/env python3
"""Live work-in-progress page: thumbnails + captions as they are generated.

Writes captions/wip.html — a dark, auto-refreshing grid where every photo shows
its thumbnail and, beside it, the caption text once it lands on disk (or a
"generating…" placeholder until then). Photos still waiting are dimmed; the next
uncaptioned photo in trip order is highlighted as "up next". Pairs with a
captioning run so you can watch progress live:

  python3 wip.py            # write once
  python3 wip.py --watch 4  # rewrite every 4s until every photo is captioned
"""
import html
import time
from datetime import datetime
from pathlib import Path

import photos as photo_src

OUT_DIR = Path("/home/mjbernaski/projects/omni-play/captions")
THUMB_DIR = OUT_DIR / "thumbs"
AUDIO_DIR = OUT_DIR / "audio"


def _caption(stem):
    f = OUT_DIR / (stem + ".txt")
    if f.exists():
        t = f.read_text().strip()
        if t:
            return t
    return None


PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="{refresh}">
<title>Captioning in progress · Greece</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: #0e0f13; color: #e6e7ea;
         font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
  header {{ position: sticky; top: 0; z-index: 5; backdrop-filter: blur(8px);
           background: rgba(14,15,19,.86); border-bottom: 1px solid #23252c;
           padding: 18px 24px; }}
  header h1 {{ margin: 0 0 8px; font-size: 20px; font-weight: 650; }}
  .track {{ height: 10px; background: #0a0b0e; border: 1px solid #23252c;
           border-radius: 999px; overflow: hidden; max-width: 520px; }}
  .bar {{ height: 100%; background: linear-gradient(90deg,#2d6cdf,#5b8df0);
         border-radius: 999px; transition: width .4s; }}
  .bar.full {{ background: linear-gradient(90deg,#1f9d57,#37c977); }}
  .sub {{ color: #9aa0ab; font-size: 13px; margin-top: 6px; }}
  main {{ max-width: 1000px; margin: 0 auto; padding: 22px 18px 80px; }}
  .row {{ display: flex; gap: 16px; padding: 14px; margin: 0 0 14px;
         background: #15171d; border: 1px solid #23252c; border-radius: 12px;
         align-items: flex-start; }}
  .row.todo {{ opacity: .5; }}
  .row.next {{ border-color: #2d6cdf; box-shadow: 0 0 0 1px #2d6cdf inset; opacity: 1; }}
  .row img {{ width: 200px; height: 140px; object-fit: cover; border-radius: 8px;
            background: #000; flex: 0 0 auto; }}
  .row .thumbph {{ width: 200px; height: 140px; border-radius: 8px; flex: 0 0 auto;
                 background: #0a0b0e; border: 1px solid #23252c; }}
  .meta {{ min-width: 0; }}
  .num {{ font-size: 12px; letter-spacing: .12em; text-transform: uppercase;
         color: #7d8492; margin-bottom: 6px; }}
  .cap {{ color: #d6d9df; }}
  .gen {{ color: #5b8df0; font-style: italic; }}
  .nextlbl {{ display: inline-block; margin-left: 8px; padding: 1px 8px;
             border-radius: 999px; background: #1d3a6b; color: #9ec3ff;
             font-size: 11px; letter-spacing: .04em; }}
  .foot {{ color: #6b717c; font-size: 13px; margin-top: 24px; }}
  a {{ color: #7fa8ff; }}
</style>
</head><body>
<header>
  <h1>Captioning in progress {pill}</h1>
  <div class="track"><div class="{barcls}" style="width:{pct:.1f}%"></div></div>
  <div class="sub">{done}/{total} captions · {pct:.0f}% · updated {ts} · refreshes every {refresh}s · <a href="index.html">gallery →</a></div>
</header>
<main>
{rows}
</main>
</body></html>
"""


def render(refresh=4):
    recs = photo_src.list_photos()
    total = len(recs)
    done = 0
    next_marked = False
    rows = []
    for i, rec in enumerate(recs, 1):
        stem = rec["stem"]
        cap = _caption(stem)
        thumb = THUMB_DIR / (stem + ".jpg")
        if thumb.exists():
            img = f'<img loading="lazy" src="thumbs/{html.escape(thumb.name)}" alt="">'
        else:
            img = '<div class="thumbph"></div>'
        if cap:
            done += 1
            body = f'<div class="cap">{html.escape(cap)}</div>'
            cls = "row"
            nextlbl = ""
        else:
            cls = "row todo"
            nextlbl = ""
            if not next_marked:
                cls = "row next"
                next_marked = True
                nextlbl = '<span class="nextlbl">up next</span>'
                body = '<div class="gen">⏳ generating caption…</div>'
            else:
                body = '<div class="gen">waiting…</div>'
        rows.append(
            f'<div class="{cls}">{img}'
            f'<div class="meta"><div class="num">{i} / {total}{nextlbl}</div>{body}</div>'
            f'</div>'
        )
    pct = (done / total * 100) if total else 0
    complete = done >= total and total > 0
    barcls = "bar full" if complete else "bar"
    pill = '✓' if complete else ''
    page = PAGE.format(
        refresh=refresh, total=total, done=done, pct=pct, barcls=barcls,
        pill=pill, ts=datetime.now().strftime("%H:%M:%S"), rows="\n".join(rows))
    (OUT_DIR / "wip.html").write_text(page)
    return complete


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=float, default=0,
                    help="seconds between refreshes; loop until all captioned")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.watch:
        done = render()
        print(f"wrote {OUT_DIR/'wip.html'} (complete={done})")
        return
    refresh = max(2, int(args.watch))
    while True:
        if render(refresh=refresh):
            print("all captioned")
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
