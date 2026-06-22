#!/usr/bin/env python3
"""Render a live status dashboard for the Greece archive build.

Scans the three pipeline stages — thumbnails, captions, narration audio — for
every photo across all folders and writes captions/status.html: a dark,
auto-refreshing page with overall + per-folder progress. Use --watch to keep
regenerating it while the pipeline runs:

  python3 status.py            # write once
  python3 status.py --watch 4  # rewrite every 4s until everything is done
"""
import html
import time
from datetime import datetime
from pathlib import Path

import photos as photo_src

OUT_DIR = Path("/home/mjbernaski/projects/omni-play/captions")
THUMB_DIR = OUT_DIR / "thumbs"
AUDIO_DIR = OUT_DIR / "audio"

STAGES = [
    ("thumbnail", "Thumbnails", THUMB_DIR, ".jpg"),
    ("caption", "Captions", OUT_DIR, ".txt"),
    ("audio", "Narration", AUDIO_DIR, ".wav"),
]


def _has(rec, directory, ext):
    f = directory / (rec["stem"] + ext)
    if not f.exists():
        return False
    if ext == ".txt":  # a caption only counts if it has text
        return bool(f.read_text().strip())
    return f.stat().st_size > 0


def scan():
    """Return per-group and overall counts for each stage."""
    recs = photo_src.list_photos()
    groups = {}
    for rec in recs:
        g = groups.setdefault(rec["group"], {"total": 0,
                                             **{k: 0 for k, *_ in STAGES}})
        g["total"] += 1
        for key, _label, d, ext in STAGES:
            if _has(rec, d, ext):
                g[key] += 1
    overall = {"total": len(recs), **{k: 0 for k, *_ in STAGES}}
    for g in groups.values():
        overall["total"] = overall["total"]  # noqa
        for key, *_ in STAGES:
            overall[key] += g[key]
    return groups, overall


def _bar(done, total):
    pct = (done / total * 100) if total else 0
    full = done >= total and total > 0
    cls = "bar full" if full else "bar"
    return (f'<div class="track"><div class="{cls}" '
            f'style="width:{pct:.1f}%"></div></div>'
            f'<div class="barlabel">{done}/{total} · {pct:.0f}%</div>')


PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="{refresh}">
<title>Build status · Greece archive</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: #0e0f13; color: #e6e7ea;
         font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
  main {{ max-width: 880px; margin: 0 auto; padding: 48px 22px 80px; }}
  h1 {{ margin: 0 0 4px; font-size: 26px; font-weight: 650; }}
  .sub {{ color: #9aa0ab; margin: 0 0 8px; }}
  .pill {{ display: inline-block; padding: 4px 12px; border-radius: 999px;
          font-size: 13px; font-weight: 600; letter-spacing: .03em; }}
  .pill.run {{ background: #1d3a6b; color: #9ec3ff; }}
  .pill.done {{ background: #1c4a2e; color: #93e6b0; }}
  .card {{ background: #15171d; border: 1px solid #23252c; border-radius: 14px;
          padding: 22px 24px; margin: 22px 0; }}
  .card h2 {{ margin: 0 0 18px; font-size: 13px; letter-spacing: .14em;
             text-transform: uppercase; color: #8b93a3; }}
  .stage {{ margin: 0 0 18px; }}
  .stage:last-child {{ margin-bottom: 0; }}
  .stage .name {{ display: flex; justify-content: space-between;
                 font-size: 14px; margin-bottom: 7px; }}
  .stage .name b {{ font-weight: 600; }}
  .track {{ height: 12px; background: #0a0b0e; border: 1px solid #23252c;
           border-radius: 999px; overflow: hidden; }}
  .bar {{ height: 100%; background: linear-gradient(90deg,#2d6cdf,#5b8df0);
         border-radius: 999px; transition: width .4s; }}
  .bar.full {{ background: linear-gradient(90deg,#1f9d57,#37c977); }}
  .barlabel {{ font-size: 12px; color: #8b93a3; margin-top: 5px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th, td {{ text-align: left; padding: 9px 10px; border-bottom: 1px solid #20222a; }}
  th {{ color: #8b93a3; font-weight: 600; font-size: 12px;
       text-transform: uppercase; letter-spacing: .08em; }}
  td.n {{ text-align: right; font-variant-numeric: tabular-nums; color: #d6d9df; }}
  td.n.ok {{ color: #6fe39a; }}
  .foot {{ color: #6b717c; font-size: 13px; margin-top: 26px; }}
  a {{ color: #7fa8ff; }}
</style>
</head><body>
<main>
  <h1>Greece archive — build status</h1>
  <p class="sub">{total} photographs across {ngroups} folders · captions by Nemotron-3-Nano-Omni · narration by XTTS-v2</p>
  <p>{pill}</p>
{overall}
{bygroup}
  <p class="foot">Updated {ts} · auto-refreshes every {refresh}s · <a href="index.html">open the gallery →</a></p>
</main>
</body></html>
"""


def render(refresh=5):
    groups, overall = scan()
    everything_done = all(overall[k] >= overall["total"]
                          for k, *_ in STAGES) and overall["total"] > 0
    pill = ('<span class="pill done">✓ complete</span>' if everything_done
            else '<span class="pill run">● building…</span>')

    # Overall card: one progress bar per stage.
    stage_rows = []
    for key, label, *_ in STAGES:
        stage_rows.append(
            f'<div class="stage"><div class="name"><b>{label}</b></div>'
            f'{_bar(overall[key], overall["total"])}</div>')
    overall_card = ('  <div class="card"><h2>Overall</h2>'
                    + "".join(stage_rows) + "</div>")

    # Per-folder table.
    head = ("<tr><th>Folder</th><th>Photos</th>"
            + "".join(f"<th>{lbl}</th>" for _k, lbl, *_ in STAGES) + "</tr>")
    rows = [head]
    for label, g in groups.items():
        cells = [f'<td>{html.escape(label)}</td>',
                 f'<td class="n">{g["total"]}</td>']
        for key, *_ in STAGES:
            ok = "ok" if g[key] >= g["total"] else ""
            cells.append(f'<td class="n {ok}">{g[key]}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    bygroup = ('  <div class="card"><h2>By folder</h2><table>'
               + "".join(rows) + "</table></div>")

    page = PAGE.format(
        refresh=refresh, total=overall["total"], ngroups=len(groups),
        pill=pill, overall=overall_card, bygroup=bygroup,
        ts=datetime.now().strftime("%H:%M:%S"))
    (OUT_DIR / "status.html").write_text(page)
    return everything_done


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=float, default=0,
                    help="seconds between refreshes; loop until all stages done")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.watch:
        done = render()
        print(f"wrote {OUT_DIR/'status.html'} (complete={done})")
        return
    refresh = max(2, int(args.watch))
    while True:
        done = render(refresh=refresh)
        if done:
            print("all stages complete")
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
