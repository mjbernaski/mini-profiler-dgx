#!/usr/bin/env python3
"""Serve the Greece gallery in production via waitress + whitenoise.

The site is fully static (captions/index.html + thumbs/ + audio/). WhiteNoise
handles correct caching headers, gzip, and HTTP range requests (the latter
matters so browsers can seek/stream the narration WAVs), and waitress is a
production WSGI server. Run:

  .venv-serve/bin/python serve.py            # binds 0.0.0.0:8888

Override with PORT / HOST env vars.
"""
import os
from pathlib import Path

from waitress import serve
from whitenoise import WhiteNoise

ROOT = Path(__file__).resolve().parent / "captions"


def _not_found(environ, start_response):
    start_response("404 Not Found", [("Content-Type", "text/plain")])
    return [b"Not found"]


# index_file=True serves index.html for "/". autorefresh=True so the live
# status page and regenerated gallery show up without a server restart — the
# cost is one os.stat per request, negligible for a personal gallery.
# WhiteNoise keeps its own media-type table, so .wav must be registered here
# (this host's mimetypes omit it) or narration serves as octet-stream.
application = WhiteNoise(_not_found, root=str(ROOT), index_file=True,
                        autorefresh=True, mimetypes={".wav": "audio/wav"})


def main():
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8888"))
    print(f"serving {ROOT} on http://{host}:{port}")
    serve(application, host=host, port=port)


if __name__ == "__main__":
    main()
