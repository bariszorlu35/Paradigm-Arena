"""Embed the trained policies.json into index.html.

Run this AFTER `python3 train.py` so the website uses your freshly trained
models. The game stays a single self-contained file you can just double-click.

    python3 train.py     # trains all 6 paradigms -> policies.json
    python3 build.py     # bakes policies.json into index.html
    # then open index.html in your browser
"""
import json, os, sys

HTML = "index.html"
POL = "policies.json"
MARK = "const POLICIES = "

if not os.path.exists(POL):
    sys.exit("policies.json not found — run `python3 train.py` first.")
if not os.path.exists(HTML):
    sys.exit("index.html not found — keep build.py in the same folder as index.html.")

policies = open(POL, encoding="utf-8").read().strip()
json.loads(policies)                                  # validate it's real JSON

html = open(HTML, encoding="utf-8").read()
start = html.find(MARK)
if start == -1:
    sys.exit("Could not find the POLICIES marker in index.html.")
end = html.find(";", start)                           # JSON has no ';' inside

new_html = html[:start] + MARK + policies + html[end:]
open(HTML, "w", encoding="utf-8").write(new_html)
print(f"OK — embedded {len(policies)} bytes of policies into {HTML}")
print("Now open index.html in your browser and press ▶ Başlat.")
