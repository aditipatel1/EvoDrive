"""
make_viewer.py - build viewer.html (the 3D replay viewer) with your replays inside.

Embedding the replays in the page means it opens with a double-click, with
no web server, and can be published or shared as a single file.

Run:   python3 export_replay.py                                  # makes replay.json
       python3 export_replay.py --stress --any-result --out replay_stress.json
       python3 make_viewer.py replay.json replay_stress.json     # makes viewer.html
Then open viewer.html in your browser.
"""
import argparse
import json
import os

TEMPLATE = "viewer_template.html"
MARKER = "/*__REPLAYS__*/[]"


def label(path, data):
    m = data.get("meta", {})
    track = m.get("title") or ("stress track" if m.get("half_width", 22) < 22 else "normal track")
    physics = "grip physics" if m.get("physics") == "grip" else "arcade physics"
    return (f"{os.path.basename(path)} · {track} · {physics} · "
            f"{m.get('result', '?')} ({m.get('laps', 0):.2f} laps)")


def main(paths, out):
    with open(TEMPLATE) as f:
        page = f.read()
    if MARKER not in page:
        raise SystemExit(f"{TEMPLATE} is missing the {MARKER} marker.")
    replays = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        replays.append({"name": label(p, data), "data": data})
    # "</" is escaped so replay text can never close the <script> tag early
    blob = json.dumps(replays, separators=(",", ":")).replace("</", "<\\/")
    with open(out, "w") as f:
        f.write(page.replace(MARKER, blob))
    print(f"saved {out} with {len(replays)} replay(s), {os.path.getsize(out) / 1e3:.0f} KB")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("replays", nargs="+", help="replay JSON files from export_replay.py")
    p.add_argument("--out", default="viewer.html")
    a = p.parse_args()
    main(a.replays, a.out)
