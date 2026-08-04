#!/usr/bin/env python
"""Drive the whole-dome viewer in a real browser and assert it behaves.

    ./.venv/bin/python tests/check_mosaic_viewer.py \
        output/BK52_WT_9805_B_mosaic/viewer.html

The single-field viewer's worst faults were invisible to every check that did
not actually run the page: a 3D view that moved the camera without asking for a
repaint, and canvases whose backing store drifted out of step with their
element. Neither could be seen by parsing the file. The same applies here, with
one addition specific to the mosaic — the field-of-view switches are the whole
point of this viewer, and "it renders" says nothing about whether turning a
field off actually removes its organoids from both panes.

Headless Firefox will not capture a WebGL canvas in a screenshot, so the harness
reads the framebuffer back with gl.readPixels rather than trusting a picture.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

HARNESS = r"""
<script>
(() => {
  const out = [];
  const ok = (c, msg) => out.push((c ? "PASS  " : "FAIL  ") + msg);

  try {
    ok(ORG.length > 0, `${ORG.length} organoids loaded`);
    ok(TILES.length === M.n_rows * M.n_cols,
       `${TILES.length} fields, matching the ${M.n_rows}x${M.n_cols} grid`);

    // --- both panes actually put pixels up ---
    dirty3 = true; draw3d();
    const gl = renderer.getContext();
    const w = renderer.domElement.width, h = renderer.domElement.height;
    const px = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
    let lit = 0;
    for (let i = 0; i < px.length; i += 4)
      if (px[i] + px[i+1] + px[i+2] > 40) lit++;
    ok(lit > 1000, `3D framebuffer has ${lit} lit pixels`);

    let inFrustum = 0;
    for (const [uid, m] of meshes) {
      const p = m.position.clone().project(camera);
      if (Math.abs(p.x) <= 1 && Math.abs(p.y) <= 1 && p.z > -1 && p.z < 1) inFrustum++;
    }
    ok(inFrustum > ORG.length * 0.4, `${inFrustum}/${ORG.length} organoids in view`);

    // --- the field switches are the point of this viewer ---
    const all = ORG.length;
    state.fovs = new Set([TILES[0].index]);
    refresh(); draw3d();
    const one = ORG.filter(shown).length;
    let hidden = 0;
    for (const [uid, m] of meshes) if (!m.visible) hidden++;
    ok(one < all && one > 0,
       `one field alone shows ${one} of ${all} organoids`);
    ok(hidden === all - one,
       `${hidden} meshes hidden in 3D, matching the ${all - one} filtered out`);

    state.fovs = new Set(TILES.slice(0, 2).map(t => t.index));
    refresh();
    const two = ORG.filter(shown).length;
    ok(two > one, `two fields show more than one (${two} > ${one})`);

    state.fovs = new Set(TILES.map(t => t.index));
    refresh(); draw3d();
    ok(ORG.filter(shown).length === all, "all fields back on restores every organoid");
    ok([...meshes.values()].every(m => m.visible), "every mesh visible again");

    // --- an organoid appears once, not once per tile that saw it ---
    const seenTwice = ORG.filter(o => o.n_views > 1).length;
    ok(new Set(ORG.map(o => o.uid)).size === ORG.length,
       `every uid is unique (${seenTwice} organoids carry more than one view)`);

    // --- selection drives the feature table ---
    const target = ORG.find(o => o.appearance_measurable) || ORG[0];
    select(target);
    const body = document.getElementById("detail").innerHTML;
    ok(body.includes("uid"), "clicking an organoid fills the feature table");
    ok(/core|rim|circularity/.test(body), "the table carries appearance features");
    ok(state.selected === target, "selection state follows the click");

    // --- the dome border line ---
    if (M.dome) {
      state.showLine = true; dirty3 = true; draw3d();
      ok(lineObj !== null, "the border line is drawn when switched on");
      if (lineObj) {
        const p = lineObj.geometry.attributes.position.array;
        const d = Math.hypot(p[3]-p[0], p[4]-p[1], p[5]-p[2]);
        ok(d > 0 && isFinite(d), `the line has a real length (${d.toFixed(0)} px)`);
      }
      state.showLine = false; dirty3 = true; draw3d();
      ok(lineObj === null, "and removed when switched off");
    }

    // --- dragging repaints, rather than silently moving the camera ---
    const before = camera.position.clone();
    dirty3 = false;
    const r = c3d.getBoundingClientRect();
    c3d.dispatchEvent(new MouseEvent("mousedown", {
      clientX: r.left + 200, clientY: r.top + 200, button: 0, bubbles: true }));
    window.dispatchEvent(new MouseEvent("mousemove", {
      clientX: r.left + 400, clientY: r.top + 240, bubbles: true }));
    window.dispatchEvent(new MouseEvent("mouseup", { bubbles: true }));
    placeCamera();
    ok(before.distanceTo(camera.position) > 1,
       `drag moved the camera by ${before.distanceTo(camera.position).toFixed(0)} units`);
    ok(dirty3 === true, "drag requested a repaint");

    // --- every canvas is the size of its box ---
    resize();
    let good = true, detail = [];
    for (const [cv, pane, isGl] of [[c2d,"p2d",false],[c3d,"p3d",true]]) {
      const el = document.getElementById(pane);
      const cw = isGl ? renderer.domElement.width : cv.width;
      const ch = isGl ? renderer.domElement.height : cv.height;
      if (cw !== el.clientWidth || ch !== el.clientHeight) good = false;
      detail.push(`${pane} ${el.clientWidth}x${el.clientHeight} vs ${cw}x${ch}`);
    }
    ok(good, `canvases match their boxes: ${detail.join(" | ")}`);

    ok((window.__errs || []).length === 0,
       `no page errors (${(window.__errs || []).join("; ") || "none"})`);
  } catch (e) {
    ok(false, "harness threw: " + e + (e.stack ? " @ " + e.stack.split("\n")[1] : ""));
  }

  if (typeof dump === "function")
    dump("HARNESS-BEGIN\n" + out.join("\n") + "\nHARNESS-END\n");
})();
</script>
"""

ERROR_TRAP = """<script>
window.__errs = [];
window.addEventListener("error", e => window.__errs.push("ERR " + (e.message || e)));
window.addEventListener("unhandledrejection", e => window.__errs.push("REJ " + e.reason));
</script>
"""


def run(viewer: Path, keep: bool = False) -> int:
    firefox = shutil.which("firefox")
    if not firefox:
        print("firefox not found; skipping", file=sys.stderr)
        return 0

    viewer = viewer.resolve()
    work = viewer.parent / "_viewercheck"
    work.mkdir(exist_ok=True)
    page = work / "harness.html"

    html = viewer.read_text(encoding="utf-8")
    html = html.replace("<div id=\"app\">", ERROR_TRAP + "<div id=\"app\">", 1)
    page.write_text(html + HARNESS, encoding="utf-8")

    prof = work / "prof"
    prof.mkdir(exist_ok=True)
    (prof / "user.js").write_text('user_pref("browser.dom.window.dump.enabled", true);\n')

    proc = subprocess.run(
        [firefox, "--new-instance", "--headless", "--profile", str(prof),
         "--window-size=1700,1000", "--screenshot", str(work / "shot.png"),
         page.as_uri()],
        capture_output=True, text=True, timeout=600)

    blob = proc.stdout + proc.stderr
    m = re.search(r"HARNESS-BEGIN\n(.*?)\nHARNESS-END", blob, re.S)
    if not m:
        print("no report from the page. browser said:\n" + blob[-1200:], file=sys.stderr)
        if not keep:
            shutil.rmtree(work, ignore_errors=True)
        return 2

    report = m.group(1)
    print(report)
    fails = sum(1 for line in report.splitlines() if line.startswith("FAIL"))
    passes = sum(1 for line in report.splitlines() if line.startswith("PASS"))
    print(f"\n{passes} passed, {fails} failed")
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    return 1 if fails else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Browser check for the mosaic viewer")
    ap.add_argument("viewer", nargs="?",
                    default="output/BK52_WT_9805_B_mosaic/viewer.html")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args(argv)
    v = Path(a.viewer)
    if not v.is_file():
        print(f"no such viewer: {v}", file=sys.stderr)
        return 2
    return run(v, keep=a.keep)


if __name__ == "__main__":
    raise SystemExit(main())
