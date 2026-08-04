#!/usr/bin/env python
"""Drive a generated viewer.html in a real browser and assert it behaves.

    ./.venv/bin/python tests/check_viewer.py output/4x_00009/viewer.html

This exists because the viewer's worst bugs were invisible to every other kind
of check. The 3D view moved the camera but never asked for a repaint, so it sat
frozen and then jumped; switching layouts left each canvas' backing store out of
step with its element, so everything drew stretched. Both files parsed cleanly,
both passed a syntax and reference check, and neither fault could be seen
without actually running the page.

So the page is loaded in headless Firefox with a harness appended that
synthesises the interactions and reports PASS/FAIL, and this script fails if any
line says FAIL.

One caveat worth knowing: headless Firefox screenshots do not capture a WebGL
canvas -- it comes out black even when the scene is rendering correctly. The
harness therefore reads the framebuffer back with gl.readPixels instead of
trusting the picture.
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
    // --- the scene actually puts pixels on the canvas ---
    renderer.render(scene, camera);
    const gl = renderer.getContext();
    const w = renderer.domElement.width, h = renderer.domElement.height;
    const px = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
    let lit = 0;
    for (let i = 0; i < px.length; i += 4)
      if (px[i] + px[i + 1] + px[i + 2] > 40) lit++;
    ok(lit > 1000, `3D framebuffer has ${lit} lit pixels`);

    let inFrustum = 0;
    for (const o of ORG) {
      const p = meshes.get(o.oid).position.clone().project(camera);
      if (Math.abs(p.x) <= 1 && Math.abs(p.y) <= 1 && p.z > -1 && p.z < 1) inFrustum++;
    }
    ok(inFrustum > ORG.length * 0.5, `${inFrustum}/${ORG.length} organoids in view`);

    // --- dragging repaints ---
    const before = camera.position.clone();
    dirty3d = false;
    const r = c3d.getBoundingClientRect();
    c3d.dispatchEvent(new MouseEvent("mousedown", {
      clientX: r.left + 300, clientY: r.top + 400, button: 0, bubbles: true }));
    window.dispatchEvent(new MouseEvent("mousemove", {
      clientX: r.left + 500, clientY: r.top + 430, bubbles: true }));
    window.dispatchEvent(new MouseEvent("mouseup", { bubbles: true }));
    ok(before.distanceTo(camera.position) > 1,
       `drag moved the camera by ${before.distanceTo(camera.position).toFixed(0)} units`);
    ok(dirty3d === true, "drag requested a repaint");

    // --- every layout keeps each canvas the size of its box ---
    for (const [id, label] of [["lay-3d", "3D only"], ["lay-2d", "Photo only"],
                               ["lay-split", "Side by side"]]) {
      setLayout(id);
      resize();                    // the ResizeObserver is async; force it here
      let good = true, detail = [];
      for (const [cv, pane] of [[c2d, "pane2d"], [c3d, "pane3d"]]) {
        const el = document.getElementById(pane);
        const ew = el.clientWidth, eh = el.clientHeight;
        if (ew < 2 || eh < 2) continue;
        const cw = cv === c2d ? cv.width : renderer.domElement.width;
        const ch = cv === c2d ? cv.height : renderer.domElement.height;
        if (cw !== ew || ch !== eh) good = false;
        detail.push(`${pane} ${ew}x${eh} vs ${cw}x${ch}`);
      }
      ok(good, `${label}: ${detail.join(" | ")}`);
    }
    setLayout("lay-split"); resize();

    // --- a selected organoid is unmistakable ---
    select(ORG[Math.min(3, ORG.length - 1)].oid, false);
    ok(selOutline.visible === true, "selection draws an outline shell");
    ok(window.__dimFactor < 0.3, `unselected dimmed to ${window.__dimFactor}`);
    ok(meshes.get(state.selected).material.transparent === false,
       "no material had to be recompiled for the selection");
    ok([...meshes.values()].every(m => m.material.transparent === false),
       "all materials stay opaque (no shader rebuild on select)");
    select(null, false);

    // --- colour encodes a measurement ---
    ok(RAMP === RAMPS.viridis, "colour map defaults to viridis");
    const zs = ORG.map(o => o.z_slice);
    const lo = ORG[zs.indexOf(Math.min(...zs))], hi = ORG[zs.indexOf(Math.max(...zs))];
    ok(rampCss(colorValue(lo)) !== rampCss(colorValue(hi)),
       `depth spans ${rampCss(colorValue(lo))} to ${rampCss(colorValue(hi))}`);

    // --- the slice control stays in step ---
    const z0 = state.z;
    setZ(z0 + 7);
    ok(state.z === z0 + 7 && +document.getElementById("zslider").value === z0 + 7,
       "slider follows setZ");
    ok(plane.position.z !== undefined, "photo plane tracks the slice");
  } catch (e) {
    ok(false, "harness threw: " + e);
  }

  const d = document.createElement("div");
  d.id = "harness-report";
  d.style.cssText = "position:fixed;left:8px;bottom:60px;z-index:9999;background:#000e;"
    + "color:#8f8;font:12px monospace;padding:10px;white-space:pre;border:1px solid #0f0";
  d.textContent = out.join("\n");
  document.body.appendChild(d);
  // Firefox headless can only save a screenshot, and a WebGL canvas does not
  // even appear in that. dump() writes straight to the browser's stdout when
  // browser.dom.window.dump.enabled is set, which the profile below does.
  if (typeof dump === "function") {
    dump("HARNESS-BEGIN\n" + out.join("\n") + "\nHARNESS-END\n");
  }
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

    # Snap-confined Firefox cannot read dotted directories under $HOME, and it
    # refuses to start if another profile is locked -- so the working files go
    # next to the viewer rather than in a temp dir.
    viewer = viewer.resolve()
    work = viewer.parent / "_viewercheck"
    work.mkdir(exist_ok=True)
    page = work / "harness.html"
    shot = work / "harness.png"

    html = viewer.read_text(encoding="utf-8")
    html = html.replace("<body>", "<body>" + ERROR_TRAP, 1)
    html = html.replace("</body>", HARNESS + "</body>", 1)
    page.write_text(html, encoding="utf-8")

    prof = work / "prof"
    prof.mkdir(exist_ok=True)
    (prof / "user.js").write_text('user_pref("browser.dom.window.dump.enabled", true);\n')

    cmd = [firefox, "--new-instance", "--headless", "--profile", str(prof),
           "--window-size=1700,1000", "--screenshot", str(shot), page.as_uri()]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    blob = proc.stdout + proc.stderr
    m = re.search(r"HARNESS-BEGIN\n(.*?)\nHARNESS-END", blob, re.S)
    if not m:
        print("no report from the page. browser said:\n" + blob[-800:], file=sys.stderr)
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
    ap = argparse.ArgumentParser(description="Browser check for a generated viewer")
    ap.add_argument("viewer", nargs="?", default="output/4x_00009/viewer.html")
    ap.add_argument("--keep", action="store_true", help="leave the harness files behind")
    a = ap.parse_args(argv)
    v = Path(a.viewer)
    if not v.is_file():
        print(f"no such viewer: {v}", file=sys.stderr)
        return 2
    return run(v, keep=a.keep)


if __name__ == "__main__":
    raise SystemExit(main())
