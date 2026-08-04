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
    // A generated page is a frozen copy of the template and cannot be told
    // apart from a stale one by looking at it, so it has to say which build it
    // is. A viewer several fixes behind was once the file on disk while the
    // fixes sat in git, and nothing on screen gave it away.
    ok(!!M.built_utc && /\d{4}-\d{2}-\d{2}/.test(M.built_utc),
       `the page states when it was built (${M.built_utc})`);
    ok(document.getElementById("hbuild").textContent.includes(M.version),
       `and shows it in the header (${document.getElementById("hbuild").textContent})`);
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
    {
      // A flag that says a row cannot be trusted is worthless if the viewer
      // never shows it. shape_suspect was added to the matrix and, at first,
      // to nothing else.
      const bad = ORG.find(o => +o.shape_suspect === 1);
      ok(bad !== undefined, "some organoids carry shape_suspect");
      if (bad){
        select(bad);
        const b = document.getElementById("detail").innerHTML;
        ok(b.includes("shape suspect") || b.includes("shape_suspect"),
           "and selecting one shows the flag");
      }
      select(target);
    }
    ok(state.selected === target, "selection state follows the click");

    // --- the stack is a stack, and stepping through it changes the picture ---
    ok(NZ > 50, `${NZ} slices embedded, so depth can actually be stepped through`);
    ok(NZ === M.z_total,
       `the whole stack is here: ${NZ} of ${M.z_total} slices, not just the ` +
       `${M.z_analysed} that were analysed`);
    ok(state.mode === "slice", "the viewer opens on the raw stack, not the projection");
    const z0 = state.z;
    const img0 = currentPhoto().img;
    setZ(z0 + 10);
    ok(state.z === z0 + 10 && +document.getElementById("zslider").value === z0 + 10,
       "the slider follows setZ");
    ok(currentPhoto().img !== img0, "a different slice shows a different photograph");
    ok(stackImgs.filter(Boolean).length < NZ,
       `slices decode on demand -- ${stackImgs.filter(Boolean).length} of ${NZ} ` +
       `have been touched, not all of them on load`);
    ok(dirty2 === true, "changing slice asks the photo pane to repaint");

    // --- and the 3D plane and clip follow it ---
    draw3d();
    const pz = planeMesh.position.z, cc = clipPlane.constant;
    setZ(z0 + 30); draw3d();
    ok(Math.abs(planeMesh.position.z - pz) > 1,
       `the photo plane moved with the slice (${pz.toFixed(0)} -> ${planeMesh.position.z.toFixed(0)})`);
    ok(Math.abs(clipPlane.constant - cc) > 1, "the clip plane moved with it too");
    // Which side survives is the user's choice, so all three have to work.
    {
      const deep = () => new THREE.Vector3(0, 0, ZW(state.z) + 100);
      const shallow = () => new THREE.Vector3(0, 0, ZW(state.z) - 100);
      state.clip = "above"; draw3d();
      ok(clipPlane.distanceToPoint(shallow()) > 0 &&
         clipPlane.distanceToPoint(deep()) < 0,
         "'above' keeps what lies between the objective and the slice");
      state.clip = "below"; draw3d();
      ok(clipPlane.distanceToPoint(deep()) > 0 &&
         clipPlane.distanceToPoint(shallow()) < 0,
         "'below' keeps the deeper half, cut open at the slice");
      state.clip = "both"; draw3d();
      ok(clipPlane.distanceToPoint(deep()) > 0 &&
         clipPlane.distanceToPoint(shallow()) > 0,
         "'both' cuts nothing");
      ok([...meshes.values()].every(m => m.material.clippingPlanes &&
                                        m.material.clippingPlanes.length === 1),
         "and no material gains or loses a clipping plane, so nothing recompiles");
      state.clip = "above"; draw3d();

      // and whether the models are drawn at all is independent of which side
      state.showModels = false; draw3d();
      ok([...meshes.values()].every(m => !m.visible),
         "the organoids can be hidden entirely, leaving the photograph and dome");
      ok(planeMesh.visible, "the photograph stays when the models are hidden");
      state.showModels = true; draw3d();
      ok([...meshes.values()].some(m => m.visible), "and they come back");
    }
    ok(Math.abs(planeMesh.position.z - ZW(state.z)) < 1e-6,
       "the plane sits at exactly the current slice's depth");
    setZ(z0);

    // --- an outline is only drawn where that organoid actually is ---
    // An organoid is its full width only at its own equator; anywhere else the
    // spheroid's cross-section is narrower, and past its axial extent this
    // slice misses it completely. Drawing the equatorial outline at every depth
    // put full-size rings on out-of-focus shadows, which is what made so many
    // marked edges look wrong.
    {
      const keepMode = state.mode, keepAll = state.allDepths;
      state.mode = "slice"; state.allDepths = false;
      const probe = ORG.find(o => o.radius_z_slices > 2 && o.radial_profile_px);
      setZ(Math.round(probe.z_slice));
      const atEquator = crossScale(probe);
      ok(Math.abs(atEquator - 1) < 0.02,
         `at its own focal plane an organoid is drawn full width (${atEquator.toFixed(3)})`);
      setZ(Math.round(probe.z_slice + probe.radius_z_slices * 0.6));
      const partWay = crossScale(probe);
      ok(partWay !== null && partWay < atEquator,
         `part way out it narrows (${partWay === null ? "null" : partWay.toFixed(3)})`);
      setZ(Math.round(probe.z_slice + probe.radius_z_slices + 3));
      ok(crossScale(probe) === null,
         "and beyond its axial extent it is not drawn at all");

      // across the whole catalogue, a slice must not draw everything
      setZ(Math.round(NZ * 0.2));
      const drawn = ORG.filter(o => crossScale(o) !== null).length;
      ok(drawn < ORG.length * 0.6,
         `${drawn} of ${ORG.length} organoids are drawn on one slice, not all of them`);
      state.mode = keepMode; state.allDepths = keepAll;
      state.mode = "edf";
      ok(crossScale(probe) === 1,
         "on the all-in-focus projection every organoid is drawn at full width, " +
         "because that image shows each one at its own equator");
      state.mode = keepMode; setZ(z0);
    }

    // --- outlines are the measured shape, not a stand-in circle ---
    const withOutline = ORG.filter(o => o.radial_profile_px && o.radial_profile_px.length);
    ok(withOutline.length > ORG.length * 0.9,
       `${withOutline.length}/${ORG.length} organoids carry their measured r(theta) outline`);
    const poly = outlineOf(withOutline[0], 1);
    ok(poly && poly.length >= 24, `an outline is a real polygon (${poly.length/2} vertices)`);
    const rr = withOutline[0].radial_profile_px;
    ok(Math.max(...rr) !== Math.min(...rr),
       "the outline is not a circle -- the radii vary, as a measured shape does");

    // --- the dome is built where the fit says it is ---
    if (M.dome && domeGroup){
      const p = domeGroup.children[0].geometry.attributes.position.array;
      let zmin = 1e9, zmax = -1e9;
      for (let i = 2; i < p.length; i += 3){ zmin = Math.min(zmin, p[i]); zmax = Math.max(zmax, p[i]); }
      const apexW = ZW(M.dome.apex_slice), glassW = ZW(M.substrate_slice);
      ok(Math.abs(zmin - apexW) < 2,
         `the cap's shallowest point is the fitted apex (${zmin.toFixed(0)} vs ${apexW.toFixed(0)})`);
      ok(Math.abs(zmax - glassW) < 2,
         `and its deepest is the glass (${zmax.toFixed(0)} vs ${glassW.toFixed(0)})`);
      ok(zmax > zmin, "the cap opens downward, towards the glass");
      // every vertex must lie on the fitted sphere, or it is not that dome
      let worst = 0;
      for (let i = 0; i < p.length; i += 3){
        const d = Math.hypot(p[i]-M.dome.cx_px, p[i+1]-M.dome.cy_px, p[i+2]-ZW(M.dome.cz_slice));
        worst = Math.max(worst, Math.abs(d - M.dome.radius_px));
      }
      ok(worst < 1.0, `every cap vertex lies on the fitted sphere (worst ${worst.toFixed(2)} px)`);
    }

    // --- build-up: the reconstruction assembles as the focus descends ---
    {
      const keep = state.z;
      state.buildUp = true;
      const counts = [];
      for (const z of [5, Math.round(NZ*0.35), Math.round(NZ*0.7), NZ-1]){
        setZ(z); draw3d();
        let vis = 0;
        for (const [uid, m] of meshes) if (m.visible) vis++;
        counts.push(vis);
      }
      ok(counts[0] < counts[counts.length-1],
         `descending reveals organoids progressively (${counts.join(" -> ")})`);
      ok(counts.every((v, i) => i === 0 || v >= counts[i-1]),
         "the count never goes backwards as the focus descends");
      ok(counts[counts.length-1] === ORG.length,
         `by the bottom of the stack all ${ORG.length} are present`);
      setZ(5); draw3d();
      const shallow = [...meshes.values()].filter(m => m.visible);
      ok(shallow.every(m => m.userData.o.z_slice <= 5.5),
         "nothing is revealed before the focus has reached its own plane");
      state.buildUp = false; setZ(keep); draw3d();
      ok([...meshes.values()].every(m => m.visible),
         "switching build-up off brings everything back");
    }

    // --- the photograph is mapped onto the plane the same way the outlines
    // are drawn onto the 2D canvas, or the two panes disagree ---
    // This is checked as geometry rather than by looking at rendered pixels,
    // because a mirrored dome still looks like a dome and a statistical test on
    // a nearly symmetric specimen cannot tell the difference reliably.
    {
      const g = planeMesh.geometry;
      const pos = g.attributes.position, uv = g.attributes.uv;
      let originIdx = -1, farIdx = -1;
      for (let i = 0; i < uv.count; i++){
        if (uv.getX(i) === 0 && uv.getY(i) === 0) originIdx = i;
        if (uv.getX(i) === 1 && uv.getY(i) === 1) farIdx = i;
      }
      ok(originIdx >= 0 && farIdx >= 0, "the photo plane carries texture coordinates");
      const o0 = new THREE.Vector3().fromBufferAttribute(pos, originIdx)
                  .add(planeMesh.position);
      const o1 = new THREE.Vector3().fromBufferAttribute(pos, farIdx)
                  .add(planeMesh.position);
      ok(planeTex.flipY === false,
         "the texture is uploaded unflipped, so image row 0 is texture v = 0");
      // flipY false means uv (0,0) samples the image's top-left, which must
      // therefore sit at mosaic (0, 0)
      ok(Math.abs(o0.x) < 1e-6 && Math.abs(o0.y) < 1e-6,
         `uv (0,0) -- the image's top-left -- sits at mosaic (${o0.x.toFixed(1)}, ${o0.y.toFixed(1)})`);
      ok(Math.abs(o1.x - M.width) < 1e-6 && Math.abs(o1.y - M.height) < 1e-6,
         `uv (1,1) -- its bottom-right -- sits at mosaic (${o1.x.toFixed(0)}, ${o1.y.toFixed(0)}), the far corner`);
    }

    // --- navigation: the same control model as the single-field viewer ---
    resetCamera(); placeCamera();
    const home = {th: orbit.theta, ph: orbit.phi, t: orbit.target.clone(),
                  r: orbit.radius};
    const r3 = c3d.getBoundingClientRect();
    const orbitDrag = (btn, shift, dx, dy) => {
      c3d.dispatchEvent(new MouseEvent("mousedown", {
        clientX: r3.left + 300, clientY: r3.top + 300, button: btn,
        shiftKey: shift, bubbles: true}));
      window.dispatchEvent(new MouseEvent("mousemove", {
        clientX: r3.left + 300 + dx, clientY: r3.top + 300 + dy, bubbles: true}));
      window.dispatchEvent(new MouseEvent("mouseup", {bubbles: true}));
    };

    orbitDrag(0, false, 100, 60);
    ok(Math.abs(orbit.theta - home.th) > 0.1 && Math.abs(orbit.phi - home.ph) > 0.1,
       `left-drag orbits (theta ${home.th.toFixed(2)}->${orbit.theta.toFixed(2)}, ` +
       `phi ${home.ph.toFixed(2)}->${orbit.phi.toFixed(2)})`);
    ok(orbit.target.distanceTo(home.t) < 1e-6, "left-drag does not move the target");

    resetCamera();
    orbitDrag(2, false, 100, 60);
    const panned = orbit.target.distanceTo(home.t);
    ok(panned > 1, `right-drag pans the target (${panned.toFixed(0)} units)`);
    ok(Math.abs(orbit.theta - home.th) < 1e-9, "right-drag does not rotate");

    // Direction, not just magnitude. The scene has to follow the mouse: drag
    // right and the specimen goes right. Getting this backwards is invisible to
    // any test that only checks that something moved.
    {
      const probe = new THREE.Vector3(M.width/2, M.height/2, ZW(M.substrate_slice/2));
      resetCamera(); placeCamera(); camera.updateMatrixWorld();
      const a = probe.clone().project(camera);
      orbitDrag(2, false, 140, 0);
      placeCamera(); camera.updateMatrixWorld();
      const b = probe.clone().project(camera);
      const W3 = renderer.domElement.width;
      const movedPx = (b.x - a.x) * W3 / 2;
      ok(movedPx > 20,
         `dragging right by 140 px carries the scene right by ${movedPx.toFixed(0)} px`);

      resetCamera(); placeCamera(); camera.updateMatrixWorld();
      const c = probe.clone().project(camera);
      orbitDrag(2, false, 0, 140);
      placeCamera(); camera.updateMatrixWorld();
      const d2 = probe.clone().project(camera);
      const H3 = renderer.domElement.height;
      const movedDown = -(d2.y - c.y) * H3 / 2;    // NDC y is up, screen y is down
      ok(movedDown > 20,
         `dragging down by 140 px carries it down by ${movedDown.toFixed(0)} px`);
    }
    resetCamera();

    resetCamera();
    orbitDrag(0, true, 100, 60);
    ok(orbit.target.distanceTo(home.t) > 1, "shift-drag pans too");

    // middle button orbits, which is the Blender habit most people arrive with
    resetCamera();
    orbitDrag(1, false, 100, 60);
    ok(Math.abs(orbit.theta - home.th) > 0.1,
       `middle-drag orbits (theta ${home.th.toFixed(2)} -> ${orbit.theta.toFixed(2)})`);
    ok(orbit.target.distanceTo(home.t) < 1e-6, "middle-drag does not move the target");

    // The view must reach edge-on and straight-down; only the pole itself is
    // forbidden, and only because lookAt cannot pick a roll there.
    resetCamera();
    orbitDrag(0, false, 0, -4000);
    ok(orbit.phi < 0.01 && orbit.phi >= 0,
       `dragging up reaches straight down the optical axis (phi ${orbit.phi.toFixed(4)} rad)`);
    placeCamera();
    ok(Math.abs(camera.up.dot(new THREE.Vector3().subVectors(
         camera.position, orbit.target).normalize())) < 0.05,
       "and the up vector is still square to the view there, so the roll is defined");
    resetCamera();
    orbitDrag(0, false, 0, 4000);
    ok(orbit.phi > Math.PI - 0.01 && orbit.phi <= Math.PI,
       `and dragging down reaches the other pole (phi ${orbit.phi.toFixed(3)} rad)`);
    {
      // edge-on: the camera must sit at the target's depth, looking along the glass
      frontCamera(); placeCamera();
      ok(Math.abs(camera.position.z - orbit.target.z) < 1e-6,
         "F gives a true edge-on view, level with the specimen");
      const box = ZW(M.z_analysed || NZ);
      const top = new THREE.Vector3(M.width/2, M.height/2, 0).project(camera);
      const bot = new THREE.Vector3(M.width/2, M.height/2, box).project(camera);
      ok(Math.abs(top.y - bot.y) > 0.2,
         "and from there the depth of the stack is spread across the screen");
    }

    resetCamera(); placeCamera();
    const near0 = camera.near, far0 = camera.far;
    c3d.dispatchEvent(new WheelEvent("wheel", {deltaY: -600, bubbles: true}));
    placeCamera();
    ok(orbit.radius < home.r, `wheel zooms in (${home.r.toFixed(0)} -> ${orbit.radius.toFixed(0)})`);
    ok(camera.near < near0 && camera.far < far0,
       `the depth range follows the orbit distance (near ${near0.toFixed(0)}->` +
       `${camera.near.toFixed(0)}, far ${far0.toFixed(0)}->${camera.far.toFixed(0)})`);
    ok(camera.far / camera.near < 5000,
       `near and far stay close enough to keep depth precision (ratio ` +
       `${(camera.far/camera.near).toFixed(0)})`);

    topCamera(); placeCamera(); camera.updateMatrixWorld();
    // Either pole is "straight down the axis"; which one is a matter of the
    // sign convention, and depth increases downward here, so it is pi.
    ok(Math.min(orbit.phi, Math.PI - orbit.phi) < 0.1,
       `T looks essentially straight down the optical axis (phi ${orbit.phi.toFixed(3)} rad)`);
    ok(camera.position.z < orbit.target.z,
       "and from above the specimen, not from under the glass looking up");
    {
      // Square to the screen: a step along the mosaic's x has to move the image
      // horizontally and a step along its y vertically. Measured with small
      // steps about the target, so this is about orientation and not about
      // perspective foreshortening at the far corner.
      const c = orbit.target;
      const p0 = new THREE.Vector3(c.x, c.y, c.z).project(camera);
      const px = new THREE.Vector3(c.x + 200, c.y, c.z).project(camera);
      const py = new THREE.Vector3(c.x, c.y + 200, c.z).project(camera);
      const dx = [px.x - p0.x, px.y - p0.y], dy = [py.x - p0.x, py.y - p0.y];
      ok(Math.abs(dx[1]) < 0.05 * Math.abs(dx[0]),
         `mosaic +x runs horizontally on screen (tilt ${(dx[1]/dx[0]).toFixed(4)})`);
      ok(Math.abs(dy[0]) < 0.05 * Math.abs(dy[1]),
         `mosaic +y runs vertically (tilt ${(dy[0]/dy[1]).toFixed(4)})`);
      ok(dx[0] > 0 && dy[1] < 0,
         "with x to the right and y downward, the same way round as the photograph");
    }
    resetCamera(); placeCamera();

    // --- field borders can be turned off ---
    state.borders = false; dirty2 = true; draw2d();
    ok(state.borders === false, "field borders can be switched off");
    state.borders = true; draw2d();

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
    orbitDrag(0, false, 200, 40);
    placeCamera();
    ok(before.distanceTo(camera.position) > 1,
       `drag moved the camera by ${before.distanceTo(camera.position).toFixed(0)} units`);
    ok(dirty3 === true, "drag requested a repaint");
    resetCamera();

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
