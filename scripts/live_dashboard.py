#!/usr/bin/env python3
"""Live dashboard for the Irona pick-and-place run. Python standard library only.

Serves the directory that source/live_view.LivePublisher writes, so it can be
started before, during or after a run and never blocks the simulation:

    scripts/live_dashboard.py --directory validation/live --port 8710
    xdg-open http://127.0.0.1:8710

The page polls state.json and reloads both frames a few times a second. The left
column is what the head camera sees with the fitted grasp drawn on it; the right
column is what the robot believes and what it intends to do next.
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--directory', type=Path,
                    default=Path(__file__).resolve().parent.parent / 'validation/live')
parser.add_argument('--host', default='127.0.0.1')
parser.add_argument('--port', type=int, default=8710)
args = parser.parse_args()

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Irona 914 - live</title>
<style>
 :root{color-scheme:dark}
 body{margin:0;background:#0d1013;color:#e7ecf2;font:13px/1.45 ui-monospace,Menlo,Consolas,monospace}
 header{padding:10px 16px;border-bottom:1px solid #232a31;display:flex;gap:16px;align-items:baseline}
 h1{font-size:14px;margin:0;font-weight:600;letter-spacing:.3px}
 #clock{color:#7d8996}
 #stale{color:#ffb454}
 main{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(320px,.95fr);gap:16px;padding:16px}
 img{width:100%;border:1px solid #232a31;border-radius:4px;display:block;background:#05070a}
 .imgs{display:grid;gap:10px}
 .cap{color:#7d8996;margin:2px 0 0}
 section{border:1px solid #232a31;border-radius:4px;padding:10px 12px;margin-bottom:12px}
 h2{font-size:11px;text-transform:uppercase;letter-spacing:.9px;color:#7d8996;margin:0 0 8px}
 .state{font-size:22px;font-weight:600;color:#7ee0a8}
 .state.failed{color:#ff6b6b}
 .state.done{color:#6cc8ff}
 ol{margin:0;padding-left:18px}
 ol li{margin:3px 0}
 ol li:first-child{color:#ffd479}
 table{width:100%;border-collapse:collapse}
 td{padding:2px 0;vertical-align:top}
 td:first-child{color:#7d8996;padding-right:10px;white-space:nowrap}
 .ev{color:#9aa7b4;max-height:190px;overflow:auto}
 .ev div{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
 .truth{color:#5c6672}
</style></head><body>
<header><h1>IRONA 914 &mdash; zero-shot pick and place</h1>
 <span id="clock">waiting for the simulation&hellip;</span><span id="stale"></span></header>
<main>
 <div class="imgs">
  <div><img id="frame" alt="head camera"><p class="cap">head camera, fitted grasp drawn: cyan
   outline = segmented object, orange rod = antipodal contacts, red cross = grasp axis,
   green cross = place spot</p></div>
  <div><img id="depth" alt="depth"><p class="cap">rendered depth, the only geometry the grasp is fitted from</p></div>
 </div>
 <div>
  <section><h2>mission state</h2><div class="state" id="mstate">-</div>
   <table><tr><td>grasp policy</td><td id="policy">-</td></tr>
    <tr><td>sim time</td><td id="t">-</td></tr></table></section>
  <section><h2>next courses of action</h2><ol id="actions"><li>-</li></ol></section>
  <section><h2>what it sees</h2>
   <table>
    <tr><td>detector</td><td id="detector">-</td></tr>
    <tr><td>support height</td><td id="support">-</td></tr>
    <tr><td>clusters</td><td id="clusters">-</td></tr>
    <tr><td>grasp width</td><td id="width">-</td></tr>
    <tr><td>grasp score</td><td id="score">-</td></tr>
    <tr><td>circle fit</td><td id="fit">-</td></tr>
    <tr><td>upright</td><td id="upright">-</td></tr>
   </table></section>
  <section><h2>arm and object</h2>
   <table>
    <tr><td>palm</td><td id="palm">-</td></tr>
    <tr><td>grasp axis</td><td id="centre">-</td></tr>
    <tr><td>place target</td><td id="place">-</td></tr>
    <tr><td>release IK</td><td id="placeres">-</td></tr>
    <tr><td>closure</td><td id="closure">-</td></tr>
    <tr><td>welded</td><td id="attached">-</td></tr>
    <tr><td class="truth">object truth</td><td class="truth" id="truth">-</td></tr>
   </table></section>
  <section><h2>log</h2><div class="ev" id="events"></div></section>
 </div>
</main>
<script>
const $ = id => document.getElementById(id);
const fmt = v => v === null || v === undefined ? '-' : v;
const vec = v => Array.isArray(v) ? '[' + v.map(x => (+x).toFixed(3)).join(', ') + ']' : '-';
let lastUpdate = 0, lastWall = 0;

async function tick() {
  try {
    const response = await fetch('state.json?' + Date.now(), {cache: 'no-store'});
    if (response.ok) {
      const s = await response.json();
      if (s.updates !== lastUpdate) {
        lastUpdate = s.updates; lastWall = Date.now();
        const stamp = Date.now();
        $('frame').src = 'frame.jpg?' + stamp;
        $('depth').src = 'depth.jpg?' + stamp;
        const st = $('mstate');
        st.textContent = s.state; st.className = 'state ' + s.state;
        $('policy').textContent = fmt(s.grasp_policy);
        $('t').textContent = s.t.toFixed(2) + ' s';
        $('clock').textContent = 'update ' + s.updates + ' at t=' + s.t.toFixed(2) + 's';
        $('actions').innerHTML = (s.actions || []).map(a => '<li>' + a + '</li>').join('') || '<li>-</li>';
        $('detector').textContent = fmt(s.detector);
        $('support').textContent = s.support_z === null || s.support_z === undefined
          ? '-' : (+s.support_z).toFixed(3) + ' m';
        const cl = s.clusters || [];
        $('clusters').textContent = cl.length + ' (' + cl.filter(c => c.graspable).length + ' graspable)';
        const g = s.grasp || {};
        $('width').textContent = g.width_mm ? g.width_mm.toFixed(0) + ' mm' : '-';
        $('score').textContent = g.score !== null && g.score !== undefined ? (+g.score).toFixed(2) : '-';
        $('fit').textContent = g.fit_residual_mm !== null && g.fit_residual_mm !== undefined
          ? g.fit_residual_mm.toFixed(2) + ' mm rms' : '-';
        $('upright').textContent = g.upright !== null && g.upright !== undefined
          ? (+g.upright).toFixed(3) : '-';
        $('palm').textContent = vec(s.palm);
        $('centre').textContent = vec(g.centre);
        $('place').textContent = vec(s.place_target);
        $('placeres').textContent = s.place_residual_mm === null || s.place_residual_mm === undefined
          ? '-' : s.place_residual_mm.toFixed(0) + ' mm';
        $('closure').textContent = (+s.closure).toFixed(2);
        $('attached').textContent = s.attached ? ('yes at ' + (+s.attached).toFixed(2) + 's'
          + (s.released ? ', released ' + (+s.released).toFixed(2) + 's' : '')) : 'no';
        $('truth').textContent = vec(s.object_truth);
        $('events').innerHTML = (s.events || []).slice().reverse()
          .map(e => '<div>[' + (+e.t).toFixed(2) + 's] ' + e.state + ' &mdash; ' + e.text + '</div>').join('');
      }
      $('stale').textContent = lastWall && Date.now() - lastWall > 4000
        ? '  (no new frame for ' + ((Date.now() - lastWall) / 1000).toFixed(0) + 's)' : '';
    } else {
      $('clock').textContent = 'waiting for the simulation to publish...';
    }
  } catch (error) {
    $('clock').textContent = 'dashboard error: ' + error;
  }
  setTimeout(tick, 300);
}
tick();
</script></body></html>
"""

TYPES = {'.jpg': 'image/jpeg', '.json': 'application/json'}


class Handler(BaseHTTPRequestHandler):
    def _send(self, body, content_type, code=200):
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def do_GET(self):
        path = self.path.split('?', 1)[0]
        if path in ('/', '/index.html'):
            self._send(PAGE.encode(), 'text/html; charset=utf-8')
            return
        name = Path(path).name
        if name not in ('frame.jpg', 'depth.jpg', 'state.json', 'summary.json'):
            self._send(b'not found', 'text/plain', 404)
            return
        target = args.directory / name
        if not target.is_file():
            # Not an error: the run may not have published its first frame yet.
            self._send(json.dumps(dict(waiting=True)).encode() if name.endswith('.json') else b'',
                       TYPES[target.suffix], 404)
            return
        try:
            self._send(target.read_bytes(), TYPES[target.suffix])
        except OSError as error:
            self._send(str(error).encode(), 'text/plain', 503)

    def log_message(self, *arguments):   # keep the console for the simulation's own output
        pass


if __name__ == '__main__':
    args.directory.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'live dashboard on http://{args.host}:{args.port}  serving {args.directory}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nstopped', flush=True)
