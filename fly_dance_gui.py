"""Ado-blue dance GUI. stdlib http.server only -- no web framework.

The sim runs in a thread and streams progress as JSON; the page renders a
live canvas of the fly plus the beat timeline. No Flask, no websockets, no
npm. Open the printed URL.

    python fly_dance_gui.py Monstruo-Ado.mp3
"""

import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

import mujoco as mj
import numpy as np

SRC = next((a for a in sys.argv[1:] if not a.startswith("-")), "Monstruo-Ado.mp3")
PORT = int(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--port=")), 8765))

RAMP = 0.09
LOOKAHEAD = 0.09
POP = 1.25
SIGMA = 0.6
LEARN_RATE = 0.004
REWARD_SCALE = 400.0
BEAT_WIN = 0.06
GRAD_CLIP = 0.02

STATE = {"status": "idle", "t": 0.0, "beats": 0, "w2": 0.0, "sync": 0.0,
         "leg": [0.0] * 6, "body": [0.0] * 12, "on_beat": False, "frame": 0}


def _grab_frame(sim):
    """Render the real MuJoCo 3D view to JPEG for the browser."""
    import imageio.v3 as iio
    try:
        rgb = sim.renderer.mj_renderer.render()
        iio.imwrite("/tmp/ado_frame.jpg", rgb, extension=".jpg")
        STATE["frame"] += 1
    except Exception:
        pass          # ponytail: a dropped preview frame is not worth crashing over


def _danceable(dof):
    """Wings, halteres, abdomen. Skip head/rostrum/eyes -- nothing to dance with."""
    s = f"{dof.parent.link} {dof.child.link} {dof.axis}"
    return any(k in s for k in ("wing", "haltere", "abdomen"))


def _dress_world(world):
    """Give the empty TetheredWorld a real outdoor scene: ground, pebbles, two
    lights. ponytail: flat rgba materials only -- MjSpec.add_texture corrupts the
    heap on a TetheredWorld (double-free at compile), so skip real textures
    until flygym stops doing that.
    """
    r = world.mjcf_root
    r.add_material(name="grass_m", rgba=[.16, .30, .13, 1], specular=.05,
                   shininess=.05)
    r.add_material(name="stone_m", rgba=[.52, .52, .55, 1], specular=.2)
    b = r.worldbody
    g = b.add_geom()
    g.name, g.type, g.material = "ground", mj.mjtGeom.mjGEOM_PLANE, "grass_m"
    g.size = np.array([0.0, 0.0, .1])       # plane: only the 3rd value is used
    g.friction = [1, .005, .0001]
    rng = np.random.default_rng(0)
    for i in range(12):                      # pebbles, deterministic layout
        p = b.add_geom()
        p.name, p.type, p.material = f"pebble{i}", mj.mjtGeom.mjGEOM_SPHERE, "stone_m"
        p.size = rng.uniform(.002, .006, 3)
        p.pos = np.append(rng.uniform(-.12, .12, 2), 0.0)
    for name, pos, dirn, dif, spc in (
            ("sun", [0, 0, .3], [-.3, 0, -1], [.9, .88, .8], [.15, .15, .12]),
            ("fill", [0, -.2, .1], [0, 1, -.3], [.25, .32, .45], [0, 0, 0])):
        l = b.add_light()
        l.name = name
        l.pos, l.dir = np.asarray(pos, float), np.asarray(dirn, float)
        l.diffuse, l.specular = np.asarray(dif, float), np.asarray(spc, float)


class Sim:
    """Tethered fly with wings. Runs in a thread; pushes progress into STATE."""

    def __init__(self, audio, secs):
        self.audio, self.secs = audio, secs
        self.beats, self.env, self.hop, self.sr = None, None, 512, 22050

    def load(self):
        bp = Path(str(self.audio) + ".beats.json")
        if not bp.exists():
            import beat_analysis
            bp.write_text(json.dumps(beat_analysis.analyze(self.audio)))
        b = json.loads(bp.read_text())
        self.beats = np.asarray(b["beat_times"], float)
        e = np.asarray(b["onset_env"], float)
        self.env = (e - e.min()) / (np.ptp(e) + 1e-9)
        self.hop, self.sr = int(b["onset_hop"]), int(b["sr"])
        self.bpm = float(b["tempo_bpm"])
        return self

    def build(self):
        from flygym import Simulation
        from flygym.anatomy import ActuatedDOFPreset, AxisOrder, JointPreset, Skeleton
        from flygym.compose import (
            ActuatorType, KinematicPosePreset, NeuroMechFly, TetheredWorld,
        )
        from flygym.utils.math import Rotation3D

        # ponytail: fork of make_locomotion_fly that keeps the wings. The stock
        # builder hardcodes JointPreset.LEGS_ONLY, so wing joints never exist in
        # the skeleton and add_actuators silently returns joints with no joint.
        neutral = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(AxisOrder.YAW_PITCH_ROLL)
        skel = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL,
                        joint_preset=JointPreset.ALL_BIOLOGICAL)
        self.fly = NeuroMechFly(name="dancer")
        joints = self.fly.add_joints(skel, neutral_pose=neutral,
                                     stiffness=0.05, damping=0.06)
        passive = ("tarsus5",)     # tarsal spring, same as the stock builder
        for dof, joint in joints.items():
            if any(t in dof.child.link for t in passive):
                joint.stiffness[0], joint.damping[0] = 7.5, 1e-2
        legs = set(map(str, skel.get_actuated_dofs_from_preset(
            ActuatedDOFPreset.LEGS_ACTIVE_ONLY)))
        pos = [d for d in skel.get_actuated_dofs_from_preset(ActuatedDOFPreset.ALL)
               if str(d) in legs or _danceable(d)]
        self.fly.add_actuators(pos, ActuatorType.POSITION, neutral_input=neutral,
                               kp=45.0, forcerange=(-65.0, 65.0))
        self.fly.add_leg_adhesion(gain=40.0)
        self.fly.colorize()

        self.world = TetheredWorld()
        _dress_world(self.world)
        # camera on the fly spec BEFORE Simulation compiles the world
        self.cam = self.fly.add_tracking_camera(
            name="body_cam", pos_offset=(-0.5, -7.5, 0.0),
            rotation=Rotation3D("euler", (1.57, 0.0, 0.0)), fovy=30.0)
        self.world.add_fly(self.fly, [0.0, 0.0, 0.5],
                           Rotation3D("quat", [1.0, 0.0, 0.0, 0.0]))
        self.sim = Simulation(self.world)
        from flygym_demo.complex_terrain import PreprogrammedSteps
        self.steps = PreprogrammedSteps()

        all_pos = self.fly.get_actuated_jointdofs_order("position")
        bodyset = set(map(str, [d for d in pos if _danceable(d)]))
        self.leg_dofs = [d for d in all_pos if str(d) not in bodyset]
        self.body_dofs = [d for d in all_pos if str(d) in bodyset]
        # precompute (group, index) per actuator so the hot loop is one pass
        self.slot = [(1, self.body_dofs.index(d)) if str(d) in bodyset
                     else (0, self.leg_dofs.index(d)) for d in all_pos]
        return self

    def run(self, policy, render=True):
        """Dance, streaming progress into STATE. Reward is beat-aligned, so it
        learns to hit the beat rather than just flail faster."""
        from flygym.compose import ActuatorType
        from flygym.utils.math import Rotation3D
        sim, fly, steps = self.sim, self.fly, self.steps
        nb = len(self.body_dofs)

        if render:
            sim.set_renderer([self.cam])
        sim.reset()
        a0 = (steps.default_pose_by_dof_order(self.leg_dofs), np.zeros(nb))
        sim.set_actuator_inputs(fly.name, ActuatorType.POSITION,
                                np.fromiter((a0[g][k] for g, k in self.slot),
                                            dtype=float, count=len(self.slot)))
        sim.set_leg_adhesion_states(fly.name, np.ones(6, dtype=bool))
        sim.warmup()

        n = int(self.secs / sim.timestep)
        leg_goal = leg_now = np.full(6, np.pi)
        body_goal = body_now = np.zeros(nb)
        bi, prev_q = 0, None
        speeds, inwin = [], []
        cur_x = cur_a = None
        pending = {}

        for i in range(n):
            t = i * sim.timestep
            if bi < len(self.beats) and t >= self.beats[bi] - LOOKAHEAD:
                k = min(int(self.beats[bi] * self.sr / self.hop), self.env.size - 1)
                cur_x = np.array([bi % 4, float(self.env[k]), t / max(self.secs, 1e-6)])
                cur_a, _ = policy.act(cur_x)
                leg_goal, body_goal = cur_a[:6], cur_a[6:]
                bi += 1

            f = min(1.0, sim.timestep / RAMP)
            leg_now = (leg_now + ((leg_goal - leg_now + np.pi) % (2 * np.pi) - np.pi) * f) % (2 * np.pi)
            body_now = (body_now + ((body_goal - body_now + np.pi) % (2 * np.pi) - np.pi) * f) % (2 * np.pi)

            ang = (steps.get_joint_angles_by_dof_order(leg_now, np.full(6, POP), self.leg_dofs),
                   (body_now - np.pi) * POP)
            sim.set_actuator_inputs(fly.name, ActuatorType.POSITION,
                                    np.fromiter((ang[g][k] for g, k in self.slot),
                                                dtype=float, count=len(self.slot)))
            sim.set_leg_adhesion_states(fly.name,
                                        steps.get_adhesion_onoff_by_phase(leg_now))
            sim.step()

            q = sim.get_joint_angles(fly.name)
            sp = 0.0 if prev_q is None else float(np.abs(q - prev_q).sum() / sim.timestep)
            prev_q = q.copy()
            speeds.append(sp)

            on = bi > 0 and abs(t - self.beats[bi - 1]) < BEAT_WIN
            inwin.append(on)
            if on:
                pending[bi - 1] = min(sp / REWARD_SCALE, 1.0)
            elif pending and cur_x is not None:
                policy.reinforce(cur_x, cur_a, pending.pop(max(pending)))
                pending = {}

            if i % 200 == 0:
                STATE.update(status="dancing", t=t, beats=bi,
                             w2=float(np.abs(policy.W2).sum()),
                             leg=leg_now.round(3).tolist(),
                             body=body_now[:12].round(3).tolist(),
                             on_beat=bool(on))
                if render:
                    sim.render_as_needed_with_profile()
                    _grab_frame(sim)

        s, iw = np.array(speeds), np.array(inwin)
        ratio = float(s[iw].mean() / max(s[~iw].mean(), 1e-6)) if iw.any() and (~iw).any() else 0.0
        STATE.update(status="done", beats=bi, sync=ratio, w2=float(np.abs(policy.W2).sum()))
        if render:
            out = Path("demo_output/fly_dance")
            out.mkdir(parents=True, exist_ok=True)
            sim.renderer.save_video(out / "ado_blue.mp4")
        return ratio


class Policy:
    """numpy MLP (3 -> 16 -> 6+nb) trained by REINFORCE. No torch, no
    hand-written choreography -- it picks its own leg poses and wing angles."""

    def __init__(self, n_out, hidden=16, seed=0):
        rng = np.random.default_rng(seed)
        self.W1, self.b1 = rng.normal(0, .3, (3, hidden)), np.zeros(hidden)
        self.W2, self.b2 = rng.normal(0, .3, (hidden, n_out)), np.zeros(n_out)

    def forward(self, x):
        h = np.tanh(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2, h

    def act(self, x):
        mu, _ = self.forward(x)
        a = mu + np.random.normal(0, SIGMA, mu.shape)
        z = (a - mu) / SIGMA
        return a % (2 * np.pi), float(-0.5 * (z * z).sum())

    def reinforce(self, x, a, reward):
        """reward in 0..1: >0.5 reinforces the action, below punishes it."""
        if abs(reward - 0.5) < 1e-6:
            return
        _, h = self.forward(x)
        mu = h @ self.W2 + self.b2
        dmu = np.sign(reward - 0.5) * (a - mu) / SIGMA**2
        dW2, db2 = np.outer(h, dmu), dmu
        dh = dmu @ self.W2.T * (1 - h**2)
        dW1, db1 = np.outer(x, dh), dh
        for g in (dW2, dW1, db2, db1):
            np.clip(g, -GRAD_CLIP, GRAD_CLIP, out=g)
        self.W2 += LEARN_RATE * dW2
        self.b2 += LEARN_RATE * db2
        self.W1 += LEARN_RATE * dW1
        self.b1 += LEARN_RATE * db1


def _worker(audio, secs):
    """Run one dance in a thread, updating STATE as it goes."""
    try:
        sim = Sim(audio, secs).load().build()
        STATE["bpm"] = sim.bpm
        sim.run(Policy(6 + len(sim.body_dofs)), render=True)
    except Exception as e:
        STATE.update(status="error", error=f"{type(e).__name__}: {e}")


def serve(audio, secs):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from ado_gui_html import HTML

    video = Path("demo_output/fly_dance/ado_blue.mp4")
    audio = Path(SRC)
    audio_mime = "audio/mpeg" if audio.suffix == ".mp3" else "audio/wav"

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            p = urlparse(self.path).path
            if p in ("/", "/index.html"):
                self._send(HTML.encode(), "text/html; charset=utf-8")
            elif p == "/state":
                self._send(json.dumps(STATE).encode(), "application/json")
            elif p == "/frame.jpg":
                fp = Path("/tmp/ado_frame.jpg")
                self._send(fp.read_bytes() if fp.exists() else b"",
                           "image/jpeg")
            elif p == "/audio":
                self._send(audio.read_bytes() if audio.exists() else b"",
                           audio_mime)
            elif p == "/video" and video.exists():
                self._send(video.read_bytes(), "video/mp4")
            else:
                self.send_error(404)

        def do_POST(self):
            if urlparse(self.path).path == "/start" and STATE["status"] != "dancing":
                STATE.update(status="starting", beats=0, t=0.0, sync=0.0)
                threading.Thread(target=_worker, args=(audio, secs), daemon=True).start()
            self.send_response(200)
            self.end_headers()

    srv = HTTPServer(("127.0.0.1", PORT), H)
    url = f"http://localhost:{PORT}"
    print(f"Ado Blue GUI -> {url}   (ctrl-c to stop)")
    webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    serve(SRC, float(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--s=")), 30.0)))
