"""J-Pop dance: discrete pose pops on the beat, not a walking cycle.

PreprogrammedSteps is a cyclic spline -- that is walking. Dancing needs
discrete keyframe poses snapped to the beat, so the legs read as *poses*
instead of a shuffle. Same spline, different phase input: jump `phase`
around instead of advancing it.

    python fly_pop.py Monstruo-Ado.mp3 --pose=choreo
    python fly_pop.py Monstruo-Ado.mp3 --pose=random --live
"""

import json
import sys
from pathlib import Path

import numpy as np
import mujoco.viewer

from flygym import Simulation
from flygym.compose import TetheredWorld
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import (
    LocomotionAction, PreprogrammedSteps, apply_locomotion_action, make_locomotion_fly,
)

OUT = Path("demo_output/fly_dance")
SRC = next((a for a in sys.argv[1:] if not a.startswith("--")), "Monstruo-Ado.mp3")
LIVE = "--live" in sys.argv
SECS = float(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--s=")), 20.0))

RAMP = 0.09        # s to travel to a new pose
LOOKAHEAD = 0.09   # start the move this early so it *lands* on the beat
POP = 1.25         # how far from neutral a leg can swing
SIGMA = 0.6       # policy exploration noise (radians)
LEARN_RATE = 0.004
REWARD_SCALE = 400.0   # summed |joint velocity| -> ~1.0 reward
BEAT_WIN = 0.06         # s around a beat that counts as "on the beat"
GRAD_CLIP = 0.02       # ponytail: per-beat updates divide by k=1, so the raw
                        # score-function grad is huge; clip tight and lower
                        # SIGMA / use a proper baseline if this proves unstable


class Policy:
    """Tiny numpy MLP: (beat_pos, loudness) -> 6 leg phases, trained by REINFORCE.

    No hand-written choreography: the fly picks its own poses and only keeps
    the ones that made it move on the beat. No torch on purpose -- this is
    three matmuls and a manual backprop.
    """

    def __init__(self, n_in=2, n_out=6, hidden=16, seed=0):
        rng = np.random.default_rng(seed)
        self.W1, self.b1 = rng.normal(0, .3, (n_in, hidden)), np.zeros(hidden)
        self.W2, self.b2 = rng.normal(0, .3, (hidden, n_out)), np.zeros(n_out)

    def forward(self, x):
        h = np.tanh(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2, h

    def act(self, x):
        mu, _ = self.forward(x)
        a = mu + np.random.normal(0, SIGMA, mu.shape)
        # Gaussian log-prob of the sampled action, summed over the 6 legs
        z = (a - mu) / SIGMA
        logp = float(-0.5 * (z * z).sum() - a.size * np.log(SIGMA * np.sqrt(2 * np.pi)))
        return a % (2 * np.pi), logp

    def reinforce(self, batch, rewards):
        """batch = [(x, action, logp), ...], rewards = [float, ...] in 0..1.

        Score-function estimator: d/dmu log p(a) = (a - mu) / sigma^2.
        """
        if len(batch) < 1:
            return 0.0
        adv = rewards.mean() - 0.5          # centre on "half-hearted"
        if abs(adv) < 1e-6:
            return 0.0
        scale = 1.0 if adv > 0 else -1.0
        dW2 = dW1 = db2 = db1 = 0.0
        for (x, a, _) in batch:
            _, h = self.forward(x)
            mu = h @ self.W2 + self.b2
            dmu = scale * (a - mu) / SIGMA**2
            dW2 += np.outer(h, dmu)
            db2 += dmu
            dh = dmu @ self.W2.T * (1 - h**2)
            dW1 += np.outer(x, dh)
            db1 += dh
        k = len(batch)
        dW2, db2, dW1, db1 = (LEARN_RATE * d / k for d in (dW2, db2, dW1, db1))
        # clip AFTER scaling: score-function grads are (a-mu)/sigma^2 and blow
        # up the weights within a few updates otherwise
        for g in (dW2, db2, dW1, db1):
            np.clip(g, -GRAD_CLIP, GRAD_CLIP, out=g)
        self.W2 += dW2
        self.b2 += db2
        self.W1 += dW1
        self.b1 += db1
        return float(np.abs(self.W2).sum())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bp = Path(str(SRC) + ".beats.json")
    if not bp.exists():
        import beat_analysis
        bp.write_text(json.dumps(beat_analysis.analyze(SRC)))
    beats = json.loads(bp.read_text())
    bt = np.asarray(beats["beat_times"], float)
    env = np.asarray(beats["onset_env"], float)
    env = (env - env.min()) / (np.ptp(env) + 1e-9)
    hop, sr = int(beats["onset_hop"]), int(beats["sr"])
    print(f"{beats['tempo_bpm']:.1f} BPM | {len(bt)} beats"
          f" | {'LIVE' if LIVE else 'render'}")

    fly = make_locomotion_fly(name="dancer", add_adhesion=True, colorize=True)
    world = TetheredWorld()
    cam = fly.add_tracking_camera(
        name="body_cam", pos_offset=(-0.5, -7.5, 0.0),
        rotation=Rotation3D("euler", (1.57, 0.0, 0.0)), fovy=30.0)
    world.add_fly(fly, [0.0, 0.0, 0.5], Rotation3D("quat", [1.0, 0.0, 0.0, 0.0]))
    sim = Simulation(world)
    sim.set_renderer([cam])

    steps = PreprogrammedSteps()
    dof_order = fly.get_actuated_jointdofs_order("position")
    sim.reset()
    apply_locomotion_action(sim, fly.name, LocomotionAction(
        joint_angles=steps.default_pose_by_dof_order(dof_order),
        adhesion_onoff=np.ones(6, dtype=bool)))
    sim.warmup()

    n = int(SECS / sim.timestep)
    viewer = mujoco.viewer.launch_passive(sim.mj_model, sim.mj_data) if LIVE else None
    policy = Policy()

    target = np.full(6, np.pi)
    goal = np.full(6, np.pi)
    bi = 0
    batch, beat_r, speeds = [], {}, []
    prev_q = None
    beat_i = -1                 # index of the beat the pose is aimed at

    for i in range(n):
        t = i * sim.timestep
        # fire LOOKAHEAD early so the pose ARRIVES on the beat
        if bi < len(bt) and t >= bt[bi] - LOOKAHEAD:
            loud = float(env[min(int(bt[bi] * sr / hop), env.size - 1)])
            x = np.array([bi % 4, loud])          # beat position + loudness
            a, logp = policy.act(x)
            batch.append((x, a, logp))
            goal = a
            beat_i = bi
            bi += 1

        d = (goal - target + np.pi) % (2 * np.pi) - np.pi
        target = (target + d * min(1.0, sim.timestep / RAMP)) % (2 * np.pi)

        angles = steps.get_joint_angles_by_dof_order(target, np.full(6, POP), dof_order)
        adh = steps.get_adhesion_onoff_by_phase(target)
        apply_locomotion_action(sim, fly.name, LocomotionAction(angles, adh))
        sim.step()

        q = sim.get_joint_angles(fly.name)
        sp = 0.0 if prev_q is None else float(np.abs(q - prev_q).sum() / sim.timestep)
        prev_q = q.copy()
        speeds.append(sp)

        # Reward = movement AT the beat, not on average. Averaging over the
        # inter-beat gap rewards constant flailing, which is not dancing.
        if batch and beat_i < len(bt):
            if abs(t - bt[beat_i]) < BEAT_WIN:
                beat_r[beat_i] = min(sp / REWARD_SCALE, 1.0)

        # one update per beat, using that beat's own reward
        if batch and beat_i in beat_r and t > bt[beat_i] + BEAT_WIN:
            policy.reinforce([batch[-1]], np.array([beat_r.pop(beat_i)]))
            batch.pop()

        if viewer is not None and i % 100 == 0:
            viewer.sync()
        if i % 500 == 0:
            sim.render_as_needed_with_profile()

    if viewer is not None:
        viewer.close()
    sim.renderer.save_video(OUT / "fly_pop_rl.mp4")
    s = np.array(speeds)
    print(f"beats: {bi} | |W2|: {np.abs(policy.W2).sum():.3f}")
    print(f"speed first 20%: {s[:len(s)//5].mean():.0f} | last 20%: {s[-len(s)//5:].mean():.0f} rad/s")
    # is it actually on the beat? compare speed inside the window vs outside
    ts = np.arange(len(s)) * sim.timestep
    inwin = np.zeros(len(s), bool)
    for b in bt:
        inwin |= np.abs(ts - b) < BEAT_WIN
    print(f"sync: on-beat {s[inwin].mean():.0f} | off-beat {s[~inwin].mean():.0f} rad/s"
          f" | ratio {s[inwin].mean() / max(s[~inwin].mean(), 1e-6):.2f}x")
    print(f"video: {(OUT / 'fly_pop_rl.mp4').resolve()}")


if __name__ == "__main__":
    main()
