"""Tethered fly, learning to dance, with a live MuJoCo window.

Replaces the single learned scalar with a small linear brain: a 3x3 weight
matrix mapping recent (beat, loudness, motion) to leg drive. Learned by the
same three-factor rule as before, so nothing about the training loop changes.

    python fly_dance_live.py Monstruo-Ado.mp3
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
    CPGController, LocomotionAction, PreprogrammedSteps,
    apply_locomotion_action, make_locomotion_fly, make_tripod_cpg_network,
)

OUT = Path("demo_output/fly_dance")
LIVE = "--headless" not in sys.argv
SRC = next((a for a in sys.argv[1:] if not a.startswith("--")), "Monstruo-Ado.mp3")

TRAIN_S = float(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--s=")), 30.0))
LR = 0.004
REWARD_SCALE = 400.0
TAU = 0.6
W_CLIP = 0.5


class LinearBrain:
    """3x3 weights: [beat, loudness, motion] -> leg drive. Three-factor rule.

    Eligibility is per-synapse: each weight only learns when *its own* input
    was active recently. That is the difference from a single scalar -- the
    beat-synaptic weight can grow without the loudness weight doing so.
    """

    def __init__(self, n_in=3, n_out=3, lr=LR, tau=TAU, rng=None):
        rng = rng or np.random.default_rng(0)
        self.W = rng.normal(0, 0.02, (n_out, n_in))
        self.elig = np.zeros(n_in)          # EMA of each input
        self.lr = lr
        self.tau = tau

    def drive(self, x):
        """x = [beat_pulse, loudness, motion]. Returns per-leg drive."""
        return np.tanh(self.W @ x)

    def update(self, dt, x, reward):
        """reward in 0..1. Eligibility = EMA of input activity, shared across rows.

        Weight (i, j) strengthens only while input j is active (dopamine gates the
        whole step). Keeping elig an EMA rather than an integral is what stops
        every weight saturating at W_CLIP within seconds.
        """
        a = dt / self.tau
        self.elig = (1 - a) * self.elig + a * x
        dW = self.lr * reward * self.elig * dt
        self.W = np.clip(self.W + dW, -W_CLIP, W_CLIP)
        return float(np.abs(dW).sum())

    def novelty(self):
        return float(np.abs(self.W).sum())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    beats_path = Path(str(SRC) + ".beats.json")
    if not beats_path.exists():
        import beat_analysis
        beats_path.write_text(json.dumps(beat_analysis.analyze(SRC)))
    beats = json.loads(beats_path.read_text())

    fly = make_locomotion_fly(name="dancer", add_adhesion=True, colorize=True)
    world = TetheredWorld()
    # camera goes on the fly spec BEFORE Simulation compiles the world
    cam = fly.add_tracking_camera(
        name="body_cam", pos_offset=(-0.5, -7.5, 0.0),
        rotation=Rotation3D("euler", (1.57, 0.0, 0.0)), fovy=30.0)
    world.add_fly(fly, [0.0, 0.0, 0.5], Rotation3D("quat", [1.0, 0.0, 0.0, 0.0]))
    sim = Simulation(world)
    renderer = sim.set_renderer([cam])

    steps = PreprogrammedSteps()
    dof_order = fly.get_actuated_jointdofs_order("position")
    cpg = make_tripod_cpg_network(timestep=sim.timestep, seed=0)
    ctl = CPGController(cpg_network=cpg, preprogrammed_steps=steps,
                         output_dof_order=dof_order)
    brain = LinearBrain()

    bt = np.asarray(beats["beat_times"], float)
    env = np.asarray(beats["onset_env"], float)
    env = (env - env.min()) / (np.ptp(env) + 1e-9)
    hop, sr = int(beats["onset_hop"]), int(beats["sr"])
    hz = float(np.clip(float(beats["tempo_bpm"]) / 60.0, 2.0, 40.0))
    print(f"tempo {beats['tempo_bpm']:.1f} BPM | {len(bt)} beats | gait {hz:.2f} Hz"
          f" | {'LIVE' if LIVE else 'headless'}")

    sim.reset()
    apply_locomotion_action(sim, fly.name, LocomotionAction(
        joint_angles=steps.default_pose_by_dof_order(dof_order),
        adhesion_onoff=np.ones(6, dtype=bool)))
    sim.warmup()

    n = int(TRAIN_S / sim.timestep)
    prev_q = None
    bi = 0
    W0 = brain.novelty()
    viewer = mujoco.viewer.launch_passive(sim.mj_model, sim.mj_data) if LIVE else None

    for i in range(n):
        t = i * sim.timestep
        on_beat = bi < len(bt) and t >= bt[bi]
        pulse = 0.0
        if on_beat:
            k = min(int(t * sr / hop), env.size - 1)
            pulse = float(env[k]); bi += 1

        apply_locomotion_action(sim, fly.name, ctl.step())
        sim.step()

        q = sim.get_joint_angles(fly.name)
        motion = 0.0 if prev_q is None else float(np.abs(q - prev_q).sum() / sim.timestep)
        prev_q = q.copy()

        x = np.array([pulse, float(env[min(int(t * sr / hop), env.size - 1)]),
                      min(motion / REWARD_SCALE, 1.0)])
        drive = brain.drive(x)                       # network output
        cpg.intrinsic_amps[:] = 1.0 + np.abs(drive).mean()
        cpg.intrinsic_freqs[:] = hz
        brain.update(sim.timestep, x, min(motion / REWARD_SCALE, 1.0))

        if viewer is not None and i % 200 == 0:
            viewer.sync()
        if i % 2000 == 0:
            sim.render_as_needed_with_profile()

    if viewer is not None:
        viewer.close()
    print(f"\n|W|: {W0:.4f} -> {brain.novelty():.4f}")
    print(np.round(brain.W, 3))
    sim.renderer.save_video(OUT / "fly_dance_live.mp4")


if __name__ == "__main__":
    main()
