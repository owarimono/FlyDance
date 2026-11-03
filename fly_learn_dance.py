"""Condition the fly to dance, with dopamine.

Three-factor learning rule, the same shape as real dopaminergic plasticity:
    dW = lr * dopamine * eligibility
  - eligibility  = recent correlation between beat and motion
  - dopamine     = reward, broadcast globally (like VUMa neurons)

No hand-authored choreography. One scalar weight `W` scales how much the
beat drives the legs. The fly starts ignorant, gets rewarded for moving its
body, and learns to amplify the music into movement.

Usage:
    python fly_learn_dance.py Monstruo-Ado.mp3
"""

import json
from pathlib import Path

import numpy as np

from flygym import Simulation
from flygym.compose import TetheredWorld
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import (
    CPGController,
    LocomotionAction,
    PreprogrammedSteps,
    apply_locomotion_action,
    make_locomotion_fly,
    make_tripod_cpg_network,
)

OUT_DIR = Path("demo_output/fly_dance")
CAM_OFFSET = (-0.5, -7.5, 0.0)
CAM_EULER = (1.57, 0.0, 0.0)
FOVY = 30.0

TRAIN_S = 30.0
# calibration knobs: hardware/biological params need tuning, not guessing
LR = 0.004         # plasticity step size
REWARD_SCALE = 400.0   # summed |joint velocity| (rad/s) -> ~1.0 dopamine
# eligibility must outlive the inter-beat gap (0.46s at 129 BPM), else the
# synapse is only plastic on the exact beat step and barely learns
TAU = 0.60
W_CLIP = (0.0, 6.0)


class DopamineController:
    """Three-factor rule: dW = lr * dopamine * eligibility.

    `W` is the single learned parameter -- how strongly the beat drives the
    legs. Eligibility is an exponential trace of beat x motion, so the weight
    only grows where moving on the beat was actually rewarded.
    """

    def __init__(self, lr=LR, tau=TAU):
        self.W = 0.05                       # start nearly ignorant
        self.lr = lr
        self.tau = max(tau, 1e-9)           # guard divide-by-zero
        self.elig = 0.0
        self.dopamine = 0.0
        self.beat_strength = 0.0
        self.history = []

    def on_beat(self, strength):
        """Beat lands: refresh the eligibility trace and tag the beat loudness."""
        self.elig = 1.0
        self.beat_strength = strength

    def update(self, dt, activity, beat_now):
        """One sim step. `activity` is any 0..1+ motion signal.

        In a tethered world the thorax is pinned, so the reward comes from
        what still moves: the leg joints.

        Dopamine is a *rate* -- reward accumulates over the whole inter-beat
        interval, not on the single step the beat lands on. A beat every
        ~0.4s is ~4000 sim steps apart, so a one-step impulse would be
        invisible in the weight.
        """
        if beat_now:
            self.elig = 1.0
        else:
            self.elig *= np.exp(-dt / self.tau)

        rate = min(activity / REWARD_SCALE, 1.0)
        # a louder beat is a stronger cue, so it teaches harder
        self.dopamine = rate * (0.5 + self.beat_strength)
        self.W += self.lr * self.dopamine * self.elig * dt
        self.W = float(np.clip(self.W, *W_CLIP))
        self.history.append((self.W, self.elig, self.dopamine))
        return self.W


def main(audio_path):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    beats_path = Path(str(audio_path) + ".beats.json")
    if not beats_path.exists():
        import beat_analysis
        beats = beat_analysis.analyze(str(audio_path))
        beats_path.write_text(json.dumps(beats))
    else:
        beats = json.loads(beats_path.read_text())

    fly = make_locomotion_fly(name="dancer", add_adhesion=True, colorize=True)
    cam = fly.add_tracking_camera(
        name="body_cam", pos_offset=CAM_OFFSET,
        rotation=Rotation3D("euler", CAM_EULER), fovy=FOVY,
    )
    # tethered: body pinned in place, legs free to dance
    world = TetheredWorld()
    world.add_fly(fly, [0.0, 0.0, 0.5], Rotation3D("quat", [1.0, 0.0, 0.0, 0.0]))

    sim = Simulation(world)
    renderer = sim.set_renderer([cam])

    steps = PreprogrammedSteps()
    dof_order = fly.get_actuated_jointdofs_order("position")
    cpg = make_tripod_cpg_network(timestep=sim.timestep, seed=0)
    controller = CPGController(cpg_network=cpg, preprogrammed_steps=steps,
                               output_dof_order=dof_order)
    brain = DopamineController()

    beat_times = np.asarray(beats["beat_times"], dtype=float)
    tempo = float(beats["tempo_bpm"])
    hz = float(np.clip(tempo / 60.0, 2.0, 40.0))
    env = np.asarray(beats["onset_env"], dtype=float)
    env = (env - env.min()) / (np.ptp(env) + 1e-9)
    hop, sr = int(beats["onset_hop"]), int(beats["sr"])

    print(f"tempo {tempo:.1f} BPM | {len(beat_times)} beats | gait {hz:.2f} Hz")

    sim.reset()
    apply_locomotion_action(sim, fly.name, LocomotionAction(
        joint_angles=steps.default_pose_by_dof_order(dof_order),
        adhesion_onoff=np.ones(6, dtype=bool)))
    sim.warmup()

    n = int(TRAIN_S / sim.timestep)
    times = np.arange(n) * sim.timestep

    W_log = np.zeros(n); spd_log = np.zeros(n); beat_log = np.zeros(n, bool)
    prev_q = None
    bi = 0

    for i in range(n):
        t = times[i]
        on_beat = bi < len(beat_times) and t >= beat_times[bi]
        if on_beat:
            k = min(int(t * sr / hop), env.size - 1)
            brain.on_beat(float(env[k]))
            bi += 1

        # W scales how much the music drives the legs
        cpg.intrinsic_amps[:] = 1.0 + brain.W
        cpg.intrinsic_freqs[:] = hz

        apply_locomotion_action(sim, fly.name, controller.step())
        sim.step_with_profile()

        # thorax is pinned in a tethered world: reward leg motion instead
        q = sim.get_joint_angles(fly.name)
        if prev_q is None:
            activity = 0.0
        else:
            activity = float(np.abs(q - prev_q).sum() / sim.timestep)
        prev_q = q.copy()

        brain.update(sim.timestep, activity, on_beat)
        W_log[i] = brain.W; spd_log[i] = activity; beat_log[i] = on_beat
        if i % 6000 == 0:
            sim.render_as_needed_with_profile()

    sim.renderer.save_video(OUT_DIR / "fly_learn_dance.mp4")
    _plot(times, W_log, spd_log, beat_log, brain)
    print(f"\nW: {W_log[0]:.3f} -> {W_log[-1]:.3f}  (clip {W_CLIP})")
    print(f"mean speed: first 5s {spd_log[:50000].mean():.1f} | last 5s {spd_log[-50000:].mean():.1f} mm/s")
    print("video:", (OUT_DIR / "fly_learn_dance.mp4").resolve())


def _plot(times, W, spd, beats, brain):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), tight_layout=True, sharex=True)
    ax[0].plot(times, W, color="tab:blue"); ax[0].set_ylabel("W"); ax[0].set_title("Learned gain")
    ax[1].plot(times, spd, color="tab:orange", lw=0.5); ax[1].set_ylabel("speed\n(mm/s)"); ax[1].set_title("Thorax speed")
    ax[2].plot(times, np.asarray([h[1] for h in brain.history]), color="tab:green", lw=0.5)
    ax[2].set_ylabel("eligibility"); ax[2].set_xlabel("Time (s)"); ax[2].set_title("Eligibility trace")
    for a in ax:
        for t in times[beats]:
            a.axvline(t, color="red", alpha=0.1, lw=0.4)
    p = OUT_DIR / "fly_learn_dance_progress.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print("progress:", p.resolve())


def _selfcheck():
    """Dopamine must increase W only when rewarded on a beat."""
    b = DopamineController()
    start = b.W
    dt = 0.001
    for _ in range(100):
        b.update(dt, 500.0, beat_now=True)      # moving on beat
    assert b.W > start, "W must grow when rewarded on-beat"

    held = b.W
    # must outlive several decay constants, whatever tau is
    decay_steps = int(10 * TAU / dt)
    for _ in range(decay_steps):
        b.update(dt, 0.0, beat_now=False)
    assert abs(b.W - held) < 1e-9, "W must not change without dopamine"
    assert b.elig < 0.01, f"eligibility must decay, got {b.elig}"
    print(f"selfcheck OK: W {start} -> {b.W:.4f}, elig -> {b.elig:.2e}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--selfcheck":
        _selfcheck()
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "Monstruo-Ado.mp3")
