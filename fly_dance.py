"""FlyGym — dance a MuJoCo fly to music.

Beat analysis (beat_analysis.py) produces a time grid offline; this script
maps that grid onto a CPG so the *gait* stays valid and only its timing
varies. Phase modulation beats driving joint angles from amplitude — the
body still does the physics.

Usage:
    python beat_analysis.py song.mp3
    python fly_dance.py song.mp3.beats.json
"""

import json
from pathlib import Path

import numpy as np

from flygym import Simulation
from flygym.anatomy import BodySegment
from flygym.compose import FlatGroundWorld
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import (
    CPGController,
    LocomotionAction,
    PreprogrammedSteps,
    apply_locomotion_action,
    make_locomotion_fly,
    make_tripod_cpg_network,
)

RUN_TIME_S = 6.0
OUTPUT_DIR = Path("demo_output/fly_dance")
VIDEO_NAME = "fly_dance.mp4"
DIAG_NAME = "fly_dance_diagnostics.png"

CAMERA_OFFSET = (-0.5, -7.5, 0.0)
CAMERA_EULER = (1.57, 0.0, 0.0)
CAMERA_FOVY = 30.0
SPAWN_POSITION = [0.0, 0.0, 0.5]
SPAWN_ROTATION_QUAT = [1.0, 0.0, 0.0, 0.0]

# Dance feel
STEP_MULTIPLIER = 1.0   # gait frequency as a multiple of the beat
AMPLITUDE_GAIN = 1.6    # how much louder -> bigger steps
MIN_STEP_HZ = 2.0
MAX_STEP_HZ = 40.0


def _print_header(title):
    line = "=" * 72
    print(f"\n{line}\n{title}\n{line}")


def _print_kv(key, value):
    print(f"  {key:<38}: {value}")


class DanceDriver:
    """Turn a beat grid into per-step CPG frequencies and amplitudes.

    Frequency tracks the beat; amplitude tracks loudness. Phases keep
    integrating so the gait stays continuous across tempo changes.
    """

    def __init__(self, beats, timestep):
        self.timestep = timestep
        self.beat_times = np.asarray(beats["beat_times"], dtype=float)
        self.tempo_bpm = float(beats["tempo_bpm"])
        env = np.asarray(beats["onset_env"], dtype=float)
        if env.size:
            lo, hi = env.min(), env.max()
            self.env = (env - lo) / (hi - lo + 1e-9)
        else:
            self.env = np.zeros(1, dtype=float)
        self.hop = int(beats.get("onset_hop", 512))
        self.sr = int(beats.get("sr", 22050))
        self._next_beat = 0
        self.beat_pulses = []      # (time, strength) while stepping

    def _env_at(self, t):
        i = int(t * self.sr / self.hop)
        if i < 0:
            return float(self.env[0])
        if i >= self.env.size:
            return float(self.env[-1])
        return float(self.env[i])

    def step(self, t, cpg):
        """Retune the CPG for time t; returns True on a beat boundary."""
        on_beat = False
        if self._next_beat < self.beat_times.size and t >= self.beat_times[self._next_beat]:
            on_beat = True
            self._next_beat += 1

        # one gait cycle per beat, clamped to something a leg can do
        hz = (self.tempo_bpm / 60.0) * STEP_MULTIPLIER
        hz = float(np.clip(hz, MIN_STEP_HZ, MAX_STEP_HZ))
        cpg.intrinsic_freqs[:] = hz

        # loudness -> step size; floor so the legs never freeze mid-air
        loud = self._env_at(t)
        mag = 1.0 + AMPLITUDE_GAIN * loud
        cpg.intrinsic_amps[:] = mag

        if on_beat:
            self.beat_pulses.append((t, loud))
        return on_beat


def main(beats_path):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    beats = json.loads(Path(beats_path).read_text())

    _print_header("1. Building fly + tracking camera")
    fly = make_locomotion_fly(name="dancer", add_adhesion=True, colorize=True)
    body_cam = fly.add_tracking_camera(
        name="body_cam",
        pos_offset=CAMERA_OFFSET,
        rotation=Rotation3D("euler", CAMERA_EULER),
        fovy=CAMERA_FOVY,
    )

    _print_header("2. Building flat-ground world")
    world = FlatGroundWorld()
    world.add_fly(fly, list(SPAWN_POSITION),
                  Rotation3D("quat", SPAWN_ROTATION_QUAT))

    _print_header("3. Creating simulation + renderer")
    sim = Simulation(world)
    renderer = sim.set_renderer([body_cam])

    _print_header("4. Wiring CPG to the beat grid")
    pre_steps = PreprogrammedSteps()
    dof_order = fly.get_actuated_jointdofs_order("position")
    cpg = make_tripod_cpg_network(timestep=sim.timestep, seed=0)
    controller = CPGController(
        cpg_network=cpg, preprogrammed_steps=pre_steps, output_dof_order=dof_order
    )
    driver = DanceDriver(beats, sim.timestep)

    _print_kv("tempo (BPM)", driver.tempo_bpm)
    _print_kv("beats detected", len(driver.beat_times))
    _print_kv("actuated DoFs", len(dof_order))
    _print_kv("timestep (s)", sim.timestep)
    _print_kv("gait frequency (Hz)", driver.tempo_bpm / 60.0)

    _print_header("5. Resetting and warming up")
    sim.reset()
    apply_locomotion_action(
        sim, fly.name,
        LocomotionAction(
            joint_angles=pre_steps.default_pose_by_dof_order(dof_order),
            adhesion_onoff=np.ones(6, dtype=bool),
        ),
    )
    sim.warmup()

    _print_header(f"6. Dancing for {RUN_TIME_S}s")
    nsteps = int(RUN_TIME_S / sim.timestep)
    times = np.arange(nsteps) * sim.timestep

    thorax_idx = fly.get_bodysegs_order().index(BodySegment("c_thorax"))
    thorax = np.full((nsteps, 3), np.nan, dtype=np.float32)
    phases = np.full((nsteps, 6), np.nan, dtype=np.float32)
    mags = np.full(nsteps, np.nan, dtype=np.float32)
    adhesion = np.zeros((nsteps, 6), dtype=bool)
    step_hz_log = np.full(nsteps, np.nan, dtype=np.float32)
    beat_marks = np.zeros(nsteps, dtype=bool)

    for i in range(nsteps):
        t = times[i]
        if driver.step(t, cpg):
            beat_marks[i] = True
        action = controller.step()
        apply_locomotion_action(sim, fly.name, action)

        thorax[i] = sim.get_body_positions(fly.name)[thorax_idx]
        phases[i] = cpg.curr_phases % (2 * np.pi)
        mags[i] = cpg.curr_magnitudes[0]
        adhesion[i] = action.adhesion_onoff
        step_hz_log[i] = cpg.intrinsic_freqs[0]

        sim.step_with_profile()
        sim.render_as_needed_with_profile()

    _print_header("7. Summary")
    n_frames = len(renderer.frames[body_cam.name])
    disp = float(thorax[-1, 0] - thorax[0, 0])
    _print_kv("rendered frames", n_frames)
    _print_kv("forward displacement (mm)", f"{disp:.2f}")
    _print_kv("beats hit during sim", int(beat_marks.sum()))
    _print_kv("beats in clip", len(driver.beat_times))
    sim.print_performance_report()

    _print_header("8. Saving outputs")
    video = OUTPUT_DIR / VIDEO_NAME
    sim.renderer.save_video(video)
    _print_kv("video", video.resolve())
    _save_diagnostics(times, thorax, phases, adhesion, step_hz_log, beat_marks, pre_steps)
    print("\nDone.")


def _save_diagnostics(times, thorax, phases, adhesion, step_hz, beat_marks, pre_steps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(9, 9), tight_layout=True, sharex=True)

    axes[0].plot(times, thorax[:, 0] - thorax[0, 0], color="k")
    axes[0].set_ylabel("Forward\npos (mm)")
    axes[0].set_title("Thorax displacement")

    axes[1].plot(times, step_hz, color="tab:green")
    axes[1].set_ylabel("Gait\nfreq (Hz)")
    axes[1].set_title("CPG frequency locked to BPM")

    axes[2].plot(times, phases, lw=0.7)
    axes[2].set_ylabel("Phase\n(rad)")
    axes[2].set_yticks(np.arange(7) * (np.pi / 3))
    axes[2].set_title("Leg CPG phases (tripod: 3 phases repeating)")

    axes[3].imshow(adhesion.T, aspect="auto", interpolation="nearest",
                  extent=(times[0], times[-1], 5.5, -0.5), cmap="gray")
    axes[3].set_yticks(np.arange(6))
    axes[3].set_yticklabels([leg.upper() for leg in pre_steps.legs])
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Leg")
    axes[3].set_title("Adhesion state")

    for ax in axes[:3]:
        for t in times[beat_marks]:
            ax.axvline(t, color="red", alpha=0.15, lw=0.5)

    path = OUTPUT_DIR / DIAG_NAME
    fig.savefig(path, dpi=150)
    plt.close(fig)
    _print_kv("diagnostics", path.resolve())


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("usage: python fly_dance.py <song.mp3.beats.json>")
    main(sys.argv[1])
