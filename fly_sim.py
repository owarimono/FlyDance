"""
FlyGym 2.1.0 — CPG-driven walking simulation.

Based on the official tutorial:
    flygym/tutorials/4a_cpg_controller.ipynb

This script:
    1. Builds a hexapod fly with adhesion-enabled legs.
    2. Places it on a flat ground world.
    3. Drives it with a tripod Central Pattern Generator (CPG).
    4. Renders a tracking-camera video and saves diagnostics.

Outputs (under ./demo_output/cpg_controller/):
    - cpg_controller.mp4             : rendered walking video
    - cpg_controller_diagnostics.png : thorax displacement + adhesion plot
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

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


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

RUN_TIME_S = 2.0
OUTPUT_DIR = Path("demo_output/cpg_controller")
VIDEO_NAME = "cpg_controller.mp4"
DIAGNOSTICS_NAME = "cpg_controller_diagnostics.png"

# Camera offset relative to the tracked body segment (mm)
CAMERA_OFFSET = (-0.5, -7.5, 0.0)
CAMERA_EULER = (1.57, 0.0, 0.0)  # pitch 90° -> looking forward
CAMERA_FOVY = 30.0

# Spawn pose for the fly
SPAWN_POSITION = [0.0, 0.0, 0.5]          # mm
SPAWN_ROTATION_QUAT = [1.0, 0.0, 0.0, 0.0]  # identity quaternion (w, x, y, z)

# CPG network parameters (see tutorial 4a for details)
CPG_INTRINSIC_AMPLITUDE = 1.0
CPG_COUPLING_STRENGTH = 10.0
CPG_CONVERGENCE_COEF = 20.0
CPG_SEED = 0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _print_header(title: str) -> None:
    line = "=" * 72
    print(f"\n{line}\n{title}\n{line}")


def _print_kv(key: str, value) -> None:
    print(f"  {key:<38}: {value}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. Build the fly and tracking camera ---------------------------- #
    _print_header("1. Building fly + tracking camera")

    fly = make_locomotion_fly(
        name="cpg_demo",
        add_adhesion=True,
        colorize=True,
    )

    body_cam = fly.add_tracking_camera(
        name="body_cam",
        pos_offset=CAMERA_OFFSET,
        rotation=Rotation3D("euler", CAMERA_EULER),
        fovy=CAMERA_FOVY,
    )

    # ---- 2. Build the world ---------------------------------------------- #
    _print_header("2. Building flat-ground world")

    world = FlatGroundWorld()

    spawn_position = list(SPAWN_POSITION)
    spawn_rotation = Rotation3D("quat", SPAWN_ROTATION_QUAT)
    world.add_fly(fly, spawn_position, spawn_rotation)

    _print_kv("spawn position (mm)", spawn_position)

    # ---- 3. Simulation + renderer ---------------------------------------- #
    _print_header("3. Creating simulation + renderer")

    sim = Simulation(world)
    renderer = sim.set_renderer([body_cam])

    # ---- 4. CPG controller ----------------------------------------------- #
    _print_header("4. Setting up CPG controller")

    preprogrammed_steps = PreprogrammedSteps()
    dof_order = fly.get_actuated_jointdofs_order("position")

    cpg_network = make_tripod_cpg_network(
        timestep=sim.timestep,
        intrinsic_amplitude=CPG_INTRINSIC_AMPLITUDE,
        coupling_strength=CPG_COUPLING_STRENGTH,
        convergence_coef=CPG_CONVERGENCE_COEF,
        seed=CPG_SEED,
    )

    controller = CPGController(
        cpg_network=cpg_network,
        preprogrammed_steps=preprogrammed_steps,
        output_dof_order=dof_order,
    )

    _print_kv("actuated DoFs", len(dof_order))
    _print_kv("simulation timestep (s)", sim.timestep)
    _print_kv(
        "CPG intrinsic frequency (Hz)",
        float(cpg_network.intrinsic_freqs[0]),
    )
    _print_kv(
        "kinematic spline cycle rate (Hz)",
        round(preprogrammed_steps.step_cycle_frequency_hz, 4),
    )

    # ---- 5. Reset and warm-up -------------------------------------------- #
    _print_header("5. Resetting and warming up")

    sim.reset()

    initial_action = LocomotionAction(
        joint_angles=preprogrammed_steps.default_pose_by_dof_order(dof_order),
        adhesion_onoff=np.ones(6, dtype=bool),
    )
    apply_locomotion_action(sim, fly.name, initial_action)
    sim.warmup()

    # ---- 6. Main loop ---------------------------------------------------- #
    _print_header(f"6. Running simulation for {RUN_TIME_S}s")

    nsteps_sim = int(RUN_TIME_S / sim.timestep)
    time_grid = np.arange(nsteps_sim) * sim.timestep

    thorax_idx = fly.get_bodysegs_order().index(BodySegment("c_thorax"))
    thorax_positions = np.full((nsteps_sim, 3), np.nan, dtype=np.float32)
    cpg_phases = np.full((nsteps_sim, 6), np.nan, dtype=np.float32)
    adhesion_onoff = np.zeros((nsteps_sim, 6), dtype=bool)

    for step_idx in trange(nsteps_sim, desc="Running CPG"):
        action = controller.step()
        apply_locomotion_action(sim, fly.name, action)
        sim.step_with_profile()

        thorax_positions[step_idx] = sim.get_body_positions(fly.name)[thorax_idx]
        cpg_phases[step_idx] = controller.cpg_network.curr_phases % (2 * np.pi)
        adhesion_onoff[step_idx] = action.adhesion_onoff

        sim.render_as_needed_with_profile()

    # ---- 7. Summary ------------------------------------------------------ #
    _print_header("7. Summary")

    n_frames = len(renderer.frames[body_cam.name])
    displacement = float(thorax_positions[-1, 0] - thorax_positions[0, 0])

    _print_kv("rendered frames (body_cam)", n_frames)
    _print_kv("forward displacement (mm)", f"{displacement:.2f}")

    sim.print_performance_report()

    # ---- 8. Save outputs ------------------------------------------------- #
    _print_header("8. Saving outputs")

    video_path = OUTPUT_DIR / VIDEO_NAME
    sim.renderer.save_video(video_path)
    _print_kv("video", video_path.resolve())

    # Diagnostics plot
    fig, axes = plt.subplots(
        2, 1, figsize=(8, 5), tight_layout=True, sharex=True
    )

    axes[0].plot(time_grid, thorax_positions[:, 0] - thorax_positions[0, 0])
    axes[0].set_ylabel("Forward\nposition (mm)")
    axes[0].set_title("Thorax displacement")

    axes[1].imshow(
        adhesion_onoff.T,
        aspect="auto",
        interpolation="nearest",
        extent=(time_grid[0], time_grid[-1], 5.5, -0.5),
        cmap="gray",
    )
    axes[1].set_yticks(np.arange(6))
    axes[1].set_yticklabels([leg.upper() for leg in preprogrammed_steps.legs])
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Leg")
    axes[1].set_title("Adhesion state")

    diagnostics_path = OUTPUT_DIR / DIAGNOSTICS_NAME
    fig.savefig(diagnostics_path, dpi=150)
    plt.close(fig)
    _print_kv("diagnostics", diagnostics_path.resolve())

    print("\nDone.")


if __name__ == "__main__":
    main()