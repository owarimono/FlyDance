
# FlyDance

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](environment.yml)
[![License](https://img.shields.io/badge/license-GPL--2.0-blue.svg)](LICENSE)
[![FlyGym](https://img.shields.io/badge/FlyGym-v2.1.0-orange.svg)](https://github.com/NeLy-EPFL/flygym)
[![FlyWire](https://img.shields.io/badge/FlyWire-783-lightgrey.svg)](https://flywire.ai/)

Audio-driven choreography for a connectome-scale *Drosophila* brain
simulation, built on FlyGym v2.1.0 and a FlyWire-derived whole-brain LIF model.

---

## Table of contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project structure](#project-structure)
- [Pipeline](#pipeline)
- [What is not in this repository](#what-is-not-in-this-repository)
- [License](#license)
- [Credits and acknowledgments](#credits-and-acknowledgments)

---

## Overview

FlyDance is a research and creative-coding project that couples two
independent systems:

1. **FlyGym v2.1.0** — a MuJoCo-based biomechanical simulation of the adult
   *Drosophila*, exposed as a Python API for neuro-mechanical experiments.
2. **A whole-brain LIF model** inherited from the
   [fly-brain](https://github.com/eonsystemspbc/fly-brain) project, with
   connectivity derived from the FlyWire connectome (dataset version 783,
   ~138k neurons).

Audio beat analysis produces a deterministic beat grid that is then mapped
onto central pattern generators (CPGs), learned low-dimensional policies, or
dopamine-gated plasticity rules. The resulting motor traces are rendered
through FlyGym to produce video. The repository additionally retains the
multi-backend neural benchmark harness from fly-brain.

FlyDance is not a production tool. It is an experimental platform intended
for research reproduction, teaching, and creative exploration.

---

## Architecture

Three subsystems communicate through files and simple in-process APIs:

```

audio ──► │  beat_analysis.py  │ ──► <track>.beats.json
└────────────────────┘
│
▼
┌────────────────────┐        ┌─────────────────────┐
│  fly_dance.py     │ ─────► │  FlyGym / MuJoCo    │
│  fly_pop.py        │        │  (biomechanical     │
│  fly_sim.py        │        │   simulation)       │
└────────────────────┘        └─────────────────────┘
│
▼
demo_output/**/.mp4

```

---

## Features

| Capability | Entry point |
|---|---|
| Beat, tempo, onset-envelope, and chroma extraction | `beat_analysis.py` |
| Beat-locked CPG gait modulation | `fly_dance.py` |
| Browser GUI with streamed MuJoCo frames | `fly_dance_gui.py`, `ado_gui_html.py` |
| MuJoCo viewer mode with a 3×3 linear brain | `fly_dance_live.py` |
| Dopamine-gated single-scalar plasticity | `fly_learn_dance.py` |
| REINFORCE-trained discrete pose policy | `fly_pop.py` |
| Standalone tripod-CPG walking baseline | `fly_sim.py` |
| Spike raster, PSTH, and top-neuron plots | `visualize_spikes.py` |
| SEZ neuron inventory and categorization | `categorize_sez.py`, `explore_sez.py` |
| Multi-backend neural benchmark runner | `main.py`, `code/` |

---

## Requirements

| Component | Version | Notes |
|---|---|---|
| Python | 3.10 | Pinned in `environment.yml` |
| Conda / Miniconda | any recent | Environment manager |
| FlyGym | v2.1.0 | Installed separately, not vendored |
| MuJoCo | bundled with FlyGym | Imported directly by GUI and viewer scripts |
| CUDA | 12.x | Optional; required only for GPU backends |
| librosa | any recent | Audio analysis; not pinned in `environment.yml` |
| soundfile | any recent | librosa I/O backend; not in `environment.yml` |

The Brian2GeNN backend requires a separate environment (`brian2genn==1.7.0`
pins `brian2<2.6`, which conflicts with the Brian2CUDA stack). That
environment is provided as `environment-brian2genn.yml`.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/owarimono/FlyDance.git
cd FlyDance

# 2. Create and activate the main environment
conda env create -f environment.yml
conda activate brain-fly

# 3. Install audio-analysis dependencies
pip install librosa soundfile

# 4. Install FlyGym v2.1.0 separately
git clone --branch v2.1.0 https://github.com/NeLy-EPFL/flygym.git
pip install -e flygym

# 5. (Optional) Brian2GeNN environment
conda env create -f environment-brian2genn.yml
conda activate brain-fly-brian2genn
```

---

Usage

Beat analysis

```bash
# Writes Monstruo-Ado.mp3.beats.json by default
python beat_analysis.py Monstruo-Ado.mp3

# Explicit output path
python beat_analysis.py song.mp3 song.beats.json
```

Input is sys.argv[1]; optional output JSON is sys.argv[2].

Beat-driven CPG dance

```bash
python fly_dance.py Monstruo-Ado.mp3.beats.json
```

Writes demo_output/fly_dance/fly_dance.mp4 and
demo_output/fly_dance/fly_dance_diagnostics.png.

Browser GUI

```bash
# Audio path defaults to Monstruo-Ado.mp3
python fly_dance_gui.py Monstruo-Ado.mp3

# Custom port and simulated duration
python fly_dance_gui.py Monstruo-Ado.mp3 --port=9000 --s=60
```

The script starts an HTTP server bound to localhost:8765 (configurable via
--port) and prints the listening URL to stdout. It does not launch a
browser automatically — open the printed URL manually. --s=30.0 sets the
default simulated duration in seconds. The GUI is served over stdlib
http.server; no Flask or WebSocket dependency.

MuJoCo viewer and plasticity

```bash
# Viewer window; add --headless to disable
python fly_dance_live.py Monstruo-Ado.mp3 --s=30

# Dopamine-gated single-scalar learning
python fly_learn_dance.py Monstruo-Ado.mp3
```

fly_dance_live.py writes demo_output/fly_dance/fly_dance_live.mp4.
fly_learn_dance.py writes an MP4 and a learning-curve plot. "Live" here
refers to the MuJoCo viewer window, not microphone input; both scripts read a
file path and its .beats.json.

REINFORCE pose pops

```bash
python fly_pop.py Monstruo-Ado.mp3 --s=20 --live
```

Flags: --live (viewer), --s=20.0 (default). Writes
demo_output/fly_dance/fly_pop_rl.mp4. The docstring mentions
--pose=choreo|random, but the current code does not parse that flag.

CPG walking baseline

```bash
python fly_sim.py
```

Writes demo_output/cpg_controller/cpg_controller.mp4 and
demo_output/cpg_controller/cpg_controller_diagnostics.png.

Spike visualization

```bash
python visualize_spikes.py data/results/round_01/brian2cpp_t1.0s_n1.parquet
```

Defaults to data/results/pytorch_t1.0s_n1.parquet when no argument is
given. Writes <input>_plot.png with a raster, PSTH, and top-20 neuron
bar chart.

SEZ inspection

```bash
python explore_sez.py
python categorize_sez.py
```

Both read data/sez_neurons.pickle and print summaries to stdout. No CLI
flags; no file outputs.

Neural benchmark harness

```bash
# Default suite
python main.py

# Paper grid: t_run=(0.1, 1, 10, 100), n_run=(1, 4, 8, 16, 32), 5 rounds
python main.py --paper --run-label my_run

# Single backend
python main.py --brian2-cpu --t_run 1 --n_run 1

# Timing-only run without spike I/O
python main.py --disable-spike-io
```

Results are appended to data/benchmark-results.csv; spike parquet files
go under data/results/. Run python main.py --help for the full flag list.

---

Project structure

Path Purpose
beat_analysis.py Beat, tempo, onset-envelope, and chroma extraction; writes .beats.json.
fly_dance.py Maps a beat grid onto a tripod CPG; renders video and diagnostics.
fly_dance_gui.py Stdlib HTTP server + browser GUI; simulation runs in a worker thread, streams JSON state and JPEG frames.
fly_dance_live.py MuJoCo viewer mode with a 3×3 linear brain trained by a three-factor rule.
fly_learn_dance.py Dopamine-gated plasticity experiment with a single learned beat-to-leg scalar.
fly_pop.py REINFORCE-trained discrete pose policy for beat-snapped "pops".
fly_sim.py Standalone CPG-driven walking simulation based on the FlyGym tutorial.
visualize_spikes.py Raster, PSTH, and top-neuron plots from spike parquet files.
categorize_sez.py Groups SEZ neuron types into descending, sensory, motor, and other.
explore_sez.py Prints the structure and first entries of data/sez_neurons.pickle.
ado_gui_html.py HTML/CSS/JS string for the GUI; imported by fly_dance_gui.py.
main.py CLI entry point for the multi-backend benchmark suite.
code/ Benchmark orchestrator (benchmark.py), backend runners, comparison scripts.
data/ FlyWire connectivity and completeness data, SEZ pickle, benchmark CSV, and results.
scripts/ WSL/CUDA setup script and NEST GPU source files.
environment.yml Main conda environment (brain-fly).
environment-brian2genn.yml Separate Brian2GeNN environment (brain-fly-brian2genn).
Monstruo-Ado.mp3 Sample audio track.
Monstruo-Ado.mp3.beats.json Pre-computed beat analysis for the sample track.

Import graph:

```
fly_dance_gui.py   ──► ado_gui_html.py
fly_dance_gui.py   ──► beat_analysis.py  (fallback if .beats.json missing)
fly_dance_live.py  ──► beat_analysis.py  (same fallback)
fly_learn_dance.py ──► beat_analysis.py  (same fallback)
fly_pop.py         ──► beat_analysis.py  (same fallback)
main.py            ──► code/benchmark.py (via sys.path insert)
```

---

Pipeline

Audio pipeline:

```
audio.mp3
   │
   ▼
beat_analysis.py
   │
   ▼
audio.mp3.beats.json
   │
   ├── fly_dance.py        ──► demo_output/fly_dance/fly_dance.mp4
   ├── fly_dance_gui.py    ──► browser GUI + demo_output/fly_dance/ado_blue.mp4
   ├── fly_dance_live.py   ──► demo_output/fly_dance/fly_dance_live.mp4
   ├── fly_learn_dance.py  ──► demo_output/fly_dance/fly_learn_dance.mp4
   └── fly_pop.py          ──► demo_output/fly_dance/fly_pop_rl.mp4
```

Neural benchmark pipeline (independent of audio):

```
main.py ──► code/benchmark.py ──► backend runners ──► data/benchmark-results.csv
                                                  └─► data/results/*.parquet
```

---

What is not in this repository

.gitignore excludes the following; they must be obtained or generated
locally:

Excluded Reason
*.parquet, data/results/* Large spike outputs. The nature_2026_07 bundle is distributed via Google Drive.
data/weight_coo.pkl, data/weight_csr.pkl PyTorch weight matrices; regenerated on first run.
demo_output/ Local simulation videos and diagnostics.
code/output* Brian2 / Brian2CUDA standalone build artifacts.
flygym/, flywire/ Separate repositories with their own Git history.
venv/, venv_flygym/ Local virtual environments.
__pycache__/, *.pyc, .DS_Store Standard Python and macOS cruft.

FlyWire source data is committed as
data/2025_Connectivity_783.parquet and data/2025_Completeness_783.csv.

---

License

Distributed under the GNU General Public License, version 2 (GPL-2.0).
See LICENSE for the full text.

The benchmark code under code/ and main.py is a derivative work of
fly-brain and remains under
the GPL terms declared in LICENSE. Original copyright notices are
preserved in the source files.

---

Credits and acknowledgments

Project Role Link
fly-brain Original whole-brain LIF model, benchmark harness, data layout https://github.com/eonsystemspbc/fly-brain
FlyGym (v2.1.0) MuJoCo biomechanical simulation, NeLy lab https://github.com/NeLy-EPFL/flygym
FlyWire (v783) Drosophila connectome dataset https://flywire.ai/
New code in this repo fly_dance*.py, beat_analysis.py, fly_pop.py, fly_sim.py, visualize_spikes.py, categorize_sez.py, explore_sez.py, ado_gui_html.py — © 2026 owarimono, GPL-2.0 —

