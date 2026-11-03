
FlyDance

Audio-driven choreography for a connectome-scale Drosophila brain simulation.

Overview

FlyDance couples two open-source systems: FlyGym v2.1.0 for MuJoCo-based biomechanical simulation of the adult fly, and a FlyWire-derived whole-brain leaky integrate-and-fire (LIF) model (~138k neurons) inherited from the fly-brain project. Audio beat analysis produces an offline beat grid, and the dance scripts map that grid onto central pattern generators (CPGs), learned policies, or dopaminergic plasticity rules so that leg and wing motion tracks the music. The repository also retains the multi-backend neural benchmark harness from fly-brain. This is a research/creative project, not a production tool.

Demo / Preview

The repository ships Monstruo-Ado.mp3 and a pre-computed Monstruo-Ado.mp3.beats.json. The primary interactive flow is:

```bash
python fly_dance_gui.py Monstruo-Ado.mp3
```

This starts a local HTTP server (default http://localhost:8765), opens a browser page, and lets you press DANCE. The page streams MuJoCo-rendered frames, beat position, on-beat/off-beat sync ratio, and policy magnitude while the track plays. After the run, demo_output/fly_dance/ado_blue.mp4 is available through the page.

Features

· Beat, tempo, onset-envelope, and chroma extraction from arbitrary audio → beat_analysis.py
· Beat-locked CPG gait modulation (frequency from BPM, amplitude from loudness) → fly_dance.py
· Browser-based GUI with live MuJoCo frames, audio playback, and telemetry → fly_dance_gui.py, ado_gui_html.py
· "Live" MuJoCo viewer mode with a small linear brain trained by a three-factor rule → fly_dance_live.py
· Dopamine-gated plasticity experiment with a single learned scalar → fly_learn_dance.py
· REINFORCE-trained discrete pose "pops" on the beat → fly_pop.py
· Standalone tripod-CPG walking baseline → fly_sim.py
· Spike raster, PSTH, and top-neuron plots from parquet spike exports → visualize_spikes.py
· SEZ neuron inventory and category summaries → categorize_sez.py, explore_sez.py
· Multi-backend benchmark runner (Brian2 CPU, Brian2CUDA, PyTorch, NEST GPU, GeNN, Brian2GeNN) → main.py + code/

Requirements

· Python 3.10 (from environment.yml)
· Conda / Miniconda
· FlyGym v2.1.0 — not vendored; must be installed separately
· FlyWire-derived data — data/2025_Connectivity_783.parquet and data/2025_Completeness_783.csv are committed
· CUDA 12.x — only required for GPU backends; CPU Brian2 works without it
· MuJoCo — pulled in by FlyGym; imported directly by the GUI and viewer scripts
· librosa — audio analysis, not pinned in environment.yml

The Brian2GeNN backend requires a separate environment because brian2genn==1.7.0 pins brian2<2.6, which conflicts with the Brian2CUDA stack.

Installation

```bash
# 1. Clone FlyDance
git clone https://github.com/owarimono/FlyDance.git
cd FlyDance

# 2. Create the main environment
conda env create -f environment.yml
conda activate brain-fly

# 3. Install audio-analysis dependencies
pip install librosa soundfile

# 4. Install FlyGym v2.1.0 separately
git clone --branch v2.1.0 https://github.com/NeLy-EPFL/flygym.git
pip install -e flygym

# 5. Optional: Brian2GeNN environment
conda env create -f environment-brian2genn.yml
conda activate brain-fly-brian2genn
```

soundfile is the audio I/O backend used by librosa; it is not listed in environment.yml. sounddevice is not imported anywhere in the current code and is not required.

Usage

Beat analysis

```bash
# Writes Monstruo-Ado.mp3.beats.json by default
python beat_analysis.py Monstruo-Ado.mp3

# Optional explicit output path
python beat_analysis.py song.mp3 song.beats.json
```

beat_analysis.py accepts the input audio as sys.argv[1] and an optional output JSON as sys.argv[2].

Beat-driven CPG dance

```bash
python fly_dance.py Monstruo-Ado.mp3.beats.json
```

Writes demo_output/fly_dance/fly_dance.mp4 and demo_output/fly_dance/fly_dance_diagnostics.png.

Browser GUI

```bash
# Positional audio defaults to Monstruo-Ado.mp3
python fly_dance_gui.py Monstruo-Ado.mp3

# Custom port and duration
python fly_dance_gui.py Monstruo-Ado.mp3 --port=9000 --s=60
```

Flags: --port=8765 (default), --s=30.0 (default). The GUI serves ado_gui_html.py over stdlib http.server; no Flask or websockets.

Live MuJoCo viewer and plasticity

```bash
# Viewer window; --headless disables it
python fly_dance_live.py Monstruo-Ado.mp3 --s=30

# Dopamine-gated single-scalar learning
python fly_learn_dance.py Monstruo-Ado.mp3
```

fly_dance_live.py writes demo_output/fly_dance/fly_dance_live.mp4. fly_learn_dance.py writes an MP4 and a learning-curve plot. "Live" in fly_dance_live.py refers to the MuJoCo viewer window, not microphone input; the script reads an audio file and its beat JSON.

REINFORCE pose pops

```bash
python fly_pop.py Monstruo-Ado.mp3 --s=20 --live
```

Flags: --live (viewer), --s=20.0 (default). Writes demo_output/fly_dance/fly_pop_rl.mp4. The docstring mentions --pose=choreo|random, but the current code does not parse that flag; only the default learned policy is available.

CPG walking baseline

```bash
python fly_sim.py
```

Writes demo_output/cpg_controller/cpg_controller.mp4 and demo_output/cpg_controller/cpg_controller_diagnostics.png.

Spike visualization

```bash
python visualize_spikes.py data/results/round_01/brian2cpp_t1.0s_n1.parquet
```

Defaults to data/results/pytorch_t1.0s_n1.parquet when no argument is given. Writes <input>_plot.png with a raster, PSTH, and top-20 neuron bar chart.

SEZ inspection

```bash
python explore_sez.py
python categorize_sez.py
```

Both read data/sez_neurons.pickle and print summaries to stdout. They have no CLI flags and no file outputs.

Neural benchmark harness

```bash
# Default suite: five main backends, default durations and trial counts
python main.py

# Paper grid: t_run=(0.1, 1, 10, 100), n_run=(1, 4, 8, 16, 32), 5 rounds
python main.py --paper --run-label my_run

# Single backend
python main.py --brian2-cpu --t_run 1 --n_run 1

# Timing-only run without spike I/O
python main.py --disable-spike-io
```

Results are appended to data/benchmark-results.csv; spike parquet files go under data/results/. Run python main.py --help for the full flag list.

Project structure

File / Directory Purpose
beat_analysis.py Offline beat, tempo, onset-envelope, and chroma extraction; writes .beats.json.
fly_dance.py Maps a beat grid onto a tripod CPG; renders a dancing fly video and diagnostics.
fly_dance_gui.py Stdlib HTTP server + browser GUI; runs the simulation in a thread and streams JSON state and JPEG frames.
fly_dance_live.py MuJoCo viewer mode with a 3×3 linear brain trained by a three-factor rule.
fly_learn_dance.py Dopamine-gated plasticity experiment with a single learned beat-to-leg scalar.
fly_pop.py REINFORCE-trained discrete pose policy for beat-snapped "pops".
fly_sim.py Standalone CPG-driven walking simulation based on the FlyGym tutorial.
visualize_spikes.py Raster, PSTH, and top-neuron plots from spike parquet files.
categorize_sez.py Groups SEZ neuron types into descending, sensory, motor, and other categories.
explore_sez.py Prints the structure and first entries of data/sez_neurons.pickle.
ado_gui_html.py HTML/CSS/JS string for the Ado Blue GUI; imported by fly_dance_gui.py.
main.py CLI entry point for the multi-backend neural benchmark suite.
code/ Benchmark orchestrator (benchmark.py), backend runners, and comparison scripts.
data/ FlyWire connectivity and completeness data, SEZ pickle, benchmark CSV, and results.
scripts/ WSL/CUDA setup script and NEST GPU source files.
environment.yml Main conda environment (brain-fly).
environment-brian2genn.yml Separate Brian2GeNN environment (brain-fly-brian2genn).
Monstruo-Ado.mp3 Sample audio track.
Monstruo-Ado.mp3.beats.json Pre-computed beat analysis for the sample track.

Dependency notes: fly_dance_gui.py imports ado_gui_html.HTML; fly_dance_gui.py, fly_dance_live.py, fly_learn_dance.py, and fly_pop.py all import beat_analysis when the .beats.json is missing. main.py adds code/ to sys.path and imports benchmark.

Pipeline / Workflow

```text
audio.mp3
   │
   ▼
beat_analysis.py
   │
   ▼
audio.mp3.beats.json
   │
   ├── fly_dance.py ──────────► demo_output/fly_dance/fly_dance.mp4
   ├── fly_dance_gui.py ──────► browser GUI + ado_blue.mp4
   ├── fly_dance_live.py ─────► fly_dance_live.mp4
   ├── fly_learn_dance.py ────► fly_learn_dance.mp4
   └── fly_pop.py ────────────► fly_pop_rl.mp4
```

Separately, the neural benchmark pipeline runs independently of audio:

```text
main.py → code/benchmark.py → backend runners → data/benchmark-results.csv
                                              → data/results/*.parquet
```

Not included in this repo

.gitignore intentionally excludes:

· *.parquet and data/results/* — large spike outputs; the nature_2026_07 bundle is distributed via Google Drive.
· data/weight_coo.pkl, data/weight_csr.pkl — generated PyTorch weight matrices.
· demo_output/ — local simulation videos and diagnostics.
· code/output* — Brian2/Brian2CUDA standalone build artifacts.
· flygym/, flywire/ — separate repositories with their own Git history.
· venv/, venv_flygym/, __pycache__/, *.pyc, .DS_Store.

License

GPL-2.0 — see LICENSE.

The LICENSE file contains the GNU General Public License, version 2 (June 1991). The legacy README and repository badge describe the project as GPL-2.0-or-later. No GPL-3.0 license text is present in the repository. The benchmark code under code/ and main.py is a derivative work of fly-brain and remains under the GPL terms declared in LICENSE.

Credits & Acknowledgments

· fly-brain — https://github.com/eonsystemspbc/fly-brain — original whole-brain LIF model, benchmark harness, and data layout. Original copyright notices remain in the source files and LICENSE.
· FlyGym — https://github.com/NeLy-EPFL/flygym — v2.1.0, MuJoCo biomechanical simulation environment, NeLy lab.
· FlyWire — https://flywire.ai/ — Drosophila connectome data (version 783).
· New code — fly_dance*.py, beat_analysis.py, fly_pop.py, fly_sim.py, visualize_spikes.py, categorize_sez.py, explore_sez.py, ado_gui_html.py — copyright © 2026 owarimono, released under the GPL terms declared in LICENSE.

---

Summary of files read and entry points

Files read: beat_analysis.py, fly_dance.py, fly_dance_gui.py, fly_dance_live.py, fly_learn_dance.py, fly_pop.py, fly_sim.py, visualize_spikes.py, categorize_sez.py, explore_sez.py, ado_gui_html.py, main.py, code/benchmark.py (header), environment.yml, environment-brian2genn.yml, .gitignore, .gitattributes, LICENSE, and directory listings for code/, data/, scripts/.

Entry points found:

· beat_analysis.py — positional audio, optional output JSON.
· fly_dance.py — required positional beats JSON.
· fly_dance_gui.py — positional audio (default Monstruo-Ado.mp3), --port=, --s=.
· fly_dance_live.py — positional audio, --headless, --s=.
· fly_learn_dance.py — positional audio.
· fly_pop.py — positional audio, --live, --s=.
· fly_sim.py — no args.
· visualize_spikes.py — positional parquet with a default.
· categorize_sez.py, explore_sez.py — no args.
· main.py — extensive argparse suite for benchmark backends, durations, trials, rounds, and labels.

Ambiguities and omissions:

· License is GPL-2.0, not GPL-3.0. The LICENSE file is GNU GPL v2. The old README calls it GPL-2.0-or-later. I reported the actual file, not the expected GPL-3.0.
· fly_pop.py documents --pose=choreo|random, but the code does not parse --pose; only the learned policy is implemented.
· fly_dance_live.py is described in the old README as microphone-driven, but the code reads an audio file and uses "live" to mean a MuJoCo viewer window. No sounddevice import exists.
· ado_gui_html.py has no __main__ block; it is an HTML string consumed by fly_dance_gui.py.
· categorize_sez.py and explore_sez.py are short exploratory scripts with a hardcoded data/sez_neurons.pickle path and no CLI.