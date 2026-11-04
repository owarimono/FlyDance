<p align="center"> # FlyDance

<p align="center">
  <a href="https://github.com/owarimono/FlyDance">
    <img alt="FlyDance" src="https://img.shields.io/badge/FlyDance-Audio--Driven%20Fly%20Simulation-7C3AED?style=for-the-badge&labelColor=18181B">
  </a>
  <a href="https://www.python.org/">
    <img alt="Python 3.10" src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white&labelColor=18181B">
  </a>
  <a href="https://github.com/NeLy-EPFL/flygym">
    <img alt="FlyGym v2.1.0" src="https://img.shields.io/badge/FlyGym-v2.1.0-FF8C42?style=for-the-badge&labelColor=18181B">
  </a>
  <a href="https://flywire.ai/">
    <img alt="FlyWire Connectome" src="https://img.shields.io/badge/FlyWire-Connectome%20v783-64748B?style=for-the-badge&labelColor=18181B">
  </a>
  <a href="https://mujoco.org/">
    <img alt="MuJoCo" src="https://img.shields.io/badge/MuJoCo-Physics%20Simulation-00A67E?style=for-the-badge&labelColor=18181B">
  </a>
  <a href="https://github.com/owarimono/FlyDance/blob/main/LICENSE">
    <img alt="GPL-2.0" src="https://img.shields.io/badge/License-GPL--2.0-2563EB?style=for-the-badge&labelColor=18181B">
  </a>
</p>

<p align="center">
  <b>Turning music into movement through computational neuroscience.</b>
  <br>
  An experimental platform exploring audio-driven locomotion and neural control in a simulated <i>Drosophila melanogaster</i>.
</p>

---

## Overview

**FlyDance** is an experimental research and creative-coding project that explores the relationship between music, neural activity, and biological movement.

It brings together two distinct computational systems:

- **FlyGym** — a MuJoCo-based biomechanical simulation framework for studying the locomotion of adult fruit flies.
- **A FlyWire-derived neural model** — a whole-brain Leaky Integrate-and-Fire (LIF) simulation based on connectivity data from the FlyWire connectome.

The audio pipeline extracts musical features such as beats, tempo, onset strength, and chroma. These features can then drive different experimental control mechanisms, including Central Pattern Generators (CPGs), reinforcement-learning policies, and dopamine-gated plasticity.

The resulting motor behavior is simulated using FlyGym and can be rendered as video.

The repository also includes a separate multi-backend neural benchmarking system for evaluating different implementations of spiking neural network simulations.

> **Project status:** Experimental. FlyDance is intended for exploration, prototyping, and reproducible experimentation. Its current controllers should not be interpreted as biologically validated models of fly dancing or natural musical perception.

## Architecture

FlyDance consists of two largely independent pipelines.

### 1. Audio-driven locomotion

The first pipeline converts an audio track into a beat representation and uses it to control simulated fly movement.

```text
                 Audio Input
                      |
                      v
              +---------------+
              | Beat Analysis |
              | beat_analysis |
              +---------------+
                      |
                      v
                Beat Grid JSON
                      |
                      v
        +-------------+-------------+
        |             |             |
        v             v             v
     CPG Control   RL Policy    Plasticity
        |             |             |
        +-------------+-------------+
                      |
                      v
               FlyGym / MuJoCo
                      |
                      v
                Rendered Video
```

Different controllers experiment with different approaches to translating musical timing into locomotor behavior.

### 2. Neural simulation and benchmarking

The second pipeline runs spiking neural network simulations using different computational backends.

```text
             FlyWire Connectivity
                      |
                      v
              Neural Model
                      |
                      v
               Benchmark CLI
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
       Brian2      PyTorch      NEST GPU
          |           |           |
          +-----------+-----------+
                      |
                      v
               Benchmark Results
                 CSV / Parquet
```

This benchmarking infrastructure originates from the external [fly-brain](https://github.com/eonsystemspbc/fly-brain) project and is maintained separately from the audio-driven locomotion experiments.

## Features

### Audio and locomotion

| Feature | Description | Entry point |
|---|---|---|
| Audio analysis | Beat, tempo, onset-envelope, and chroma extraction | `beat_analysis.py` |
| Beat-driven locomotion | Maps musical timing onto a tripod CPG controller | `fly_dance.py` |
| Browser interface | Local GUI with streamed simulation frames | `fly_dance_gui.py` |
| Interactive simulation | MuJoCo viewer with a small neural control network | `fly_dance_live.py` |
| Dopamine-gated plasticity | Experimental single-scalar learning mechanism | `fly_learn_dance.py` |
| Reinforcement learning | REINFORCE-based discrete pose policy | `fly_pop.py` |
| Walking baseline | Standalone tripod-CPG locomotion controller | `fly_sim.py` |

### Neural analysis and benchmarking

| Feature | Description | Entry point |
|---|---|---|
| Neural benchmarks | Multi-backend simulation benchmark runner | `main.py`, `code/` |
| Spike visualization | Raster plots, PSTH, and top-neuron activity | `visualize_spikes.py` |
| SEZ exploration | Inspection of subesophageal zone neuron data | `explore_sez.py` |
| SEZ categorization | Groups neurons into functional categories | `categorize_sez.py` |

## Requirements

### Main environment

| Component | Version / Requirement |
|---|---|
| Python | 3.10 |
| Conda / Miniconda | Recent version |
| FlyGym | v2.1.0 |
| MuJoCo | Provided through FlyGym |
| librosa | Required for audio analysis |
| soundfile | Required for audio file I/O |

The main Conda environment is defined in `environment.yml`.

### Optional GPU backends

CUDA 12.x is required only for GPU-based simulation backends.

The Brian2GeNN backend requires a separate environment because its dependency constraints conflict with the main Brian2CUDA stack.

A separate environment definition is provided in:

```text
environment-brian2genn.yml
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/owarimono/FlyDance.git
cd FlyDance
```

### 2. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate brain-fly
```

### 3. Install audio dependencies

```bash
pip install librosa soundfile
```

### 4. Install FlyGym

FlyGym is an external dependency and is intentionally kept outside this repository.

Install the required version separately:

```bash
git clone --branch v2.1.0 https://github.com/NeLy-EPFL/flygym.git

pip install -e flygym
```

Refer to the official [FlyGym repository](https://github.com/NeLy-EPFL/flygym) for additional installation requirements.

### 5. Optional: Brian2GeNN environment

```bash
conda env create -f environment-brian2genn.yml
conda activate brain-fly-brian2genn
```

## Usage

The following examples assume that the required dependencies and input files are available.

### Audio analysis

Extract musical features and generate a beat-grid JSON file:

```bash
python beat_analysis.py Monstruo-Ado.mp3
```

Specify an output path:

```bash
python beat_analysis.py song.mp3 song.beats.json
```

By default, the output is written next to the input audio file.

### Beat-driven locomotion

Run the CPG-based dance simulation:

```bash
python fly_dance.py Monstruo-Ado.mp3.beats.json
```

Generated files:

```text
demo_output/
└── fly_dance/
    ├── fly_dance.mp4
    └── fly_dance_diagnostics.png
```

### Browser GUI

Launch the local browser interface:

```bash
python fly_dance_gui.py Monstruo-Ado.mp3
```

Customize the port and simulation duration:

```bash
python fly_dance_gui.py Monstruo-Ado.mp3 --port=9000 --s=60
```

The server uses Python's standard-library `http.server` module and binds to localhost. It does not require Flask or a WebSocket dependency.

Open the URL printed in the terminal manually.

### Interactive MuJoCo simulation

Launch the viewer-based controller:

```bash
python fly_dance_live.py Monstruo-Ado.mp3 --s=30
```

Use `--headless` to disable the interactive viewer.

The output is saved to:

```text
demo_output/fly_dance/fly_dance_live.mp4
```

### Dopamine-gated plasticity

Run the experimental plasticity controller:

```bash
python fly_learn_dance.py Monstruo-Ado.mp3
```

This experiment produces a simulation video and a learning-curve plot.

### Reinforcement-learning controller

Run the REINFORCE-based pose policy:

```bash
python fly_pop.py Monstruo-Ado.mp3 --s=20 --live
```

Available documented options:

- `--live`: Enable the interactive viewer.
- `--s`: Set the simulation duration in seconds.

The generated video is saved to:

```text
demo_output/fly_dance/fly_pop_rl.mp4
```

### Walking baseline

Run the standalone CPG walking controller:

```bash
python fly_sim.py
```

This produces a walking simulation video and diagnostic plots.

### Neural spike visualization

Visualize a Parquet spike-recording file:

```bash
python visualize_spikes.py data/results/round_01/brian2cpp_t1.0s_n1.parquet
```

When no input file is specified, the script uses:

```text
data/results/pytorch_t1.0s_n1.parquet
```

The visualization includes a spike raster, PSTH, and top-20 neuron activity chart.

### SEZ exploration

Inspect and categorize the SEZ neuron dataset:

```bash
python explore_sez.py
python categorize_sez.py
```

Both scripts read:

```text
data/sez_neurons.pickle
```

### Neural benchmarking

Run the default benchmark suite:

```bash
python main.py
```

Run the paper benchmark configuration:

```bash
python main.py --paper --run-label my_run
```

Run a single backend:

```bash
python main.py --brian2-cpu --t_run 1 --n_run 1
```

Disable spike recording and I/O:

```bash
python main.py --disable-spike-io
```

For all available options:

```bash
python main.py --help
```

Benchmark summaries are written to:

```text
data/benchmark-results.csv
```

Spike recordings are stored under:

```text
data/results/
```

## Project Structure

```text
FlyDance/
│
├── beat_analysis.py
├── fly_dance.py
├── fly_dance_gui.py
├── fly_dance_live.py
├── fly_learn_dance.py
├── fly_pop.py
├── fly_sim.py
│
├── visualize_spikes.py
├── explore_sez.py
├── categorize_sez.py
│
├── ado_gui_html.py
├── main.py
│
├── code/
│   ├── benchmark.py
│   ├── run_brian2_cuda.py
│   ├── run_pytorch.py
│   ├── run_nestgpu.py
│   ├── run_genn.py
│   └── run_brian2_genn.py
│
├── data/
│   ├── 2025_Connectivity_783.parquet
│   ├── 2025_Completeness_783.csv
│   └── ...
│
├── scripts/
│
├── environment.yml
├── environment-brian2genn.yml
├── LICENSE
│
├── Monstruo-Ado.mp3
└── Monstruo-Ado.mp3.beats.json
```

### Main components

| File | Responsibility |
|---|---|
| `beat_analysis.py` | Extracts audio features and generates beat-grid JSON |
| `fly_dance.py` | Beat-driven tripod CPG locomotion |
| `fly_dance_gui.py` | Local HTTP server and browser GUI |
| `fly_dance_live.py` | Interactive MuJoCo simulation with a small neural controller |
| `fly_learn_dance.py` | Dopamine-gated plasticity experiment |
| `fly_pop.py` | REINFORCE-based discrete pose policy |
| `fly_sim.py` | Standalone CPG walking baseline |
| `visualize_spikes.py` | Neural spike visualization |
| `explore_sez.py` | SEZ dataset inspection |
| `categorize_sez.py` | SEZ neuron categorization |
| `ado_gui_html.py` | Embedded HTML, CSS, and JavaScript for the GUI |
| `main.py` | Neural benchmark CLI entry point |
| `code/` | Benchmark orchestration and backend implementations |
| `data/` | Connectome data, neuron metadata, and benchmark outputs |
| `scripts/` | Supporting setup scripts and backend sources |

## Data and External Dependencies

FlyDance uses external scientific datasets and software projects that are maintained independently.

### FlyGym

[FlyGym](https://github.com/NeLy-EPFL/flygym) provides the biomechanical simulation environment used for locomotion experiments.

It is installed separately and is not vendored into this repository.

### FlyWire

[FlyWire](https://flywire.ai/) provides the connectome data used by the external neural-modeling infrastructure.

The repository includes connectivity and completeness data associated with dataset version **783**.

### fly-brain

[fly-brain](https://github.com/eonsystemspbc/fly-brain) is the upstream project associated with the whole-brain LIF model and the neural benchmarking infrastructure.

FlyDance retains the relevant benchmark components and data layout while keeping the external project separate.

The source repository and its history remain available through the upstream project.

## Repository Data

Some generated and large files are intentionally excluded from Git.

| Path | Reason |
|---|---|
| `*.parquet` and `data/results/*` | Large neural spike recordings |
| `data/weight_coo.pkl` | Generated PyTorch weight matrices |
| `data/weight_csr.pkl` | Generated PyTorch weight matrices |
| `demo_output/` | Locally generated videos and diagnostics |
| `code/output*` | Backend build artifacts |
| `flygym/`, `flywire/` | External repositories |
| `venv/`, `venv_flygym/` | Local Python environments |
| `__pycache__/`, `*.pyc` | Python cache files |
| `.DS_Store` | macOS metadata |

The large `nature_2026_07` spike-data bundle is distributed separately.

The committed FlyWire data files include:

```text
data/2025_Connectivity_783.parquet
data/2025_Completeness_783.csv
```

## Experimental Scope and Limitations

FlyDance is an experimental platform rather than a validated biological model of music perception or dance.

Its controllers explore different computational approaches to generating movement from musical timing. These include deterministic CPG modulation, small neural networks, plasticity mechanisms, and reinforcement learning.

The presence of a connectome-derived neural model in the repository does not imply that the audio-driven controllers reproduce the biological neural mechanisms of *Drosophila*.

Likewise, the neural benchmarking pipeline is a separate computational component and should not be interpreted as evidence that the locomotion controllers are biologically equivalent to a whole-brain simulation.

Quantitative validation, cross-backend comparisons, and biological interpretation should be evaluated independently.

## License

FlyDance is distributed under the **GNU General Public License v2.0 (GPL-2.0)**.

See [LICENSE](LICENSE) for the complete license text.

The benchmark code under `code/` and `main.py` contains derivative work from the external `fly-brain` project and retains the applicable GPL licensing terms and original copyright notices.

## Credits and Acknowledgments

FlyDance builds upon the work of several open-source and scientific projects.

| Project | Contribution |
|---|---|
| [FlyGym](https://github.com/NeLy-EPFL/flygym) | Biomechanical simulation of *Drosophila* |
| [FlyWire](https://flywire.ai/) | Connectome dataset and neural connectivity |
| [fly-brain](https://github.com/eonsystemspbc/fly-brain) | Whole-brain LIF model and neural benchmarking infrastructure |
| [MuJoCo](https://mujoco.org/) | Physics simulation engine |

### Original Work

Audio analysis, locomotion controllers, experimental learning mechanisms, GUI components, and visualization utilities developed for FlyDance are authored by **owarimono**.

Copyright © 2026 owarimono.

---

<p align="center">
  <i>From neural activity to movement — and from music to motion.</i>
</p>
