FlyDance 🪰🎵

An audio‑synchronized, whole‑brain simulation of the adult Drosophila — dancing to the beat.

https://img.shields.io/badge/License-GPL%202.0%2B-blue.svg
https://img.shields.io/badge/python-3.9%2B-blue.svg
https://img.shields.io/badge/CUDA-12.x-green.svg

---

Overview

FlyDance is an original project that brings together two powerful open‑source ecosystems — FlyGym for biomechanical fly simulation and fly-brain / FlyWire-based LIF emulation for whole‑brain neural activity — and uses them as the runtime environment for a new class of experiments: music‑driven neural simulation.

In FlyDance, the ~138k‑neuron connectome‑based brain model is driven in real time by audio beat detection, live microphone input, or user‑selected tracks, producing observable spiking activity that is synchronized, visualized, and analyzed. The project includes an interactive GUI, live dashboards, spike raster/rate visualization, and plasticity experiments.

Clarification: FlyDance is not a fork of FlyGym or fly-brain. Those projects are used as underlying environments/libraries. All scripts, analysis pipelines, GUI components, and simulation‑synchronization logic in this repository are original work by the author.

---

What FlyDance Adds

Layer Provided by What FlyDance contributes
Biomechanical fly model FlyGym Uses as simulation environment
Whole‑brain LIF emulation (~138k neurons) fly-brain / FlyWire Uses as neural substrate
Audio beat/onset detection — beat_analysis.py
Beat‑synchronized neural driving — fly_dance.py, fly_dance_live.py
Interactive GUI + audio playback — fly_dance_gui.py, ado_gui_html.py
Live microphone‑driven simulation — fly_dance_live.py
Learning / plasticity experiments — fly_learn_dance.py, fly_pop.py
Lightweight simulation wrapper — fly_sim.py
Spike visualization (raster/rate) — visualize_spikes.py

---

Features

· 🎧 Audio beat detection — extracts tempo, beat times, and onset events (librosa‑based) from any audio file.
· 🧠 Music‑synchronized whole‑brain simulation — drives the connectome‑based LIF model in sync with detected beats or live audio.
· 🖥️ Interactive GUI — real‑time visualization of spiking activity alongside audio playback.
· 🎤 Live input mode — reacts to microphone input for interactive demos and performances.
· 📊 Spike visualization — raster plots and rate animations for exported spike data.
· 🧪 Experimental tooling — plasticity/learning scripts and population‑level analysis.
· ⚡ Lightweight entry point — fly_sim.py for quick experiments without the full benchmark harness.

---

Quick Start

Prerequisites

· Linux (tested on Ubuntu 22.04 under WSL2)
· NVIDIA GPU with CUDA 12.x (for GPU‑accelerated backends)
· Miniconda / Anaconda
· pkg-config and libffi-dev (for PyGeNN on Ubuntu/WSL)

Installation

```bash
# 1. Clone the repository
git clone https://github.com/owarimono/FlyDance.git
cd FlyDance

# 2. Create the main Conda environment
conda env create -f environment.yml
conda activate brain-fly

# 3. Install audio‑analysis dependencies
pip install librosa soundfile sounddevice
```

Usage

Analyze beats in an audio file

```bash
python beat_analysis.py --audio Monstruo-Ado.mp3
```

Launch the beat‑synchronized GUI

```bash
python fly_dance_gui.py --audio Monstruo-Ado.mp3
```

Run a live, microphone‑driven simulation

```bash
python fly_dance_live.py
```

Generate spike visualizations

```bash
python visualize_spikes.py --input data/results/round_01/brian2cpp_t1.0s_n1.parquet
```

Quick lightweight simulation

```bash
python fly_sim.py --duration 1.0
```

---

Project Structure

```
FlyDance/
├── beat_analysis.py          # Audio beat/onset detection
├── fly_dance.py              # Beat‑driven simulation wrapper
├── fly_dance_gui.py          # Interactive GUI with audio playback
├── fly_dance_live.py         # Live microphone‑driven simulation
├── fly_learn_dance.py        # Plasticity / learning experiments
├── fly_pop.py                # Population‑level analysis
├── fly_sim.py                # Lightweight simulation interface
├── visualize_spikes.py       # Spike raster & rate visualizations
├── ado_gui_html.py           # HTML dashboard for the Ado track
├── Monstruo-Ado.mp3          # Example audio track
├── Monstruo-Ado.mp3.beats.json  # Pre‑computed beat annotations
├── code/                     # Simulation backends
├── data/                     # Connectome data & results
└── scripts/                  # Setup and utility scripts
```

---

Built On

FlyDance would not be possible without these outstanding open‑source projects, which serve as its simulation environments:

· FlyGym — NeuroMechFly biomechanical simulation of Drosophila.
· fly-brain — Whole‑brain leaky integrate‑and‑fire emulation based on the FlyWire connectome (Shiu et al., 2023–2024).
· FlyWire Consortium — Connectome data (version 783).
· Simulation backends: Brian2, Brian2CUDA, PyTorch, NEST GPU, GeNN, Brian2GeNN.

All credit for the underlying environment and neural model belongs to their respective authors. FlyDance's original contribution is the audio‑visual synchronization layer, GUI tooling, and experimental scripts built on top.

---

License

This project is distributed under the GNU General Public License v2.0 or later (GPL‑2.0‑or‑later). See the LICENSE file for the full text.

Underlying dependencies (FlyGym, fly-brain, Brian2, etc.) retain their respective licenses.

---

Contributing

Contributions are welcome — new audio features, visualization modes, or synchronization strategies. Please open an issue to discuss larger changes before submitting a pull request.

---

Acknowledgments

· The FlyWire Consortium for the connectome data.
· The FlyGym and fly-brain authors for the simulation environments that make this project possible.
· The developers of Brian2, Brian2CUDA, PyTorch, NEST GPU, and GeNN for their simulation backends.

---

Maintained by owarimono.

