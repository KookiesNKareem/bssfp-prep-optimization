# Joint T₂-preparation and fat-saturation design

Design a shorter joint T₂-preparation/fat-saturation RF pulse for bSSFP. The
objective targets conventional-or-better blood–myocardium contrast, fat
suppression, and B₀/B₁ robustness under a matched RF-energy limit.

KomaMRI supplies forward simulations and a manual adjoint supplies RF gradients.
The preparation RF changes; acquisition timing, gradients, ramp pulses, and
readout RF stay fixed. The script exports the pulse and compares full-phantom
images with conventional preparation.

## Run

Requires Julia 1.11.7 and an NVIDIA CUDA GPU.

```sh
git clone --recurse-submodules https://github.com/KookiesNKareem/bssfp-prep-optimization.git
cd bssfp-prep-optimization
cp cuda.toml LocalPreferences.toml
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. optimize.jl
```

Set the phantom, conventional reference, initial joint RF pulse, B₀ map, B₁
scales, and optimization budget at the top of `optimize.jl`. Select a GPU with
`CUDA_VISIBLE_DEVICES`. Outputs go to `results/joint_p8/`.

For a warm REPL:

```julia
include("optimize.jl")
using .BSSFPPrep
prepare!()
fit!()
replay!()
report!()
```

## Objective and inputs

The objective is `10 * acquired_p8 + heatmap_p8 + RF_peak`. Acquired-signal terms
penalize contrast deficits, excess fat, water-signal loss, and B₁ sensitivity.
Heatmap terms target conventional nominal-field water magnetization and zero fat
magnetization across ±100 Hz B₀ and 85–115% B₁. Eighth-power aggregation emphasizes
large errors; the RF peak term discourages amplitudes above 17 µT. Preparation
RF energy is capped at conventional energy.

`data/` provides example sequence inputs, a synthetic static cardiac phantom,
starting controls, and per-condition fat targets. Replace these with compatible
inputs for your design. Set `b0_map` to a JLD2 file containing `b0_hz` in phantom
spin order to add a field map. The current acquisition model supports the example's
1.5 mm, 96-partition geometry, five dummy heartbeats, and 60 bpm; other protocols
require adapting that model.

`src/` contains supporting functions; `deps/` pins KomaMRI. Optimization uses
1 µs RF integration and full-phantom simulation uses 20 µs.

MIT license; KomaMRI retains its own MIT license.
