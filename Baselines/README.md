# Backbone inventory

`basicts/` started as a minimal BasicTS snapshot from `../DropoutTS` commit
`64a096e`. The five models found there remain traceable to that snapshot. Three
2025 forecasting models were subsequently adapted from their papers and official
repositories into the same `[batch, time, channel]` interface.

| Backbone | Year | Status | Depth control | Width control |
| --- | ---: | --- | --- | --- |
| Crossformer | 2023 | migrated | `num_layers` | FFN `intermediate_size` |
| PatchTST | 2023 | migrated | `num_layers` | `hidden_size` with coupled FFN |
| TimesNet | 2023 | migrated | `num_layers` | Inception `intermediate_size` |
| iTransformer | 2024 | migrated | `num_layers` | FFN `intermediate_size` |
| TimeMixer | 2024 | migrated | `num_layers` | `hidden_size` with coupled mixing FFN |
| WPMixer | 2025 | adapted | resolution mixer blocks | mixer FFN `intermediate_size` |
| TimeFilter | 2025 | adapted | filtered graph blocks | graph-block FFN `intermediate_size` |
| MultiPatchFormer | 2025 | adapted | temporal/channel encoder blocks | encoder FFN `intermediate_size` |

These controls are architecture-level capacity proxies. In particular,
Crossformer's `num_layers` changes multi-scale encoder/decoder structure and
TimesNet's width changes convolutional channels, so equal integer settings do
not imply equal FLOPs across Backbones. The sweep records actual parameters and
profiled FLOPs for this reason.

Formal RAW forecasting does not infer these settings from capacity labels. Each
model has a complete, explicit and distinct `benchmark_config` in
`Baselines/registry.json`. The `raw_depth/raw_width` fields retained for the
depth/width pre-experiment are a separate capacity reference for those sweeps.

## 2025 model sources

- WPMixer: `Secure-and-Intelligent-Systems-Lab/WPMixer`, commit
  `74104c9dddd54d279eb8323f48934b4fd75fcae7`, MIT. The adapter retains the
  multi-resolution wavelet, patching and MLP-mixing path while exposing repeatable
  mixer depth and explicit FFN width.
- TimeFilter: `TROUBADOUR000/TimeFilter`, commit
  `dffde87e4fff0fdeeebbacde03dc1e432e15b3a1`. The official repository does not
  declare a software license, so this tree contains an independent BasicTS
  implementation of the paper's patch graph construction and filtration design.
- MultiPatchFormer: `bioinfoUQAM/MultiPatchFormer`, commit
  `965e6bd60822d509183253ef9c51fc3f9efe23f3`. The official repository does not
  declare a software license, so this tree contains an independent BasicTS
  implementation of its multi-scale patch, temporal/channel attention and
  segmented prediction design.
