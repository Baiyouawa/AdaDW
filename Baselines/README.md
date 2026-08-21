# Backbone inventory

`basicts/` is a minimal BasicTS snapshot from `../DropoutTS` commit `64a096e`.
Only the five selected model packages found in that repository were copied.
The snapshot is kept separate from future AdaWD method code so RAW results stay
traceable to the original baseline implementation.

| Backbone | Year | Status | Depth control | Width control |
| --- | ---: | --- | --- | --- |
| Crossformer | 2023 | migrated | `num_layers` | FFN `intermediate_size` |
| PatchTST | 2023 | migrated | `num_layers` | FFN `intermediate_size` |
| TimesNet | 2023 | migrated | `num_layers` | Inception `intermediate_size` |
| iTransformer | 2024 | migrated | `num_layers` | FFN `intermediate_size` |
| TimeMixer | 2024 | migrated | `num_layers` | mixing-block `intermediate_size` |
| WPMMixer | 2025 | missing | pending official implementation | pending |
| TimeFilter | 2025 | missing | pending official implementation | pending |
| MultiPatchFormer | 2025 | missing | pending official implementation | pending |

These controls are architecture-level capacity proxies. In particular,
Crossformer's `num_layers` changes multi-scale encoder/decoder structure and
TimesNet's width changes convolutional channels, so equal integer settings do
not imply equal FLOPs across Backbones. The sweep records actual parameters and
profiled FLOPs for this reason.

