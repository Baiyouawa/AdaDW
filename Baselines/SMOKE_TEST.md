# Historical Eight-Backbone Forecasting Smoke Test

All eight registered Backbones completed a real train/validation/test cycle on
ILI using input length 24, prediction length 24, seed 42, one epoch and batch
size 8. This is an integration check, not a benchmark: one epoch is insufficient
for comparing model accuracy.

This record predates the current protocol signatures and explicit RAW
`benchmark_config` separation. It proves only that the eight adapter paths once
completed on ILI; it must not be used as evidence for all nine datasets or the
current 864-run protocol.

The runs were written outside the formal result tree to
`/tmp/adawd-eight-smoke`. Every run produced a best checkpoint,
`test_metrics.json`, and finite `inputs.npy`, `prediction.npy`, and `targets.npy`
arrays with shape `[171, 24, 7]`. The final short batch was also verified as
written. A combined local-loss table was successfully generated from all eight
manifests.

| Model | Parameters | Test MAE | Test RMSE | Test MAPE | Test WAPE | Train+test wall time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Crossformer | 46,842,416 | 71,256.4 | 200,847.7 | 0.440 | 0.461 | 14.58 |
| PatchTST | 812,582 | 31,374.5 | 93,941.0 | 0.635 | 0.517 | 3.40 |
| TimesNet | 18,466,487 | 30,400.5 | 96,462.6 | 0.641 | 0.593 | 7.38 |
| iTransformer | 802,840 | 29,577.8 | 88,808.0 | 0.549 | 0.514 | 3.27 |
| TimeMixer | 529,652 | 40,526.1 | 111,745.9 | 0.944 | 0.800 | 6.04 |
| WPMixer | 3,247,934 | 29,884.0 | 88,526.4 | 0.665 | 0.579 | 6.13 |
| TimeFilter | 284,312 | 31,676.3 | 94,760.4 | 0.687 | 0.543 | 3.79 |
| MultiPatchFormer | 1,534,014 | 26,374.1 | 82,322.3 | 0.591 | 0.564 | 5.08 |

ILI contains channels on very different physical scales, including provider and
case counts, so raw-scale MAE/RMSE are numerically large. They are included only
to prove that original-scale evaluation and result persistence work. Formal
comparisons require the configured training budget and multiple seeds.

During this test, MultiPatchFormer's multiscale alignment initially used CUDA
linear interpolation, whose backward pass is unavailable under strict
determinism. It was replaced with a learnable deterministic patch projection and
the complete run then passed.
