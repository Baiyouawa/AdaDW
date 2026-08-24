import pytest
import torch

from adawd_preexp.capacity import build_model, plan_sweep


BACKBONES = (
    "Crossformer",
    "PatchTST",
    "TimesNet",
    "iTransformer",
    "TimeMixer",
    "WPMixer",
    "TimeFilter",
    "MultiPatchFormer",
)


@pytest.mark.parametrize("model_name", BACKBONES)
def test_backbone_forecast_shape(model_name):
    run = plan_sweep(model_name, "ILI", "width", output_length=24, seeds=[42])[0]
    model_class, model_config, use_timestamps = build_model(run)
    model = model_class(model_config).eval()
    inputs = torch.randn(2, 24, 7)
    kwargs = {"inputs_timestamps": torch.zeros(2, 24, 4)} if use_timestamps else {}

    with torch.inference_mode():
        prediction = model(inputs, **kwargs)

    assert prediction.shape == (2, 24, 7)
    assert torch.isfinite(prediction).all()


@pytest.mark.parametrize("model_name", BACKBONES)
def test_depth_and_width_candidates_change_parameter_count(model_name):
    depth_runs = plan_sweep(model_name, "ILI", "depth", output_length=24, seeds=[42])
    width_runs = plan_sweep(model_name, "ILI", "width", output_length=24, seeds=[42])

    def parameter_counts(runs):
        counts = []
        for run in runs:
            model_class, model_config, _ = build_model(run)
            model = model_class(model_config)
            counts.append(sum(parameter.numel() for parameter in model.parameters()))
        return counts

    depth_counts = parameter_counts(depth_runs)
    width_counts = parameter_counts(width_runs)
    assert depth_counts == sorted(depth_counts) and len(set(depth_counts)) == len(depth_counts)
    assert width_counts == sorted(width_counts) and len(set(width_counts)) == len(width_counts)
