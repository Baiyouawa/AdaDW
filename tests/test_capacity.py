from adawd_preexp.capacity import build_model, plan_sweep


def test_patchtst_depth_sweep_is_centered_on_official_raw_capacity():
    runs = plan_sweep("PatchTST", "ETTh1", "depth", output_length=96, seeds=[42])
    assert [run.depth for run in runs] == [1, 2, 3, 4, 5]
    assert {run.width for run in runs} == {16}
    assert {run.coupled_width for run in runs} == {128}


def test_model_specific_width_sweeps_have_two_scales_around_raw():
    expected = {
        "PatchTST": [4, 8, 16, 32, 64],
        "TimeMixer": [4, 8, 16, 32, 64],
        "MultiPatchFormer": [64, 128, 256, 512, 1024],
    }
    expected_d_ff = {
        "PatchTST": [32, 64, 128, 256, 512],
        "TimeMixer": [8, 16, 32, 64, 128],
        "MultiPatchFormer": [64, 128, 256, 512, 1024],
    }
    for model, widths in expected.items():
        runs = plan_sweep(model, "ETTh1", "width", output_length=96, seeds=[3407])
        assert [run.width for run in runs] == widths
        assert [run.coupled_width for run in runs] == expected_d_ff[model]
        for run in runs:
            _, model_config, _ = build_model(run)
            assert model_config.hidden_size == run.width
            assert model_config.intermediate_size == run.coupled_width


def test_new_backbones_are_registered():
    for model in ("WPMixer", "TimeFilter", "MultiPatchFormer"):
        run = plan_sweep(model, "ILI", "raw", output_length=24, seeds=[42])[0]
        model_class, model_config, use_timestamps = build_model(run)

        assert model_class.__name__.endswith("ForForecasting")
        assert model_config.num_layers == run.depth
        assert model_config.intermediate_size == run.width
        assert not use_timestamps


def test_raw_runs_use_each_backbones_explicit_benchmark_configuration():
    expected = {
        "Crossformer": (2, 2048),
        "PatchTST": (1, 1024),
        "TimesNet": (1, 1024),
        "iTransformer": (1, 1024),
        "TimeMixer": (1, 1024),
        "WPMixer": (2, 1024),
        "TimeFilter": (2, 256),
        "MultiPatchFormer": (1, 512),
    }
    for model, (depth, effective_width) in expected.items():
        run = plan_sweep(model, "ETTh1", "raw", output_length=96, seeds=[3407])[0]
        _, model_config, _ = build_model(run)
        assert model_config.num_layers == depth
        assert model_config.intermediate_size == effective_width

    patch_run = plan_sweep("PatchTST", "ETTh1", "raw", 96, [3407])[0]
    _, patch_config, _ = build_model(patch_run)
    assert patch_config.hidden_size == 256
    assert patch_config.n_heads == 1

    mixer_run = plan_sweep("TimeMixer", "ETTh1", "raw", 96, [3407])[0]
    _, mixer_config, _ = build_model(mixer_run)
    assert mixer_config.hidden_size == 256
