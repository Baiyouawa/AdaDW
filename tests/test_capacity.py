from adawd_preexp.capacity import build_model, plan_sweep


def test_patchtst_depth_sweep_is_centered_on_official_raw_capacity():
    runs = plan_sweep("PatchTST", "ETTh1", "depth", output_length=96, seeds=[42])
    assert [run.depth for run in runs] == [1, 2, 3, 4, 5]
    assert {run.width for run in runs} == {128}


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
