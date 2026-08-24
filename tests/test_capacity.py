from adawd_preexp.capacity import build_model, plan_sweep


def test_depth_sweep_has_four_candidates_per_seed_and_horizon():
    runs = plan_sweep("PatchTST", "ETTh1", "depth", output_length=96, seeds=[42])
    assert [run.depth for run in runs] == [1, 2, 4, 8]
    assert {run.width for run in runs} == {1024}


def test_new_backbones_are_registered():
    for model in ("WPMixer", "TimeFilter", "MultiPatchFormer"):
        run = plan_sweep(model, "ILI", "raw", output_length=24, seeds=[42])[0]
        model_class, model_config, use_timestamps = build_model(run)

        assert model_class.__name__.endswith("ForForecasting")
        assert model_config.num_layers == run.depth
        assert model_config.intermediate_size == run.width
        assert not use_timestamps
