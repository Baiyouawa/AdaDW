import pytest

from adawd_preexp.capacity import plan_sweep


def test_depth_sweep_has_four_candidates_per_seed_and_horizon():
    runs = plan_sweep("PatchTST", "ETTh1", "depth", output_length=96, seeds=[42])
    assert [run.depth for run in runs] == [1, 2, 4, 8]
    assert {run.width for run in runs} == {1024}


def test_missing_backbone_is_explicit():
    with pytest.raises(RuntimeError, match="No implementation in DropoutTS"):
        plan_sweep("WPMMixer", "ETTh1", "raw")
