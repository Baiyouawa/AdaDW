import pytest

from pre_experiments.run_model_preexperiments import build_plan


def test_three_year_model_plan_contains_30_runs():
    plan = build_plan(
        ["PatchTST", "TimeMixer", "MultiPatchFormer"],
        "ETTh1",
        96,
        [3407],
    )

    assert len(plan) == 30
    assert {run["axis"] for run in plan} == {"depth", "width"}
    assert {run["model"] for run in plan} == {
        "PatchTST",
        "TimeMixer",
        "MultiPatchFormer",
    }


def test_three_year_model_plan_rejects_duplicate_years():
    with pytest.raises(ValueError, match="2023, 2024 and 2025"):
        build_plan(["PatchTST", "TimesNet", "MultiPatchFormer"], "ETTh1", 96, [3407])
