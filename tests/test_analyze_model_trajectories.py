from pathlib import Path

import pytest

from pre_experiments.analyze_model_trajectories import analyze


def test_analyze_explains_how_to_build_a_missing_loss_table(tmp_path: Path):
    losses = tmp_path / "local_losses.csv"

    with pytest.raises(
        FileNotFoundError,
        match=r"local_losses\.csv.*pixi run model-preexp-losses",
    ):
        analyze(losses, tmp_path / "output")
