import argparse
import sys

from pre_experiments import run_preexperiments
from pre_experiments.run_preexperiments import capacity_command, prepare_command


def test_prepare_command_is_strict():
    assert prepare_command("ETTh1") == [
        sys.executable,
        "pre_experiments/prepare_dataset.py",
        "--dataset",
        "ETTh1",
        "--strict-shape",
    ]


def test_capacity_command_builds_training_sweep():
    command = capacity_command("depth", "ETTh1", "PatchTST", 96, [42, 43], "0", False)

    assert command[-3:] == ["--gpu", "0", "--all"]
    assert command[command.index("--seeds") + 1 : -3] == ["42", "43"]
    assert ["--axis", "depth"] == command[command.index("--axis") : command.index("--axis") + 2]


def test_capacity_command_builds_cpu_dry_run():
    command = capacity_command("width", "ILI", "TimesNet", 24, [42], None, True)

    assert command[-1] == "--dry-run"
    assert "--gpu" not in command
    assert "--all" not in command


def test_dry_run_does_not_prepare_dataset(monkeypatch):
    commands = []
    monkeypatch.setattr(run_preexperiments, "run_command", commands.append)
    args = argparse.Namespace(
        stage="depth",
        dataset="ETTh1",
        model="PatchTST",
        horizon=96,
        seeds=[42],
        gpu="0",
        cpu=False,
        dry_run=True,
        skip_prepare=False,
    )

    run_preexperiments.run_capacity_stage(args)

    assert len(commands) == 1
    assert "--dry-run" in commands[0]
