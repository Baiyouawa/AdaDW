from types import SimpleNamespace

import numpy as np
import torch

from basicts.runners import BasicTSRunner


def test_saved_results_use_a_running_offset_for_short_final_batch(tmp_path):
    runner = SimpleNamespace(
        ckpt_save_dir=str(tmp_path),
        test_data_loader=SimpleNamespace(dataset=range(5)),
    )
    first = {
        name: torch.arange(3, dtype=torch.float32).reshape(3, 1, 1)
        for name in ("inputs", "prediction", "targets")
    }
    second = {
        name: torch.arange(3, 5, dtype=torch.float32).reshape(2, 1, 1)
        for name in ("inputs", "prediction", "targets")
    }

    BasicTSRunner._save_results(runner, 0, first)
    BasicTSRunner._save_results(runner, 1, second)

    for name in ("inputs", "prediction", "targets"):
        saved = np.load(tmp_path / "test_results" / f"{name}.npy")
        assert saved[:, 0, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
