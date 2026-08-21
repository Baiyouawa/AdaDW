import numpy as np

from adawd_preexp.profiler import ProfilerConfig, profile_segments, profile_window


def test_scores_are_bounded_and_change_raises_u():
    length = 96
    time = np.arange(length)
    stable = np.sin(2 * np.pi * time / 24.0)
    changed = stable.copy()
    changed[length // 2 :] += 3.0
    config = ProfilerConfig(window_size=length, max_windows_per_segment=None)

    stable_record = profile_window(stable[:, None], config)[0]
    changed_record = profile_window(changed[:, None], config)[0]

    for record in (stable_record, changed_record):
        for key in ("u_change", "u_spectral", "u_surprise", "m_peak", "m_band", "m_channel", "U", "M"):
            assert 0.0 <= record[key] <= 1.0
    assert changed_record["U"] > stable_record["U"]


def test_multifrequency_signal_raises_m():
    length = 96
    time = np.arange(length)
    single = np.sin(2 * np.pi * time / 24.0)
    multiple = single + 0.8 * np.sin(2 * np.pi * time / 12.0) + 0.6 * np.sin(2 * np.pi * time / 6.0)
    config = ProfilerConfig(window_size=length, max_windows_per_segment=None)

    single_record = profile_window(single[:, None], config)[0]
    multiple_record = profile_window(multiple[:, None], config)[0]
    assert multiple_record["M"] > single_record["M"]


def test_windows_do_not_cross_segments():
    config = ProfilerConfig(window_size=32, stride=16, max_windows_per_segment=None)
    segments = [("first", np.ones((64, 2))), ("second", np.ones((48, 2)))]
    frame = profile_segments("demo", segments, config)
    assert frame[["segment", "window_start"]].drop_duplicates().shape[0] == 5
    assert set(frame["segment"]) == {"first", "second"}


def test_channel_subsampling_preserves_original_ids():
    config = ProfilerConfig(window_size=32)
    values = np.random.default_rng(0).normal(size=(32, 4))
    records = profile_window(values, config, channels=[1, 3])
    assert [record["channel"] for record in records] == [1, 3]
    assert all(record["channel_effective_rank"] >= 1.0 for record in records)
