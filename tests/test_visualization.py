"""Tests for fed_playground.src.visualization."""

import matplotlib

matplotlib.use("Agg")  # headless: never open a window

import numpy as np
import pytest
from matplotlib.figure import Figure

from fed_playground.src.visualization import PrivacyUtilityVisualizer


class TestPrivacyUtilityVisualizer:
    def _data(self):
        return {
            "Laplace": {0.2: 74.0, 1.0: 3.0, 5.0: 0.13},
            "Gaussian": {0.2: 40.0, 1.0: 2.0, 5.0: 0.10},
        }

    def test_returns_figure(self):
        viz = PrivacyUtilityVisualizer()
        fig = viz.plot(self._data(), baseline=0.003)
        assert isinstance(fig, Figure)

    def test_saves_png(self, tmp_path):
        viz = PrivacyUtilityVisualizer(save_dir=str(tmp_path))
        viz.plot(self._data(), filename="pu.png")
        assert (tmp_path / "pu.png").exists()

    def test_log_x_axis(self):
        viz = PrivacyUtilityVisualizer()
        fig = viz.plot(self._data())
        assert fig.axes[0].get_xscale() == "log"

    def test_empty_data_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            PrivacyUtilityVisualizer().plot({})

    def test_single_mechanism(self):
        viz = PrivacyUtilityVisualizer()
        fig = viz.plot({"Laplace": {1.0: 3.0, 2.0: 1.0}})
        # One mechanism line; no baseline line drawn.
        assert len(fig.axes[0].lines) == 1
        assert np.isfinite(fig.axes[0].lines[0].get_ydata()).all()
