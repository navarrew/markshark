"""
Re-annotation worker thread.

Calls score_pdf() with teacher corrections applied, regenerating
both the annotated PDF and the scored CSV with updated scores.
"""

from typing import Optional

from PySide6.QtCore import QThread, Signal


class ReAnnotateWorker(QThread):
    """Worker thread that re-runs score_pdf() with corrections.

    Parameters are loaded from the original results_params.json so the
    re-annotation uses the same template, thresholds, and calibration
    settings as the original scoring run.
    """

    finished = Signal(str)   # emits out_csv path on success
    error = Signal(str)      # emits error message on failure

    def __init__(
        self,
        input_path: str,
        bubblemap_path: str,
        out_csv: str,
        out_pdf: str,
        key_txt: Optional[str],
        roster_csv: Optional[str],
        corrections: dict,
        params: dict,
        parent=None,
    ):
        super().__init__(parent)
        self._input_path = input_path
        self._bubblemap_path = bubblemap_path
        self._out_csv = out_csv
        self._out_pdf = out_pdf
        self._key_txt = key_txt
        self._roster_csv = roster_csv
        self._corrections = corrections
        self._params = params

    def run(self):  # noqa: D401 – QThread override
        """Execute score_pdf() in a background thread."""
        try:
            from markshark.score_core import score_pdf
            from markshark.defaults import SCORING_DEFAULTS

            p = self._params

            # min_fill is stored as 0-100 in params; score_pdf() expects 0-1 fraction
            min_fill_pct = p.get("min_fill", SCORING_DEFAULTS.min_fill)
            min_fill = min_fill_pct / 100.0

            fixed_thresh = p.get("fixed_thresh")
            if fixed_thresh == "auto":
                fixed_thresh = None

            score_pdf(
                input_path=self._input_path,
                bublmap_path=self._bubblemap_path,
                out_csv=self._out_csv,
                min_fill=min_fill,
                fixed_thresh=fixed_thresh,
                key_txt=self._key_txt,
                out_pdf=self._out_pdf,
                dpi=p.get("dpi", 150),
                auto_calibrate_thresh=p.get("auto_calibrate_thresh", True),
                calibrate_background=p.get("calibrate_background", True),
                background_percentile=p.get("background_percentile", 10.0),
                adaptive_rescoring=p.get("adaptive_rescoring", True),
                adaptive_max_adjustment=p.get("adaptive_max_adjustment", 40),
                adaptive_min_above_floor=p.get("adaptive_min_above_floor", 30),
                roster_csv=self._roster_csv,
                corrections=self._corrections,
            )
            self.finished.emit(self._out_csv)

        except Exception as e:
            self.error.emit(str(e))
