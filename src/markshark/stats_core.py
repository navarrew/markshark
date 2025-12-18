
from typing import Optional
from .tools.stats_tools import run as _run_stats
def compute_stats(results_csv: str, output_csv: str = "results_with_item_stats.csv", item_report_csv: str = "item_analysis.csv",
                  exam_stats_csv: str = "exam_stats.csv", plots_dir: Optional[str] = None, decimals: int = 3,
                  item_pattern: str = r"^Q\d+$", percent: bool = False, label_col: str = "Name", key_row_index: Optional[int] = None,
                  answers_mode: str = "letters", key_label: str = "KEY") -> None:
    _run_stats(input_csv=results_csv, output_csv=output_csv, item_pattern=item_pattern, percent=percent, label_col=label_col,
               exam_stats_csv=exam_stats_csv, plots_dir=plots_dir, key_row_index=key_row_index, answers_mode=answers_mode,
               item_report_csv=item_report_csv, key_label=key_label, decimals=decimals)
