"""
Async CLI command runner using QProcess.

Runs markshark CLI commands without blocking the UI.
"""

import sys
from typing import List, Optional

from PySide6.QtCore import QObject, QProcess, Signal


class CLIRunner(QObject):
    """
    Runs markshark CLI commands asynchronously.

    Signals:
        output_received: Emitted when stdout data is available
        error_received: Emitted when stderr data is available
        finished: Emitted when process completes (exit_code, step_name)
        started: Emitted when process starts
    """

    output_received = Signal(str)
    error_received = Signal(str)
    finished = Signal(int, str)  # exit_code, step_name
    started = Signal(str)  # step_name

    def __init__(self, parent=None):
        super().__init__(parent)
        self._process = QProcess(self)
        self._process.readyReadStandardOutput.connect(self._read_stdout)
        self._process.readyReadStandardError.connect(self._read_stderr)
        self._process.finished.connect(self._on_finished)
        self._current_step = ""

    def run(self, args: List[str], step_name: str = ""):
        """
        Run a markshark CLI command.

        Args:
            args: Command arguments (e.g., ["align", "input.pdf", "--template", "t.pdf"])
            step_name: Human-readable name for this step (e.g., "Aligning scans")
        """
        if self.is_running():
            raise RuntimeError("A process is already running")

        self._current_step = step_name
        program = sys.executable
        full_args = ["-m", "markshark.cli"] + args

        self.started.emit(step_name)
        self._process.start(program, full_args)

    def run_raw(self, program: str, args: List[str], step_name: str = ""):
        """
        Run an arbitrary command (not markshark CLI).

        Args:
            program: The program to run
            args: Command arguments
            step_name: Human-readable name for this step
        """
        if self.is_running():
            raise RuntimeError("A process is already running")

        self._current_step = step_name
        self.started.emit(step_name)
        self._process.start(program, args)

    def is_running(self) -> bool:
        """Check if a process is currently running."""
        return self._process.state() != QProcess.ProcessState.NotRunning

    def kill(self):
        """Kill the running process."""
        if self.is_running():
            self._process.kill()

    def current_step(self) -> str:
        """Get the name of the current step."""
        return self._current_step

    def _read_stdout(self):
        """Read and emit stdout data."""
        data = self._process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        if data:
            self.output_received.emit(data)

    def _read_stderr(self):
        """Read and emit stderr data."""
        data = self._process.readAllStandardError().data().decode("utf-8", errors="replace")
        if data:
            self.error_received.emit(data)

    def _on_finished(self, exit_code: int, _exit_status):
        """Handle process completion."""
        self.finished.emit(exit_code, self._current_step)
