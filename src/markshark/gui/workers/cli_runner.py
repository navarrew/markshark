"""
Async CLI command runner.

When running from a normal Python install, uses QProcess to spawn
``python -m markshark.cli <args>``.

When running from a PyInstaller frozen bundle, sys.executable is the
frozen binary (not a Python interpreter), so QProcess won't work.
In that case we run the CLI entry-point directly inside a QThread,
capturing stdout/stderr via stream redirection.
"""

import io
import sys
import traceback
from typing import List

from PySide6.QtCore import QObject, QProcess, QThread, Signal


def _is_frozen() -> bool:
    """Return True when running inside a PyInstaller bundle."""
    return getattr(sys, "frozen", False)


# ── Thread-based runner for frozen builds ──────────────────────────


class _SignalStream(io.TextIOBase):
    """File-like stream that emits a Qt signal on each write.

    Allows the GUI log viewer to receive incremental output instead
    of waiting until the command finishes.
    """

    def __init__(self, signal: Signal):
        super().__init__()
        self._signal = signal

    def write(self, text: str) -> int:
        if text:
            self._signal.emit(text)
        return len(text)

    def flush(self):
        pass


class _CLIWorkerThread(QThread):
    """Run the markshark CLI entry-point in a background thread.

    Captures stdout/stderr and emits them as signals so the UI
    stays responsive and the LogViewer receives live output.
    """

    output_ready = Signal(str)
    error_ready = Signal(str)
    done = Signal(int)  # exit_code

    def __init__(self, args: List[str], parent=None):
        super().__init__(parent)
        self._args = args

    def run(self):  # noqa: D401 – QThread override
        """Execute the CLI command."""
        from markshark.cli import app as typer_app  # lazy import

        # Replace stdout/stderr with signal-emitting streams so
        # the GUI log viewer gets live output line-by-line.
        out_stream = _SignalStream(self.output_ready)
        err_stream = _SignalStream(self.error_ready)
        old_stdout, old_stderr = sys.stdout, sys.stderr

        exit_code = 0
        try:
            sys.stdout = out_stream
            sys.stderr = err_stream

            # Typer's main() raises SystemExit on completion
            try:
                typer_app(self._args, standalone_mode=False)
            except SystemExit as exc:
                exit_code = int(exc.code) if exc.code else 0
            except Exception:
                traceback.print_exc(file=err_stream)
                exit_code = 1
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        self.done.emit(exit_code)


# ── Public API ─────────────────────────────────────────────────────


class CLIRunner(QObject):
    """
    Runs markshark CLI commands asynchronously.

    Automatically selects QProcess (normal install) or QThread
    (PyInstaller frozen bundle) so callers don't need to care.

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

        # QProcess path (non-frozen)
        self._process = QProcess(self)
        self._process.readyReadStandardOutput.connect(self._read_stdout)
        self._process.readyReadStandardError.connect(self._read_stderr)
        self._process.finished.connect(self._on_process_finished)

        # QThread path (frozen)
        self._thread: _CLIWorkerThread | None = None

        self._current_step = ""

    # ── run markshark CLI command ──────────────────────────────────

    def run(self, args: List[str], step_name: str = ""):
        """
        Run a markshark CLI command.

        Args:
            args: Command arguments (e.g., ["align", "input.pdf", ...])
            step_name: Human-readable name for this step
        """
        if self.is_running():
            raise RuntimeError("A process is already running")

        self._current_step = step_name
        self.started.emit(step_name)

        if _is_frozen():
            self._run_in_thread(args)
        else:
            self._run_in_process(args)

    def run_raw(self, program: str, args: List[str], step_name: str = ""):
        """
        Run an arbitrary command (not markshark CLI).

        Always uses QProcess regardless of frozen state.
        """
        if self.is_running():
            raise RuntimeError("A process is already running")

        self._current_step = step_name
        self.started.emit(step_name)
        self._process.start(program, args)

    # ── state queries ──────────────────────────────────────────────

    def is_running(self) -> bool:
        """Check if a process is currently running."""
        process_running = (
            self._process.state() != QProcess.ProcessState.NotRunning
        )
        thread_running = (
            self._thread is not None and self._thread.isRunning()
        )
        return process_running or thread_running

    def kill(self):
        """Kill the running process or thread."""
        if self._process.state() != QProcess.ProcessState.NotRunning:
            self._process.kill()
        if self._thread is not None and self._thread.isRunning():
            self._thread.terminate()

    def current_step(self) -> str:
        """Get the name of the current step."""
        return self._current_step

    # ── QProcess path (normal install) ─────────────────────────────

    def _run_in_process(self, args: List[str]):
        program = sys.executable
        full_args = ["-m", "markshark.cli"] + args
        self._process.start(program, full_args)

    def _read_stdout(self):
        data = (
            self._process.readAllStandardOutput()
            .data()
            .decode("utf-8", errors="replace")
        )
        if data:
            self.output_received.emit(data)

    def _read_stderr(self):
        data = (
            self._process.readAllStandardError()
            .data()
            .decode("utf-8", errors="replace")
        )
        if data:
            self.error_received.emit(data)

    def _on_process_finished(self, exit_code: int, _exit_status):
        self.finished.emit(exit_code, self._current_step)

    # ── QThread path (PyInstaller frozen) ──────────────────────────

    def _run_in_thread(self, args: List[str]):
        self._thread = _CLIWorkerThread(args, parent=self)
        self._thread.output_ready.connect(self.output_received)
        self._thread.error_ready.connect(self.error_received)
        self._thread.done.connect(self._on_thread_finished)
        self._thread.start()

    def _on_thread_finished(self, exit_code: int):
        self.finished.emit(exit_code, self._current_step)
        self._thread = None
