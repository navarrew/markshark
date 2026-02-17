"""
PDF Tools page — common PDF manipulation utilities.

Four tools in a tabbed layout:
1. Images to PDF — convert a folder of images to a multi-page PDF
2. Merge PDFs — combine multiple PDFs in user-specified order
3. Reorder to Roster — reorder scan pages by student name/ID
4. Interdigitate — interleave front/back PDFs from duplex scanning
"""

import csv
from pathlib import Path

import fitz  # PyMuPDF

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QComboBox,
    QCheckBox,
    QListWidget,
    QListWidgetItem,
    QFileDialog,
    QMessageBox,
    QAbstractItemView,
    QGroupBox,
)

from ..widgets import FileSelector, PageHeader
from ..utils import RUN_BUTTON_STYLE as _RUN_BTN_STYLE

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}


class PdfToolsPage(QWidget):
    """PDF Tools page with tabbed sub-tools."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        header = PageHeader(
            "MarkShark PDF Tools",
            "Common PDF utilities for scan preparation and post-processing.",
        )
        layout.addWidget(header)

        tabs = QTabWidget()
        tabs.addTab(self._build_images_tab(), "Convert Images to PDF")
        tabs.addTab(self._build_merge_tab(), "Combine PDFs")
        tabs.addTab(self._build_reorder_tab(), "Sort/Reorder Pages")
        tabs.addTab(self._build_interdigitate_tab(), "Interdigitate Scans")
        layout.addWidget(tabs, 1)

    # ------------------------------------------------------------------
    # Tab 1: Images to PDF
    # ------------------------------------------------------------------

    def _build_images_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        desc = QLabel(
            "Use this utility to convert a folder of scanned bubblesheet images into a multipage PDF."
            ""
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)


        self.img_folder = FileSelector(
            "Image folder:",
            "",
            "Select folder with JPG/PNG/TIFF images...",
            directory_mode=True,
        )
        layout.addWidget(self.img_folder)

        sort_row = QHBoxLayout()
        sort_row.addWidget(QLabel("Sort PDF pages by:"))
        self.img_sort = QComboBox()
        self.img_sort.addItems(["Image filenames", "Date modified"])
        sort_row.addWidget(self.img_sort)
        sort_row.addStretch()
        layout.addLayout(sort_row)

        self.img_output = FileSelector(
            "Output PDF:",
            "PDF files (*.pdf)",
            "Save combined PDF as...",
            save_mode=True,
        )
        layout.addWidget(self.img_output)

        layout.addStretch()

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.img_status = QLabel("")
        btn_row.addWidget(self.img_status)
        self.img_run_btn = QPushButton("Convert to PDF")
        self.img_run_btn.setStyleSheet(_RUN_BTN_STYLE)
        self.img_run_btn.clicked.connect(self._run_images_to_pdf)
        btn_row.addWidget(self.img_run_btn)
        layout.addLayout(btn_row)

        return tab

    def _run_images_to_pdf(self):
        folder = self.img_folder.path()
        output = self.img_output.path()
        if not folder or not Path(folder).is_dir():
            QMessageBox.warning(self, "Missing Input", "Please select an image folder.")
            return
        if not output:
            QMessageBox.warning(self, "Missing Output", "Please specify an output PDF path.")
            return

        images = [
            f for f in Path(folder).iterdir()
            if f.is_file() and f.suffix.lower() in _IMAGE_EXTS
        ]
        if not images:
            QMessageBox.warning(
                self, "No Images",
                "No JPG/PNG/TIFF images found in the selected folder.",
            )
            return

        if self.img_sort.currentText() == "Date modified":
            images.sort(key=lambda f: f.stat().st_mtime)
        else:
            images.sort(key=lambda f: f.name.lower())

        self.img_status.setText(f"Converting {len(images)} images...")
        self.img_run_btn.setEnabled(False)
        try:
            doc = fitz.open()
            for img_path in images:
                img_doc = fitz.open(str(img_path))
                pdf_bytes = img_doc.convert_to_pdf()
                img_doc.close()
                img_pdf = fitz.open("pdf", pdf_bytes)
                doc.insert_pdf(img_pdf)
                img_pdf.close()
            doc.save(output)
            doc.close()
            self.img_status.setText(f"Done — {len(images)} pages saved.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create PDF:\n{e}")
            self.img_status.setText("Error.")
        finally:
            self.img_run_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Tab 2: Merge PDFs
    # ------------------------------------------------------------------

    def _build_merge_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        layout.addWidget(QLabel("Add PDF files and arrange their order:"))

        list_row = QHBoxLayout()

        self.merge_list = QListWidget()
        self.merge_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.merge_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        list_row.addWidget(self.merge_list, 1)

        btn_col = QVBoxLayout()
        add_btn = QPushButton("Add PDF...")
        add_btn.clicked.connect(self._merge_add)
        btn_col.addWidget(add_btn)

        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._merge_remove)
        btn_col.addWidget(remove_btn)

        up_btn = QPushButton("Move Up")
        up_btn.clicked.connect(self._merge_move_up)
        btn_col.addWidget(up_btn)

        down_btn = QPushButton("Move Down")
        down_btn.clicked.connect(self._merge_move_down)
        btn_col.addWidget(down_btn)

        btn_col.addStretch()
        list_row.addLayout(btn_col)
        layout.addLayout(list_row, 1)

        self.merge_output = FileSelector(
            "Output PDF:",
            "PDF files (*.pdf)",
            "Save merged PDF as...",
            save_mode=True,
        )
        layout.addWidget(self.merge_output)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.merge_status = QLabel("")
        btn_row.addWidget(self.merge_status)
        self.merge_run_btn = QPushButton("Merge")
        self.merge_run_btn.setStyleSheet(_RUN_BTN_STYLE)
        self.merge_run_btn.clicked.connect(self._run_merge)
        btn_row.addWidget(self.merge_run_btn)
        layout.addLayout(btn_row)

        return tab

    def _merge_add(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select PDF Files", "", "PDF files (*.pdf)"
        )
        for p in paths:
            item = QListWidgetItem(Path(p).name)
            item.setData(Qt.ItemDataRole.UserRole, p)
            item.setToolTip(p)
            self.merge_list.addItem(item)

    def _merge_remove(self):
        for item in self.merge_list.selectedItems():
            self.merge_list.takeItem(self.merge_list.row(item))

    def _merge_move_up(self):
        row = self.merge_list.currentRow()
        if row > 0:
            item = self.merge_list.takeItem(row)
            self.merge_list.insertItem(row - 1, item)
            self.merge_list.setCurrentRow(row - 1)

    def _merge_move_down(self):
        row = self.merge_list.currentRow()
        if row < self.merge_list.count() - 1:
            item = self.merge_list.takeItem(row)
            self.merge_list.insertItem(row + 1, item)
            self.merge_list.setCurrentRow(row + 1)

    def _run_merge(self):
        count = self.merge_list.count()
        if count < 2:
            QMessageBox.warning(self, "Not Enough Files", "Add at least 2 PDFs to merge.")
            return
        output = self.merge_output.path()
        if not output:
            QMessageBox.warning(self, "Missing Output", "Please specify an output PDF path.")
            return

        self.merge_status.setText(f"Merging {count} PDFs...")
        self.merge_run_btn.setEnabled(False)
        try:
            merged = fitz.open()
            total_pages = 0
            for i in range(count):
                pdf_path = self.merge_list.item(i).data(Qt.ItemDataRole.UserRole)
                src = fitz.open(pdf_path)
                merged.insert_pdf(src)
                total_pages += len(src)
                src.close()
            merged.save(output)
            merged.close()
            self.merge_status.setText(f"Done — {total_pages} pages merged.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to merge PDFs:\n{e}")
            self.merge_status.setText("Error.")
        finally:
            self.merge_run_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Tab 3: Sort/Reorder Pages
    # ------------------------------------------------------------------

    def _build_reorder_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        desc = QLabel(
            "Sort your student scans PDF pages alphabetically or "
            "by Student ID number."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        self.reorder_pdf = FileSelector(
            "Scans PDF:",
            "PDF files (*.pdf)",
            "Select the scanned/aligned PDF...",
        )
        layout.addWidget(self.reorder_pdf)

        self.reorder_csv = FileSelector(
            "Results CSV:",
            "CSV files (*.csv)",
            "Select results.csv with Page/StudentID columns...",
        )
        layout.addWidget(self.reorder_csv)

        self.reorder_roster = FileSelector(
            "Roster (optional):",
            "CSV files (*.csv)",
            "Optional roster for name lookup...",
        )
        layout.addWidget(self.reorder_roster)

        sort_row = QHBoxLayout()
        sort_row.addWidget(QLabel("Sort by:"))
        self.reorder_sort = QComboBox()
        self.reorder_sort.addItems(["Last Name, First Name", "Student ID"])
        sort_row.addWidget(self.reorder_sort)
        sort_row.addStretch()
        layout.addLayout(sort_row)

        self.reorder_output = FileSelector(
            "Output PDF:",
            "PDF files (*.pdf)",
            "Save reordered PDF as...",
            save_mode=True,
        )
        layout.addWidget(self.reorder_output)

        layout.addStretch()

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.reorder_status = QLabel("")
        btn_row.addWidget(self.reorder_status)
        self.reorder_run_btn = QPushButton("Reorder")
        self.reorder_run_btn.setStyleSheet(_RUN_BTN_STYLE)
        self.reorder_run_btn.clicked.connect(self._run_reorder)
        btn_row.addWidget(self.reorder_run_btn)
        layout.addLayout(btn_row)

        return tab

    def _run_reorder(self):
        pdf_path = self.reorder_pdf.path()
        csv_path = self.reorder_csv.path()
        output = self.reorder_output.path()

        if not pdf_path or not Path(pdf_path).is_file():
            QMessageBox.warning(self, "Missing Input", "Please select a scans PDF.")
            return
        if not csv_path or not Path(csv_path).is_file():
            QMessageBox.warning(self, "Missing Input", "Please select a results CSV.")
            return
        if not output:
            QMessageBox.warning(self, "Missing Output", "Please specify an output PDF path.")
            return

        # Parse results CSV
        try:
            entries = self._parse_results_csv(csv_path)
        except Exception as e:
            QMessageBox.critical(self, "CSV Error", f"Failed to parse results CSV:\n{e}")
            return

        if not entries:
            QMessageBox.warning(self, "No Data", "No student rows found in the CSV.")
            return

        # Optionally enrich with roster names
        roster_path = self.reorder_roster.path()
        if roster_path and Path(roster_path).is_file():
            try:
                roster = self._load_roster_dict(roster_path)
                for entry in entries:
                    sid = entry.get("student_id", "")
                    if sid in roster:
                        if not entry.get("last_name"):
                            entry["last_name"] = roster[sid].get("last_name", "")
                        if not entry.get("first_name"):
                            entry["first_name"] = roster[sid].get("first_name", "")
            except Exception:
                pass  # roster is optional — don't fail

        # Sort
        sort_by = self.reorder_sort.currentText()
        if sort_by == "Student ID":
            entries.sort(key=lambda e: e.get("student_id", ""))
        else:
            entries.sort(key=lambda e: (
                e.get("last_name", "").lower(),
                e.get("first_name", "").lower(),
            ))

        self.reorder_status.setText(f"Reordering {len(entries)} pages...")
        self.reorder_run_btn.setEnabled(False)
        try:
            src = fitz.open(pdf_path)
            out = fitz.open()
            for entry in entries:
                pg = entry["page_0idx"]
                if 0 <= pg < len(src):
                    out.insert_pdf(src, from_page=pg, to_page=pg)
            out.save(output)
            out.close()
            src.close()
            self.reorder_status.setText(f"Done — {len(entries)} pages reordered.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to reorder PDF:\n{e}")
            self.reorder_status.setText("Error.")
        finally:
            self.reorder_run_btn.setEnabled(True)

    @staticmethod
    def _parse_results_csv(csv_path: str) -> list:
        """Parse results.csv and return list of dicts with page/name/id info."""
        entries = []
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Find columns case-insensitively
                page_str = ""
                sid = ""
                last = ""
                first = ""
                for k, v in row.items():
                    kl = k.strip().lower()
                    if kl == "page":
                        page_str = v.strip()
                    elif kl in ("studentid", "student_id", "sid", "id"):
                        sid = v.strip()
                    elif kl in ("lastname", "last_name", "last", "surname"):
                        last = v.strip()
                    elif kl in ("firstname", "first_name", "first"):
                        first = v.strip()

                # Skip KEY rows and empty pages
                if not page_str or page_str == "0":
                    continue
                if sid.upper() == "KEY":
                    continue
                try:
                    page_num = int(page_str)
                except ValueError:
                    continue

                entries.append({
                    "page_0idx": page_num - 1,  # results.csv is 1-based
                    "student_id": sid,
                    "last_name": last,
                    "first_name": first,
                })
        return entries

    @staticmethod
    def _load_roster_dict(roster_path: str) -> dict:
        """Load roster CSV into {student_id: {last_name, first_name}} dict."""
        roster = {}
        with open(roster_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = ""
                last = ""
                first = ""
                for k, v in row.items():
                    kl = k.strip().lower()
                    if kl in ("studentid", "student_id", "sid", "id"):
                        sid = v.strip()
                    elif kl in ("lastname", "last_name", "last", "surname"):
                        last = v.strip()
                    elif kl in ("firstname", "first_name", "first"):
                        first = v.strip()
                if sid:
                    roster[sid] = {"last_name": last, "first_name": first}
        return roster

    # ------------------------------------------------------------------
    # Tab 4: Interdigitate
    # ------------------------------------------------------------------

    def _build_interdigitate_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        desc = QLabel(
            "Interleave two PDFs — useful when front and back sides "
            "are scanned as separate stacks."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        self.inter_fronts = FileSelector(
            "Front pages PDF:",
            "PDF files (*.pdf)",
            "Select PDF with front sides...",
        )
        layout.addWidget(self.inter_fronts)

        self.inter_backs = FileSelector(
            "Back pages PDF:",
            "PDF files (*.pdf)",
            "Select PDF with back sides...",
        )
        layout.addWidget(self.inter_backs)

        self.inter_reverse = QCheckBox("Reverse back pages order")
        self.inter_reverse.setChecked(True)
        self.inter_reverse.setToolTip(
            "Check this if the back-side stack was scanned in reverse order "
            "(common with document feeders)."
        )
        layout.addWidget(self.inter_reverse)

        self.inter_output = FileSelector(
            "Output PDF:",
            "PDF files (*.pdf)",
            "Save interleaved PDF as...",
            save_mode=True,
        )
        layout.addWidget(self.inter_output)

        layout.addStretch()

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.inter_status = QLabel("")
        btn_row.addWidget(self.inter_status)
        self.inter_run_btn = QPushButton("Interdigitate")
        self.inter_run_btn.setStyleSheet(_RUN_BTN_STYLE)
        self.inter_run_btn.clicked.connect(self._run_interdigitate)
        btn_row.addWidget(self.inter_run_btn)
        layout.addLayout(btn_row)

        return tab

    def _run_interdigitate(self):
        fronts_path = self.inter_fronts.path()
        backs_path = self.inter_backs.path()
        output = self.inter_output.path()

        if not fronts_path or not Path(fronts_path).is_file():
            QMessageBox.warning(self, "Missing Input", "Please select the front pages PDF.")
            return
        if not backs_path or not Path(backs_path).is_file():
            QMessageBox.warning(self, "Missing Input", "Please select the back pages PDF.")
            return
        if not output:
            QMessageBox.warning(self, "Missing Output", "Please specify an output PDF path.")
            return

        self.inter_status.setText("Interleaving...")
        self.inter_run_btn.setEnabled(False)
        try:
            fronts = fitz.open(fronts_path)
            backs = fitz.open(backs_path)
            reverse = self.inter_reverse.isChecked()

            if len(fronts) != len(backs):
                reply = QMessageBox.question(
                    self,
                    "Page Count Mismatch",
                    f"Front PDF has {len(fronts)} pages, back PDF has {len(backs)} pages.\n\n"
                    "Continue anyway? (Extra pages from the longer PDF will be appended at the end.)",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    self.inter_status.setText("Cancelled.")
                    return

            out = fitz.open()
            n = max(len(fronts), len(backs))
            for i in range(n):
                if i < len(fronts):
                    out.insert_pdf(fronts, from_page=i, to_page=i)
                if i < len(backs):
                    bi = (len(backs) - 1 - i) if reverse else i
                    out.insert_pdf(backs, from_page=bi, to_page=bi)

            out.save(output)
            out.close()
            fronts.close()
            backs.close()
            total = len(out) if not out.is_closed else n * 2
            self.inter_status.setText(f"Done — {n} pairs interleaved.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to interdigitate:\n{e}")
            self.inter_status.setText("Error.")
        finally:
            self.inter_run_btn.setEnabled(True)
