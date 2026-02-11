"""
Corrections log for MarkShark GUI.

Implements an append-only corrections log that tracks changes made during
the review process. Corrections are stored alongside the scored results
and can be applied to regenerate final grades.

Design principles:
- Original scored data is never modified
- Corrections are append-only (new entries, including reverts)
- Full history is preserved for audit trail
- Corrections can be applied to produce final output
"""

import csv
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Iterator


@dataclass
class Correction:
    """A single correction entry."""
    timestamp: str  # ISO format
    correction_type: str  # ANSWER, STUDENT_ID, NAME, REVERT
    student_id: str
    field: str  # Question number (Q1, Q15) or field name (student_id, name)
    original_value: str
    corrected_value: str
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Correction":
        return cls(**data)

    def is_revert(self) -> bool:
        """Check if this is a revert entry."""
        return self.correction_type == "REVERT"


class CorrectionLog:
    """
    Append-only corrections log for scored results.

    File format (CSV):
        # applies to: /path/to/results.csv
        timestamp,type,student_id,field,original,corrected,reason
        2024-01-15T10:23:00,ANSWER,1003,Q15,B,C,misread - clearly C
        2024-01-15T10:24:30,ANSWER,1003,Q22,--,A,no mark detected
        2024-01-15T10:30:00,STUDENT_ID,1004,student_id,1004,1040,transposed
        2024-01-16T09:00:00,REVERT,1003,Q15,,,original was correct

    Usage:
        log = CorrectionLog(Path("corrections.csv"), "/path/to/results.csv")
        log.add_answer_correction("1003", "Q15", "B", "C", "misread")
        log.revert("1003", "Q15", "original was correct")

        # Apply corrections to get final values
        final = log.apply_corrections(original_data)
    """

    FIELD_NAMES = ["timestamp", "type", "student_id", "field", "original", "corrected", "reason"]
    HEADER_PREFIX = "# applies to: "

    def __init__(self, path: Path, scoring_run_id: str):
        """
        Initialize correction log.

        Args:
            path: Path to corrections CSV file
            scoring_run_id: Path to the results CSV this log applies to
        """
        self.path = Path(path)
        self.scoring_run_id = scoring_run_id
        self._corrections: Optional[List[Correction]] = None

    @property
    def corrections(self) -> List[Correction]:
        """Get all corrections, loading from disk if needed."""
        if self._corrections is None:
            self._corrections = self._load()
        return self._corrections

    def exists(self) -> bool:
        """Check if corrections file exists."""
        return self.path.exists()

    def _load(self) -> List[Correction]:
        """Load corrections from CSV file."""
        if not self.path.exists():
            return []

        corrections = []
        try:
            with open(self.path, "r", newline="", encoding="utf-8") as f:
                # Skip header comment (accept both old and new prefix)
                first_line = f.readline()
                if not (first_line.startswith(self.HEADER_PREFIX)
                        or first_line.startswith("# applies_to: ")):
                    # No header, rewind
                    f.seek(0)

                reader = csv.DictReader(f)

                # Validate that this looks like a corrections file
                if reader.fieldnames is None:
                    return []

                # Check for required columns
                required = {"timestamp", "type", "student_id", "field"}
                if not required.issubset(set(reader.fieldnames)):
                    # This is not a valid corrections file
                    return []

                for row in reader:
                    # Skip rows with missing required fields
                    if not row.get("timestamp") or not row.get("student_id"):
                        continue

                    corrections.append(Correction(
                        timestamp=row.get("timestamp", ""),
                        correction_type=row.get("type", ""),
                        student_id=row.get("student_id", ""),
                        field=row.get("field", ""),
                        original_value=row.get("original", ""),
                        corrected_value=row.get("corrected", ""),
                        reason=row.get("reason", "")
                    ))
        except Exception:
            # If anything goes wrong reading the file, return empty
            return []

        return corrections

    def _save_header(self):
        """Write header to new file."""
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            f.write(f"{self.HEADER_PREFIX}{self.scoring_run_id}\n")
            writer = csv.writer(f)
            writer.writerow(["timestamp", "type", "student_id", "field", "original", "corrected", "reason"])

    def _append(self, correction: Correction):
        """Append a correction to the log."""
        # Create file with header if it doesn't exist
        if not self.path.exists():
            self._save_header()

        # Append the correction
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                correction.timestamp,
                correction.correction_type,
                correction.student_id,
                correction.field,
                correction.original_value,
                correction.corrected_value,
                correction.reason
            ])

        # Update in-memory cache
        if self._corrections is not None:
            self._corrections.append(correction)

    def reload(self):
        """Reload corrections from disk."""
        self._corrections = self._load()

    # -------------------------------------------------------------------------
    # Adding Corrections
    # -------------------------------------------------------------------------

    def add_answer_correction(
        self,
        student_id: str,
        question: str,
        original: str,
        corrected: str,
        reason: str = ""
    ) -> Correction:
        """
        Add a correction to a student's answer.

        Args:
            student_id: The student ID
            question: Question identifier (e.g., "Q15", "Q1")
            original: Original detected value
            corrected: Corrected value
            reason: Optional reason for correction

        Returns:
            The created Correction object
        """
        correction = Correction(
            timestamp=datetime.now().isoformat(),
            correction_type="ANSWER",
            student_id=student_id,
            field=question,
            original_value=original,
            corrected_value=corrected,
            reason=reason
        )
        self._append(correction)
        return correction

    def add_student_id_correction(
        self,
        original_id: str,
        corrected_id: str,
        reason: str = ""
    ) -> Correction:
        """
        Correct a student's ID.

        Args:
            original_id: Original detected ID
            corrected_id: Corrected ID
            reason: Optional reason for correction

        Returns:
            The created Correction object
        """
        correction = Correction(
            timestamp=datetime.now().isoformat(),
            correction_type="STUDENT_ID",
            student_id=original_id,
            field="student_id",
            original_value=original_id,
            corrected_value=corrected_id,
            reason=reason
        )
        self._append(correction)
        return correction

    def add_name_correction(
        self,
        student_id: str,
        original_name: str,
        corrected_name: str,
        reason: str = ""
    ) -> Correction:
        """
        Correct a student's name.

        Args:
            student_id: The student ID
            original_name: Original detected name
            corrected_name: Corrected name
            reason: Optional reason for correction

        Returns:
            The created Correction object
        """
        correction = Correction(
            timestamp=datetime.now().isoformat(),
            correction_type="NAME",
            student_id=student_id,
            field="name",
            original_value=original_name,
            corrected_value=corrected_name,
            reason=reason
        )
        self._append(correction)
        return correction

    def revert(
        self,
        student_id: str,
        field: str,
        reason: str = ""
    ) -> Correction:
        """
        Revert a previous correction by adding a REVERT entry.

        This effectively removes the correction for this field.

        Args:
            student_id: The student ID
            field: Field to revert (e.g., "Q15", "student_id", "name")
            reason: Optional reason for reverting

        Returns:
            The created Correction object
        """
        correction = Correction(
            timestamp=datetime.now().isoformat(),
            correction_type="REVERT",
            student_id=student_id,
            field=field,
            original_value="",
            corrected_value="",
            reason=reason
        )
        self._append(correction)
        return correction

    # -------------------------------------------------------------------------
    # Querying Corrections
    # -------------------------------------------------------------------------

    def get_corrections_for_student(self, student_id: str) -> List[Correction]:
        """Get all corrections for a specific student."""
        return [c for c in self.corrections if c.student_id == student_id]

    def get_effective_corrections(self) -> Dict[str, Dict[str, str]]:
        """
        Get the effective corrections after applying all entries in order.

        REVERT entries remove previous corrections.

        Returns:
            Dict mapping student_id -> {field: corrected_value}
        """
        effective: Dict[str, Dict[str, str]] = {}

        for correction in self.corrections:
            student_id = correction.student_id

            if student_id not in effective:
                effective[student_id] = {}

            if correction.is_revert():
                # Remove the correction for this field
                effective[student_id].pop(correction.field, None)
            else:
                # Add/update the correction
                effective[student_id][correction.field] = correction.corrected_value

        # Clean up empty student entries
        return {k: v for k, v in effective.items() if v}

    def get_effective_value(
        self,
        student_id: str,
        field: str,
        original_value: str
    ) -> str:
        """
        Get the effective value for a field after corrections.

        Args:
            student_id: The student ID
            field: Field name (e.g., "Q15", "student_id")
            original_value: The original value from scored data

        Returns:
            The corrected value if a correction exists, otherwise original
        """
        effective = self.get_effective_corrections()
        student_corrections = effective.get(student_id, {})
        return student_corrections.get(field, original_value)

    def has_correction(self, student_id: str, field: str) -> bool:
        """Check if a field has an active correction."""
        effective = self.get_effective_corrections()
        return field in effective.get(student_id, {})

    def count(self) -> int:
        """Get total number of correction entries."""
        return len(self.corrections)

    def effective_count(self) -> int:
        """Get number of effective corrections (after reverts)."""
        effective = self.get_effective_corrections()
        return sum(len(fields) for fields in effective.values())

    # -------------------------------------------------------------------------
    # Applying Corrections
    # -------------------------------------------------------------------------

    def apply_to_row(
        self,
        row: Dict[str, str],
        student_id_field: str = "student_id"
    ) -> Dict[str, str]:
        """
        Apply corrections to a single data row.

        Args:
            row: Dictionary representing a row from scored CSV
            student_id_field: Name of the student ID field in the row

        Returns:
            New dictionary with corrections applied
        """
        result = dict(row)
        student_id = row.get(student_id_field, "")

        if not student_id:
            return result

        effective = self.get_effective_corrections()
        student_corrections = effective.get(student_id, {})

        # Apply student ID correction (special case - changes the ID itself)
        if "student_id" in student_corrections:
            result[student_id_field] = student_corrections["student_id"]

        # Apply other corrections
        for field, corrected_value in student_corrections.items():
            if field in result and field != "student_id":
                result[field] = corrected_value

        return result

    def apply_to_data(
        self,
        rows: List[Dict[str, str]],
        student_id_field: str = "student_id"
    ) -> List[Dict[str, str]]:
        """
        Apply corrections to all rows.

        Args:
            rows: List of dictionaries from scored CSV
            student_id_field: Name of the student ID field

        Returns:
            New list with corrections applied
        """
        return [self.apply_to_row(row, student_id_field) for row in rows]

    # -------------------------------------------------------------------------
    # Copying Corrections
    # -------------------------------------------------------------------------

    def copy_applicable_corrections(
        self,
        target_log: "CorrectionLog",
        filter_types: Optional[List[str]] = None
    ) -> int:
        """
        Copy corrections that can apply to new scoring results.

        Useful when re-scoring with a different key but same scans.
        Answer and ID corrections can carry over; score corrections cannot.

        Args:
            target_log: The target CorrectionLog to copy to
            filter_types: Types to copy (default: ANSWER, STUDENT_ID, NAME)

        Returns:
            Number of corrections copied
        """
        if filter_types is None:
            filter_types = ["ANSWER", "STUDENT_ID", "NAME"]

        count = 0
        for correction in self.corrections:
            if correction.correction_type in filter_types:
                # Re-create the correction in the target log
                new_correction = Correction(
                    timestamp=datetime.now().isoformat(),
                    correction_type=correction.correction_type,
                    student_id=correction.student_id,
                    field=correction.field,
                    original_value=correction.original_value,
                    corrected_value=correction.corrected_value,
                    reason=f"Copied from {self.scoring_run_id}: {correction.reason}"
                )
                target_log._append(new_correction)
                count += 1

        return count

    # -------------------------------------------------------------------------
    # History & Debugging
    # -------------------------------------------------------------------------

    def get_history_for_field(
        self,
        student_id: str,
        field: str
    ) -> List[Correction]:
        """Get full correction history for a specific field."""
        return [
            c for c in self.corrections
            if c.student_id == student_id and c.field == field
        ]

    def clear(self):
        """
        Clear all corrections (deletes the file).

        Use with caution - this is destructive!
        """
        if self.path.exists():
            self.path.unlink()
        self._corrections = []

    def __iter__(self) -> Iterator[Correction]:
        """Iterate over all corrections."""
        return iter(self.corrections)

    def __len__(self) -> int:
        """Get total number of corrections."""
        return len(self.corrections)
