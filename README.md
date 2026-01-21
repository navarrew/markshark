# MarkSharkOMR - in development
## A fast, accurate, customizable, open-source test bubble sheet scanner

MarkSharkOMR is a versatile and fast tool to **grade, and analyze your own bubble-sheet exams**.

![MarkShark Logo](images/shark.png)

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

---

![OMR pipeline](images/processmap.png)

---

## Prerequisites

- **Python 3.9–3.12**

---

## Installation

Clone and install the package locally:

```bash
git clone https://github.com/navarrew/markshark.git
cd markshark
pip install -e .
```

Make sure you install required dependencies (**OpenCV**, **Typer**, **Streamlit**).

---

# Using MarkShark

MarkShark is in development.  It currently works well after being tested in a real-world scenario of a class of 270 students with two midterms and a final (a total of over 700 scans).  You can go from 500 scanned student bubblesheets to a final report in less than five minutes. 

We provide a manual as a pdf in the main github directory.  Help is also available from the command line.  For the time being documentation is being kept current in the pdf manual and not on this markdown page. 

MarkShark works with a variety of bubblesheet formats (you can easily generate your own custom bubblesheets) and it rapidly generates reports about student performance and the quality of the test questions.  We provide template bubblesheets and necessary mapping files for each that you can customize for your own use. 

It accepts multiple versions of the same test (you provide a single key file that has the correct answsers for all versions).  

It can flag issues like unfilled buubles, rows where more than one bubble was filled in.

If you provide MarkShark with your class roster can merge the data effortlessly with your class roster and paste it back into your course management system (Blackboard, Canvas, etc).  It can tell you who was missing from the test (absent) and flag 'orphan' scans (where the student didn't fill in their information properly.

It can generate fake student data for you to test with including fake filled in bubblesheets that you can print out, rescan, and test for yourself before committing yourself to using MarkShark.

To use the graphical interface instead of the command line - from the command line type:
```bash
markshark-gui
```

---

## License

**MarkShark — the open-source bubble hunter**  
Copyright © 2026 William Navarre, University of Toronto  
Licensed under the **GNU Affero General Public License v3 (AGPL-3.0)**.

You may use, modify, and redistribute this software for **academic, research, and open-source** purposes, provided derivative works remain open-source under the same license.

**Commercial or institutional use** (e.g., SaaS platforms, proprietary educational tools, or for-profit distribution) requires a separate license.

For licensing inquiries, contact [william.navarre@utoronto.ca](mailto:william.navarre@utoronto.ca).
