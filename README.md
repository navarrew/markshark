# MarkShark
## A fast, accurate, customizable, open-source test bubble sheet scanner

![MarkShark Logo](images/shark.png)

MarkSharkOMR is a versatile and fast tool to **grade, and analyze your own bubble-sheet exams**.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![PyPI version](https://img.shields.io/pypi/v/markshark)](https://pypi.org/project/markshark/)

---

## Prerequisites

- **Python 3.9–3.12**

---

## Installation

```bash
pip install markshark
```

## Dependencies (installs automatically with pip)
- typer>=0.12
- rich>=13.7
- opencv-contrib-python>=4.9
- numpy>=1.26,<2.0
- pymupdf>=1.24
- pdf2image>=1.17
- pandas>=2.1,<2.3
- matplotlib>=3.8
- streamlit>=1.35
- pyyaml
- openpyxl>=3.0
- rapidfuzz>=3.0
- streamlit-antd-components>=0.3.2


---

# About MarkShark

MarkShark is in development.  It currently works well after being tested in a real-world scenario of a class of 270 students with two midterms and a final (a total of over 700 scans).  You can go from 500 scanned student bubblesheets to a final report in less than five minutes. 

We provide a manual as a pdf in the main github directory.  Help is also available from the command line.  For the time being documentation is being kept current in the pdf manual and not on this markdown page. 

MarkShark works with a variety of bubblesheet formats (you can easily generate your own custom bubblesheets) and it rapidly generates reports about student performance and the quality of the test questions.  We provide template bubblesheets and necessary mapping files for each that you can customize for your own use. 

It accepts multiple versions of the same test (you provide a single key file that has the correct answsers for all versions).  

It can flag issues like unfilled buubles, rows where more than one bubble was filled in.

If you provide MarkShark with your class roster it can tell you who was missing from the test (absent) and flag 'orphan' scans (where the student didn't fill in their information properly.  It provides student scores in a format that is easily pasted into spreadsheets and into LMSs like Blackboard and Canvas.

For testing prior to use MarkShark can generate fake student data ncluding fake filled in bubblesheets that you can print out, rescan, and test for yourself before committing yourself to using MarkShark.

# What you provide
If you want to use a bubblesheet you've been using in the past you will need to generate a map (bubblemap.yaml) file that tells MarkShark where the bubbles are and what the bubbles represent.  You need to provide MarkShark with a pdf of the blank bubblesheet (the master template) and its corresponding bubblemap one time.  You can save these templates into MarkShark and reuse them again and again with a single click.

Then for each test you simply upload your scanned student sheets as a pdf. If you have folder of jpg or png images we have a utitlity that quickly connverts them to a pdf.  A key is optional but required for getting student scores (percent correct, etc).  If you also provide the class roster, MarkShark identify students who were absent and give you back scores in a format you can easily upload back to your learning management software (LMS like Blackboard, Canvas, Moodle, Sakai, Open edX).

## License

**MarkShark — the open-source bubble hunter**  
Copyright © 2026 William Navarre, University of Toronto  
Licensed under the **GNU Affero General Public License v3 (AGPL-3.0)**.

You may use, modify, and redistribute this software for **academic, research, and open-source** purposes, provided derivative works remain open-source under the same license.

**Commercial or institutional use** (e.g., SaaS platforms, proprietary educational tools, or for-profit distribution) requires a separate license.

For licensing inquiries, contact [william.navarre@utoronto.ca](mailto:william.navarre@utoronto.ca).
