MarkShark Sample Data
=====================

This folder contains sample mock exam data for testing MarkShark.

Files:
------
- mock_scans.pdf      - 10 synthetic student answer sheets (64 questions each)
- mock_answer_key.txt - Answer key with versions A and B
- mock_student_responses.csv - Ground truth student responses

Usage:
------
Try these commands to test MarkShark:

1. Score the sample scans:

   markshark quick-grade mock_scans.pdf \
       --template MarkShark-64Q \
       --key-txt mock_answer_key.txt \
       --out-csv results.csv

2. Or use the two-step process:

   markshark align mock_scans.pdf \
       --template path/to/templates/MarkShark-64Q/master_template.pdf \
       --out-pdf aligned.pdf

   markshark score aligned.pdf \
       --bublmap path/to/templates/MarkShark-64Q/bubblemap.yaml \
       --key-txt mock_answer_key.txt \
       --out-csv results.csv

3. Generate a report:

   markshark report results.csv --out-xlsx report.xlsx

Generated with:
---------------
markshark mock-dataset --template MarkShark-64Q --num-students 10 --seed 42
