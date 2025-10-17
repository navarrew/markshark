# make_aruco_markers.py
import cv2 as cv
#CV means 'computer vision'

aruco = cv.aruco
DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

for id_ in [0,1,2,3,4,5]:
    img = aruco.generateImageMarker(DICT, id_, 200)
    cv.imwrite(f"aruco_{id_}.png", img)
    
     
"""
Parameters for the aruco.generateImageMarker() function are as follows:
DICT= the ArUco dictionary (set of patterns)
id_ = which marker from that dictionary
200 = output size in pixels

More about ArUco:
An ArUco dictionary is a predefined set of binary square markers — 
each marker has a unique internal bit pattern (like a barcode) 
that can be recognized by computer vision.

Each dictionary has:
    • A grid size (e.g., 4×4 or 6×6 bits),
    • A number of unique markers (e.g., 50, 100, 250, 1000),
    • A defined error correction capability (so detection is 
      robust even if some bits are misread).
    
When you see something like "DICT_4X4_50", it means:
'4×4 bits per marker, 50 unique markers in the set.'

IMPORTANT! marker #1 from dictionary 4x4_50 is not the same as marker #1 from 4x4_100
Therefore: You must use the same dictionary for both generation and detection.

Why?
Each dictionary (like DICT_4X4_50, DICT_4X4_100, etc.) is generated separately to 
maximize the Hamming distance (bitwise difference) between markers within that set.

Each dictionary is designed with an algorithm that selects unique bit patterns that are:
    • As different from one another as possible (for low misclassification),
    • Resilient to rotations (since markers can be viewed upside down),
    • Constrained by how many total markers you request (50, 100, 250…).
    
So even though both 4X4_50 and 4X4_100 use 4×4 bit grids, 
their internal binary patterns are different — and it is highly unlikely
that marker ID #1, or any given ID, will correspond
to the same bit pattern between two different ArUco dictionaries.

So when you ask for a larger dictionary (say, 100 instead of 50), 
OpenCV recomputes the set of valid bit matrices to maximize uniqueness
across all 100 — not just “adds 50 more” to the first 50.

"""