"""Some basic sanity checks to make sure the detector is reporting a
reasonable similarity percent for certain file pairs.
"""

from pathlib import Path

import pytest

from copydetect import CodeFingerprint, compare_files

TESTS_DIR = str(Path(__file__).parent)

def test_rot_c():
    """Make sure clearly copied c code is correctly reported as high
    similarity
    """
    fp1 = CodeFingerprint(TESTS_DIR + "/sample_sanity_check/rot1.c", 25, 1)
    fp2 = CodeFingerprint(TESTS_DIR + "/sample_sanity_check/rot2.c", 25, 1)
    token_overlap, similarities, slices = compare_files(fp1, fp2)

    assert similarities[0] > 0.8
    assert similarities[1] > 0.8

def test_sample_xml():
    """Make sure identical XML file compared to itself returns 100%
    similarity
    """
    fp1 = CodeFingerprint(TESTS_DIR + "/sample_sanity_check/sample.xml", 25, 1)
    fp2 = CodeFingerprint(TESTS_DIR + "/sample_sanity_check/sample.xml", 25, 1)
    token_overlap, similarities, slices = compare_files(fp1, fp2)

    assert similarities[0] == 1.0
    assert similarities[1] == 1.0
