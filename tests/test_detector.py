"""Unit tests for the main detector code"""

import pytest
from copydetect import CopyDetector, CodeFingerprint, compare_files
import numpy as np
from pathlib import Path

tests_dir = str(Path(__file__).parent)

class TestTwoFileDetection():
    """Test of the user-facing copydetect code for a simple two-file
    case. The two files both use several sections from a boilerplate
    file but are otherwise different.
    """
    def test_compare(self):
        config = {
          "test_directories" : [tests_dir + "/sample/code"],
          "reference_directories" : [tests_dir + "/sample/code"],
          "extensions" : ["py"],
          "noise_threshold" : 25,
          "guarantee_threshold" : 25,
          "display_threshold" : 0
        }
        detector = CopyDetector(config, silent=True)
        detector.run()

        assert np.array_equal(np.array([[-1,1137/2052],[1137/1257,-1]]),
                              detector.similarity_matrix)
        assert np.array_equal(np.array([[-1,1137],[1137,-1]]),
                              detector.token_overlap_matrix)

        html_out = detector.generate_html_report(output_mode="return")

        # verify highlighting code isn't being escaped
        test_str1 = "data[2] = [</span>0<span class='highlight-red'>, 6, 1]"
        test_str2 = "data[2] = [</span>3<span class='highlight-green'>, 6, 1]"
        # verify input code is being escaped
        test_str3 = "print(&#34;Incorrect num&#34;"
        assert test_str1 in html_out
        assert test_str2 in html_out
        assert test_str3 in html_out

    def test_compare_manual_config(self):
        detector = CopyDetector(noise_t=25, guarantee_t=25, silent=True)
        detector.add_file(tests_dir + "/sample/code/sample1.py")
        detector.add_file(tests_dir + "/sample/code/sample2.py")
        detector.run()

        assert np.array_equal(np.array([[-1,1137/2052],[1137/1257,-1]]),
                              detector.similarity_matrix)
        assert np.array_equal(np.array([[-1,1137],[1137,-1]]),
                              detector.token_overlap_matrix)

    def test_compare_saving(self, tmpdir):
        config = {
          "test_directories" : [tests_dir + "/sample/code"],
          "reference_directories" : [tests_dir + "/sample/code"],
          "extensions" : ["py"],
          "noise_threshold" : 25,
          "guarantee_threshold" : 25,
          "display_threshold" : 0,
          "disable_autoopen" : True
        }
        detector = CopyDetector(config, silent=True)
        detector.run()
        detector.generate_html_report(tmpdir)
        output_paths = [path.name for path in Path(tmpdir).rglob("*")]

        # check for expected files
        assert any([path == "report.html" for path in output_paths])
        assert any([path == "sim_histogram.png" for path in output_paths])
        assert any([path == "sim_matrix.png" for path in output_paths])

    def test_compare_boilerplate(self):
        config = {
          "test_directories" : [tests_dir + "/sample/code"],
          "reference_directories" : [tests_dir + "/sample/code"],
          "boilerplate_directories" : [tests_dir + "/sample/boilerplate"],
          "extensions" : ["py"],
          "noise_threshold" : 25,
          "guarantee_threshold" : 25,
          "display_threshold" : 0
        }
        detector = CopyDetector(config, silent=True)
        detector.run()

        assert np.array_equal(np.array([[-1,0],[0,-1]]),
                              detector.similarity_matrix)
        assert np.array_equal(np.array([[-1,0],[0,-1]]),
                              detector.token_overlap_matrix)

    def test_severalfiles(self, tmpdir):
        """Run the detector over all the files in the tests directory
        and perform some basic sanity checking.
        """
        config = {
          "test_directories" : [tests_dir],
          "reference_directories" : [tests_dir],
          "extensions" : ["*"],
          "noise_threshold" : 25,
          "guarantee_threshold" : 30,
          "display_threshold" : 0.3,
          "disable_autoopen" : True
        }
        detector = CopyDetector(config, silent=True)
        detector.run()
        html_out = detector.generate_html_report(tmpdir)

        skipped_files = detector.similarity_matrix == -1
        assert np.all(detector.similarity_matrix[~skipped_files] >= 0)
        assert np.any(detector.similarity_matrix[~skipped_files] > 0)
        assert np.all(detector.similarity_matrix[~skipped_files] <= 1)
        assert np.all(detector.token_overlap_matrix[~skipped_files] >= 0)

class TestTwoFileAPIDetection():
    """Performs the same checks as the other two-file check, but uses
    the API instead of the command line code.
    """
    def test_compare(self):
        fp1 = CodeFingerprint(tests_dir + "/sample/code/sample1.py", 25, 1)
        fp2 = CodeFingerprint(tests_dir + "/sample/code/sample2.py", 25, 1)
        token_overlap, similarities, slices = compare_files(fp1, fp2)

        assert token_overlap == 1137
        assert similarities[0] == 1137/2052
        assert similarities[1] == 1137/1257

    def test_compare_boilerplate(self):
        bp_fingerprint = CodeFingerprint(
            tests_dir + "/sample/boilerplate/handout.py", 25, 1)
        fp1 = CodeFingerprint(tests_dir + "/sample/code/sample1.py", 25, 1,
                              bp_fingerprint.hashes)
        fp2 = CodeFingerprint(tests_dir + "/sample/code/sample2.py", 25, 1,
                              bp_fingerprint.hashes)

        token_overlap, similarities, slices = compare_files(fp1, fp2)

        assert token_overlap == 0
        assert similarities[0] == 0
        assert similarities[1] == 0

class TestParameters():
    """Test cases for individual parameters"""
    def test_ignore_leaf(self):
        detector = CopyDetector(test_dirs=[tests_dir + "/sample"],
                                ignore_leaf=True, silent=True)
        detector.run()

        # sample1 and sample2 should not have been compared
        # + 4 self compares = 6 total skips
        assert np.sum(detector.similarity_matrix == -1) == 6

    def test_same_name_only(self):
        detector = CopyDetector(test_dirs=[tests_dir + "/sample"],
                                same_name_only=True, silent=True)
        detector.run()

        # the only comparison should be between the two handout.py files
        assert np.sum(detector.similarity_matrix != -1) == 2

    def test_disable_filtering(self):
        detector = CopyDetector(test_dirs=[tests_dir + "/sample"],
                                disable_filtering=True, silent=True)
        detector.run()

        fingerprint1 = detector.file_data[
            str(Path(tests_dir+"/sample/code/sample1.py"))]
        assert fingerprint1.raw_code == fingerprint1.filtered_code

    def test_force_language(self):
        detector = CopyDetector(test_dirs=[tests_dir + "/sample"],
                                force_language="java", silent=True)
        detector.run()

        fingerprint1 = detector.file_data[
            str(Path(tests_dir + "/sample/handout.py"))]

        # "#" isn't a comment in java, so it won't be removed
        assert fingerprint1.filtered_code[0] == "#"

    def test_truncation(self):
        detector = CopyDetector(
            test_dirs=[tests_dir + "/sample/boilerplate"],
            noise_t=10, guarantee_t=10, truncate=True, silent=True)
        detector.add_file(str(Path(tests_dir + "/sample/handout.py")))
        detector.run()
        code_list = detector.get_copied_code_list()

        assert len(code_list[0][4]) < 500 and len(code_list[0][5]) < 500
