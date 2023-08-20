"""Unit tests for the main detector code"""

import pytest
from copydetect import CopyDetector, CodeFingerprint, compare_files
import numpy as np
from pathlib import Path

TESTS_DIR = str(Path(__file__).parent)


@pytest.fixture
def sample_file_metrics():
    return {
        "file1_len": 2052,
        "file2_len": 1257,
        "token_overlap": 1155
    }


class TestTwoFileDetection:
    """Test of the user-facing copydetect code for a simple two-file
    case. The two files both use several sections from a boilerplate
    file but are otherwise different.
    """
    def test_compare(self, sample_file_metrics):
        config = {
            "test_directories" : [TESTS_DIR + "/sample_py/code"],
            "reference_directories" : [TESTS_DIR + "/sample_py/code"],
            "extensions" : ["py"],
            "noise_threshold" : 25,
            "guarantee_threshold" : 25,
            "display_threshold" : 0,
            "silent" : True
        }
        detector = CopyDetector.from_config(config)
        detector.run()

        overlap = sample_file_metrics["token_overlap"]
        sim1 = overlap / sample_file_metrics["file1_len"]
        sim2 = overlap / sample_file_metrics["file2_len"]

        # file order is not guaranteed, so there are two possible
        # similarity matrices depending on the order of the files
        possible_mtx_1 = np.array([[[-1, -1], [sim1, sim2]],
                                  [[sim2, sim1], [-1, -1]]])
        possible_mtx_2 = np.flip(possible_mtx_1, 2)
        assert (np.array_equal(possible_mtx_1, detector.similarity_matrix)
                or np.array_equal(possible_mtx_2, detector.similarity_matrix))
        assert np.array_equal(np.array([[-1, overlap],[overlap,-1]]),
                              detector.token_overlap_matrix)

        html_out = detector.generate_html_report(output_mode="return")

        # verify highlighting code isn't being escaped
        test_str1 = "data[2] = [</span>0<span class='highlight-"
        test_str2 = "data[2] = [</span>3<span class='highlight-"
        # verify input code is being escaped
        test_str3 = "print(&#34;Incorrect num&#34;"
        assert test_str1 in html_out
        assert test_str2 in html_out
        assert test_str3 in html_out

    def test_compare_manual_config(self, sample_file_metrics):
        detector = CopyDetector(noise_t=25, guarantee_t=25, silent=True)
        detector.add_file(TESTS_DIR + "/sample_py/code/sample1.py")
        detector.add_file(TESTS_DIR + "/sample_py/code/sample2.py")
        detector.run()

        overlap = sample_file_metrics["token_overlap"]
        sim1 = overlap / sample_file_metrics["file1_len"]
        sim2 = overlap / sample_file_metrics["file2_len"]

        assert np.array_equal(np.array([[[-1, -1], [sim1, sim2]],
                                        [[sim2, sim1], [-1, -1]]]),
                              detector.similarity_matrix)
        assert np.array_equal(np.array([[-1,overlap],[overlap,-1]]),
                              detector.token_overlap_matrix)

    def test_compare_saving(self, tmpdir):
        config = {
            "test_directories" : [TESTS_DIR + "/sample_py/code"],
            "reference_directories" : [TESTS_DIR + "/sample_py/code"],
            "extensions" : ["py"],
            "noise_threshold" : 25,
            "guarantee_threshold" : 25,
            "display_threshold" : 0,
            "disable_autoopen" : True,
            "out_file" : tmpdir,
            "silent" : True
        }
        detector = CopyDetector.from_config(config)
        detector.run()
        detector.generate_html_report()

        # check for expected files
        assert Path(tmpdir + "/report.html").exists()

    def test_compare_boilerplate(self):
        config = {
            "test_directories" : [TESTS_DIR + "/sample_py/code"],
            "reference_directories" : [TESTS_DIR + "/sample_py/code"],
            "boilerplate_directories" : [TESTS_DIR + "/sample_py/boilerplate"],
            "extensions" : ["py"],
            "noise_threshold" : 25,
            "guarantee_threshold" : 25,
            "display_threshold" : 0,
            "silent": True
        }
        detector = CopyDetector.from_config(config)
        detector.run()

        assert np.array_equal(np.array([[[-1,-1],[0,0]],[[0,0],[-1,-1]]]),
                              detector.similarity_matrix)
        assert np.array_equal(np.array([[-1,0],[0,-1]]),
                              detector.token_overlap_matrix)

    def test_severalfiles(self, tmpdir):
        """Run the detector over all the files in the tests directory
        and perform some basic sanity checking.
        """
        config = {
            "test_directories" : [TESTS_DIR],
            "reference_directories" : [TESTS_DIR],
            "extensions" : ["*"],
            "noise_threshold" : 25,
            "guarantee_threshold" : 30,
            "display_threshold" : 0.3,
            "disable_autoopen" : True,
            "out_file" : tmpdir,
            "silent": True
        }
        detector = CopyDetector.from_config(config)
        detector.run()
        html_out = detector.generate_html_report()

        skipped_files = detector.similarity_matrix[:,:,0] == -1
        assert np.all(detector.similarity_matrix[~skipped_files, 0] >= 0)
        assert np.any(detector.similarity_matrix[~skipped_files, 0] > 0)
        assert np.all(detector.similarity_matrix[~skipped_files, 0] <= 1)
        assert np.all(detector.token_overlap_matrix[~skipped_files] >= 0)

class TestTwoFileAPIDetection():
    """Performs the same checks as the other two-file check, but uses
    the API instead of the command line code.
    """
    def test_compare(self, sample_file_metrics):
        fp1 = CodeFingerprint(TESTS_DIR+"/sample_py/code/sample1.py", 25, 1)
        fp2 = CodeFingerprint(TESTS_DIR+"/sample_py/code/sample2.py", 25, 1)
        token_overlap, similarities, slices = compare_files(fp1, fp2)

        overlap = sample_file_metrics["token_overlap"]

        assert token_overlap == overlap
        assert similarities[0] == overlap / sample_file_metrics["file1_len"]
        assert similarities[1] == overlap / sample_file_metrics["file2_len"]

    def test_compare_boilerplate(self):
        bp_fingerprint = CodeFingerprint(
            TESTS_DIR + "/sample_py/boilerplate/handout.py", 25, 1)
        fp1 = CodeFingerprint(TESTS_DIR+"/sample_py/code/sample1.py", 25, 1,
                              np.array(list(bp_fingerprint.hashes)))
        fp2 = CodeFingerprint(TESTS_DIR+"/sample_py/code/sample2.py", 25, 1,
                              np.array(list(bp_fingerprint.hashes)))

        token_overlap, similarities, slices = compare_files(fp1, fp2)

        assert token_overlap == 0
        assert similarities[0] == 0
        assert similarities[1] == 0

class TestParameters():
    """Test cases for individual parameters"""
    def test_ignore_leaf(self):
        detector = CopyDetector(test_dirs=[TESTS_DIR + "/sample_py"],
                                ignore_leaf=True, silent=True)
        detector.run()

        # sample1 and sample2 should not have been compared
        # + 4 self compares = 6 total skips
        assert np.sum(detector.similarity_matrix[:,:,0] == -1) == 6

    def test_same_name_only(self):
        detector = CopyDetector(test_dirs=[TESTS_DIR + "/sample_py"],
                                same_name_only=True, silent=True)
        detector.run()

        # the only comparison should be between the two handout.py files
        assert np.sum(detector.similarity_matrix[:,:,0] != -1) == 2

    def test_disable_filtering(self):
        detector = CopyDetector(test_dirs=[TESTS_DIR + "/sample_py"],
                                disable_filtering=True, silent=True)
        detector.run()

        fingerprint1 = detector.file_data[
            str(Path(TESTS_DIR + "/sample_py/code/sample1.py"))]
        assert fingerprint1.raw_code == fingerprint1.filtered_code

    def test_force_language(self):
        detector = CopyDetector(test_dirs=[TESTS_DIR + "/sample_py"],
                                force_language="java", silent=True)
        detector.run()

        fingerprint1 = detector.file_data[
            str(Path(TESTS_DIR + "/sample_py/handout.py"))]

        # "#" isn't a comment in java, so it won't be removed
        assert fingerprint1.filtered_code[0] == "#"

    def test_truncation(self):
        detector = CopyDetector(
            test_dirs=[TESTS_DIR + "/sample_py/boilerplate"],
            noise_t=10, guarantee_t=10, truncate=True, silent=True)
        detector.add_file(str(Path(TESTS_DIR + "/sample_py/handout.py")))
        detector.run()
        code_list = detector.get_copied_code_list()

        assert len(code_list[0][4]) < 500 and len(code_list[0][5]) < 500

    def test_out_file(self, tmpdir):
        detector = CopyDetector(test_dirs=[TESTS_DIR + "/sample_py"],
            silent=True, out_file=tmpdir + "/test", autoopen=False)
        detector.run()
        detector.generate_html_report()

        assert Path(tmpdir + "/test.html").exists()

        with pytest.raises(ValueError):
            detector = CopyDetector(test_dirs=[TESTS_DIR + "/sample_py"],
            silent=True, out_file=tmpdir + "/not_a_dir/test")

        detector = CopyDetector(test_dirs=[TESTS_DIR + "/sample_py"],
            silent=True, out_file=tmpdir, autoopen=False)
        detector.run()
        detector.generate_html_report()

        assert Path(tmpdir + "/report.html").exists()

    def test_encoding_specification(self):
        detector = CopyDetector(test_dirs=[TESTS_DIR + "/sample_py"],
                                extensions=["c"], encoding="utf-16",
                                silent=True)
        detector.add_file(TESTS_DIR + "/sample_other/c_sample_utf16.c")
        detector.run()

        # make sure utf-16 file was loaded correctly
        assert len(list(detector.file_data.values())[0].raw_code) > 0
