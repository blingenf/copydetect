"""Unit tests for the main detector code"""

import unittest
from copydetect import CopyDetector, CodeFingerprint, compare_files
import numpy as np
import os

class TwoFileTestCase(unittest.TestCase):
    def test_compare(self):
        config = {
          "test_directories" : [os.path.dirname(__file__) + "/sample/code"],
          "reference_directories" : [os.path.dirname(__file__)+"/sample/code"],
          "extensions" : ["py"],
          "noise_threshold" : 25,
          "guarantee_threshold" : 25,
          "display_threshold" : 0
        }
        detector = CopyDetector(config, silent=True)
        detector.run()

        sim_test = np.array_equal(np.array([[-1,1137/2052],[1137/1257,-1]]),
                                  detector.similarity_matrix)
        token_test = np.array_equal(np.array([[-1,1137],[1137,-1]]),
                                    detector.token_overlap_matrix)
        self.assertTrue(sim_test)
        self.assertTrue(token_test)

        html_out = detector.generate_html_report(output_mode="return")

        # verify highlighting code isn't being escaped
        test_str1 = "data[2] = [</span>0<span class='highlight-red'>, 6, 1]"
        test_str2 = "data[2] = [</span>3<span class='highlight-green'>, 6, 1]"
        # verify input code is being escaped
        test_str3 = "print(&#34;Incorrect num&#34;"
        self.assertTrue(test_str1 in html_out)
        self.assertTrue(test_str2 in html_out)
        self.assertTrue(test_str3 in html_out)

    def test_compare_boilerplate(self):
        config = {
          "test_directories" : [os.path.dirname(__file__) + "/sample/code"],
          "reference_directories" : [os.path.dirname(__file__)+"/sample/code"],
          "boilerplate_directories" : [os.path.dirname(__file__)
                                       + "/sample/boilerplate"],
          "extensions" : ["py"],
          "noise_threshold" : 25,
          "guarantee_threshold" : 25,
          "display_threshold" : 0
        }
        detector = CopyDetector(config, silent=True)
        detector.run()

        sim_test = np.array_equal(np.array([[-1,0],[0,-1]]),
                                  detector.similarity_matrix)
        token_test = np.array_equal(np.array([[-1,0],[0,-1]]),
                                    detector.token_overlap_matrix)
        self.assertTrue(sim_test)
        self.assertTrue(token_test)

class TwoFileAPITestCase(unittest.TestCase):
    def test_compare(self):
        fp1 = CodeFingerprint(os.path.dirname(__file__)
                              + "/sample/code/sample1.py", 25, 1)
        fp2 = CodeFingerprint(os.path.dirname(__file__)
                              + "/sample/code/sample2.py", 25, 1)
        token_overlap, similarities, slices = compare_files(fp1, fp2)

        self.assertEqual(token_overlap, 1137)
        self.assertEqual(similarities[0], 1137/2052)
        self.assertEqual(similarities[1], 1137/1257)

    def test_compare_boilerplate(self):
        bp_fingerprint = CodeFingerprint(
            os.path.dirname(__file__)+"/sample/boilerplate/handout.py", 25, 1)
        fp1 = CodeFingerprint(
            os.path.dirname(__file__) + "/sample/code/sample1.py", 25, 1,
            bp_fingerprint.hashes)
        fp2 = CodeFingerprint(
            os.path.dirname(__file__) + "/sample/code/sample2.py", 25, 1,
            bp_fingerprint.hashes)

        token_overlap, similarities, slices = compare_files(fp1, fp2)

        self.assertEqual(token_overlap, 0)
        self.assertEqual(similarities[0], 0)
        self.assertEqual(similarities[1], 0)

if __name__ == '__main__':
    unittest.main()
