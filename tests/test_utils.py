"""Unit tests for the copy detect utils functions"""

import copydetect.utils as cd
from pathlib import Path
import numpy as np
import pytest

TESTS_DIR = str(Path(__file__).parent)

class TestSmallDoc():
    """Test the copy detection pipeline for two simple strings"""

    @pytest.fixture(autouse=True)
    def _testcase_variables(self):
        self.doc1 = "helloabcdefghijklmnop"
        self.doc2 = "asdafhellonopfdvcdef"
        self.boilerplate = cd.hashed_kgrams("nope", 3)

    def test_doc_overlap(self):
        fp1, idx1 = cd.get_document_fingerprints(self.doc1, 3, 1)
        fp2, idx2 = cd.get_document_fingerprints(self.doc2, 3, 1)
        overlap_1,overlap_2 = cd.find_fingerprint_overlap(fp1, fp2, idx1, idx2)

        assert np.array_equal(np.sort(overlap_1), np.array([0,1,2,7,8,18]))
        assert np.array_equal(np.sort(overlap_2), np.array([5,6,7,10,16,17]))

    def test_doc_overlap_boilerplate(self):
        fp1, idx1 = cd.get_document_fingerprints(self.doc1, 3, 1,
                                                 self.boilerplate)
        fp2, idx2 = cd.get_document_fingerprints(self.doc2, 3, 1,
                                                 self.boilerplate)
        overlap_1,overlap_2 = cd.find_fingerprint_overlap(fp1, fp2, idx1, idx2)

        assert np.array_equal(np.sort(overlap_1), np.array([0,1,2,7,8]))
        assert np.array_equal(np.sort(overlap_2), np.array([5,6,7,16,17]))

    def test_slice_computation(self):
        fp1, idx1 = cd.get_document_fingerprints(self.doc1, 3, 1)
        fp2, idx2 = cd.get_document_fingerprints(self.doc2, 3, 1)
        overlap_1,overlap_2 = cd.find_fingerprint_overlap(fp1, fp2, idx1, idx2)

        slices1 = cd.get_copied_slices(overlap_1, 3)
        slices2 = cd.get_copied_slices(overlap_2, 3)

        assert np.array_equal(slices1, np.array([[0,7,18], [5,11,21]]))
        assert np.array_equal(slices2, np.array([[5,10,16],[10,13,20]]))

    def test_highlighting(self):
        fp1, idx1 = cd.get_document_fingerprints(self.doc1, 3, 1)
        fp2, idx2 = cd.get_document_fingerprints(self.doc2, 3, 1)
        overlap_1,overlap_2 = cd.find_fingerprint_overlap(fp1, fp2, idx1, idx2)

        slices1 = cd.get_copied_slices(overlap_1, 3)
        slices2 = cd.get_copied_slices(overlap_2, 3)

        hl1, similarity1 = cd.highlight_overlap(self.doc1, slices1, '->', '<')
        hl2, similarity2 = cd.highlight_overlap(self.doc2, slices2, '->', '<')

        assert hl1 == "->hello<ab->cdef<ghijklm->nop<"
        assert hl2 == "asdaf->hello<->nop<fdv->cdef<"
        assert 12/21 == similarity1
        assert 12/20 == similarity2

class TestTokenizerPythonSample():
    """Test code tokenization, filtering, and copy detection on a small
    example function.
    """

    @pytest.fixture(autouse=True)
    def _testcase_variables(self):
        sample1 = ("def hashed_kgrams(string, k):\n"
                   "    \"\"\"Return hashes of all k-grams in string\"\"\"\n"
                   "    hashes = [hash(string[offset:offset+k])\n"
                   "              for offset in range(len(string) - k + 1)]\n"
                   "    return np.array(hashes)")
        sample2 = ("def hash_f(s, k):\n"
                   "    h = [hash(s[o:o+k]) for o in range(len(s)-k+1)]\n"
                   "    return np.array(h)")
        self.sample_code = sample1
        self.sample_copied_code = sample2

    def test_tokenization(self):
        expected_out = ("defF(V,V):SV=[hash(V[V:V+V])forV"
                        "inrange(len(V)-V+1)]returnV.V(V)")
        expected_offsets = np.array([[0,0],[2,0],[3,1],[5,13],[6,18],[7,19],
            [9,19],[9,20],[10,24],[10,67],[10,68],[11,72],[11,77],[12,78],
            [19,79],[21,84],[23,89],[25,94],[27,94],[27,95],[30,109],[31,110],
            [31,115],[33,116],[44,117],[45,122],[46,123],[47,124],[47,124],
            [48,125],[51,126],[51,127],[57,131],[58,132],[60,133],[62,137],
            [63,142]])

        out_code, offsets = cd.filter_code(self.sample_code, "test.py")

        assert out_code == expected_out
        assert np.array_equal(np.array(offsets), expected_offsets)

    def test_copydetect(self):
        out_code1, offsets1 = cd.filter_code(self.sample_code, "1.py")
        out_code2, offsets2 = cd.filter_code(self.sample_copied_code, "2.py")

        hashes1, idx1 = cd.get_document_fingerprints(out_code1, 20, 1)
        hashes2, idx2 = cd.get_document_fingerprints(out_code2, 20, 1)
        ol_idx1, ol_idx2 = cd.find_fingerprint_overlap(hashes1, hashes2,
                                                       idx1, idx2)
        slices1 = cd.get_copied_slices(ol_idx1, 20)
        slices2 = cd.get_copied_slices(ol_idx2, 20)

        hl_filter1, _ = cd.highlight_overlap(out_code1, slices1, ">", "<")
        hl_filter2, _ = cd.highlight_overlap(out_code2, slices2, ">", "<")

        slices1 += offsets1[:,1][np.clip(
            np.searchsorted(offsets1[:,0], slices1), 0, offsets1.shape[0] - 1)]
        slices2 += offsets2[:,1][np.clip(
            np.searchsorted(offsets2[:,0], slices2), 0, offsets2.shape[0] - 1)]

        hl_code1, similarity1 = cd.highlight_overlap(self.sample_code,
                                                     slices1, ">", "<")
        hl_code2, similarity2 = cd.highlight_overlap(self.sample_copied_code,
                                                     slices2, ">", "<")

        gt_hl1 = ("def hashed_kgrams(string, k):\n"
                  "    \"\"\"Return hashes of all k-grams in string\"\"\"\n"
                  "    >hashes = [hash(string[offset:offset+k])\n"
                  "              for offset in range(len(string) - k + 1)]\n"
                  "    return np.array(hashes)<")
        gt_hl2 = ("def hash_f(s, k):\n"
                  "    >h = [hash(s[o:o+k]) for o in range(len(s)-k+1)]\n"
                  "    return np.array(h)<")

        assert similarity1 == 123/len(self.sample_code)
        assert similarity2 == 70/len(self.sample_copied_code)
        assert hl_code1 == gt_hl1
        assert hl_code2 == gt_hl2

class TestTokenizerOtherSamples():
    """Test code tokenization and filtering for a variety of common
    languages
    """

    def test_php_tokenization(self):
        with open(TESTS_DIR + "/sample_other/php_sample.php") as php_f:
            php_sample = php_f.read()
        out_code, offsets = cd.filter_code(php_sample, "php_sample.php")

        # PHP variables are tokenized differently because they
        # have a $ prefix
        assert "var1" not in out_code
        # preprocessor directive should be filtered, builtin
        # function should not
        assert "Psession_start()" in out_code

    def test_c_tokenization(self):
        with open(TESTS_DIR + "/sample_other/c_sample.c") as c_f:
            c_sample = c_f.read()
        out_code, offsets = cd.filter_code(c_sample, "c_sample.c")

        # Preprocessor directive and function name should be filtered,
        # return type should not
        assert "PintF" in out_code
        # * placement shouldn't matter
        assert "char*V" in out_code

    def test_java_tokenization(self):
        with open(TESTS_DIR + "/sample_other/java_sample.java") as java_f:
            java_sample = java_f.read()
        out_code, offsets = cd.filter_code(java_sample, "java_sample.java")

        # member variables should be tokenized
        assert "this.V=V" in out_code
        assert "modifier" not in out_code
        # decorators shouldn't be filtered
        # String isn't a primitive type in java so it's tokenized as
        # Token.Name, which can't be differentiated from normal variables
        assert "@OverridepublicVF" in out_code

    def test_get_token_coverage(self):
        sample = "0123456789"
        idx1 = {0: [0], 1: [5]}

        # two 5-grams starting at 0 and 5 cover all 10 tokens
        assert cd.get_token_coverage(idx1, 5, len(sample)) == len(sample)

        # with 3-grams, 3, 4, 8, and 9 are not covered
        assert cd.get_token_coverage(idx1, 3, len(sample)) == len(sample) - 4

        # with 1-grams only 0 and 5 are covered
        assert cd.get_token_coverage(idx1, 1, len(sample)) == 2

        # k-gram overlap shouldn't matter
        idx = {i: [i] for i in range(8)}
        assert cd.get_token_coverage(idx, 3, len(sample)) == len(sample)
