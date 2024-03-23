"""This module contains functions for detecting overlap between
a set of test files (files to check for plagairism) and a set of
reference files (files that might have been plagairised from).
"""

from pathlib import Path
import time
import logging
import webbrowser
import importlib.resources
import io
import base64
import json

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from jinja2 import Template

import copydetect.data as data_files
from .utils import (filter_code, highlight_overlap, get_copied_slices,
                    get_document_fingerprints, find_fingerprint_overlap,
                    get_token_coverage)
from . import __version__
from . import defaults
from ._config import CopydetectConfig

class CodeFingerprint:
    """Class for tokenizing, filtering, fingerprinting, and winnowing
    a file. Maintains information about fingerprint indexes and token
    indexes to assist code highlighting for the output report.

    Parameters
    ----------
    file : str
        Path to the file fingerprints should be extracted from.
    k : int
        Length of k-grams to extract as fingerprints.
    win_size : int
        Window size to use for winnowing (must be >= 1).
    boilerplate : array_like, optional
        List of fingerprints to use as boilerplate. Any fingerprints
        present in this list will be discarded from the hash list.
    filter : bool, default=True
        If set to to False, code will not be tokenized & filtered.
    fp : TextIO, default=None
        I/O stream for data to create a fingerprint for. If provided,
        the "file" argument will not be used to load a file from disk
        but will still be used for language detection and displayed on
        the report.
    encoding : str, default="utf-8"
        Text encoding to use for reading the file. If "DETECT", the
        chardet library will be used (if installed) to automatically
        detect file encoding

    Attributes
    ----------
    filename : str
        Name of the originally provided file.
    raw_code : str
        Unfiltered code.
    filtered_code : str
        Code after tokenization and filtering. If filter=False, this is
        the same as raw_code.
    offsets : Nx2 array of ints
        The cumulative number of characters removed during filtering at
        each index of the filtered code. Used for translating locations
        in the filtered code to locations in the unfiltered code.
    hashes : Set[int]
        Set of fingerprint hashes extracted from the filtered code.
    hash_idx : Dict[int, List[int]]
        Mapping of each fingerprint hash back to all indexes in the
        original code in which this fingerprint appeared.
    k : int
        Value of provided k argument.
    language : str
        If set, will force the tokenizer to use the provided language
        rather than guessing from the file extension.
    token_coverage : int
        The number of tokens in the tokenized code which are considered
        for fingerprint comparison, after performing winnowing and
        removing boilerplate.
    """
    def __init__(self, file, k, win_size, boilerplate=None, filter=True,
                 language=None, fp=None, encoding: str = "utf-8"):
        if boilerplate is None:
            boilerplate = []
        if fp is not None:
            code = fp.read()
        elif encoding == "DETECT":
            try:
                import chardet
                with open(file, "rb") as code_fp:
                    code = code_fp.read()
                detected_encoding = chardet.detect(code)["encoding"]
                if detected_encoding is not None:
                    code = code.decode(detected_encoding)
                else:
                    # if encoding can't be detected, just use the default
                    # encoding (the file may be empty)
                    code = code.decode()
            except ModuleNotFoundError as e:
                logging.error(
                    "encoding detection requires chardet to be installed"
                )
                raise e
        else:
            with open(file, encoding=encoding) as code_fp:
                code = code_fp.read()
        if filter:
            filtered_code, offsets = filter_code(code, file, language)
        else:
            filtered_code, offsets = code, np.array([])
        hashes, idx = get_document_fingerprints(filtered_code, k, win_size,
                                                boilerplate)

        self.filename = file
        self.raw_code = code
        self.filtered_code = filtered_code
        self.offsets = offsets
        self.hashes = hashes
        self.hash_idx = idx
        self.k = k
        self.token_coverage = get_token_coverage(idx, k, len(filtered_code))

def compare_files(file1_data, file2_data):
    """Computes the overlap between two CodeFingerprint objects
    using the generic methods from copy_detect.py. Returns the
    number of overlapping tokens and two tuples containing the
    overlap percentage and copied slices for each unfiltered file.

    Parameters
    ----------
    file1_data : CodeFingerprint
        CodeFingerprint object of file #1.
    file2_data : CodeFingerprint
        CodeFingerprint object of file #2.

    Returns
    -------
    token_overlap : int
        Number of overlapping tokens between the two files.
    similarities : tuple of 2 ints
        For both files: number of overlapping tokens divided by the
        total number of tokens in that file.
    slices : tuple of 2 2xN int arrays
        For both files: locations of copied code in the unfiltered
        text. Dimension 0 contains slice starts, dimension 1 contains
        slice ends.
    """
    if file1_data.k != file2_data.k:
        raise ValueError("Code fingerprints must use the same noise threshold")
    idx1, idx2 = find_fingerprint_overlap(
        file1_data.hashes, file2_data.hashes,
        file1_data.hash_idx, file2_data.hash_idx)
    slices1 = get_copied_slices(idx1, file1_data.k)
    slices2 = get_copied_slices(idx2, file2_data.k)
    if len(slices1[0]) == 0:
        return 0, (0,0), (np.array([]), np.array([]))

    token_overlap1 = np.sum(slices1[1] - slices1[0])
    token_overlap2 = np.sum(slices2[1] - slices2[0])

    if len(file1_data.filtered_code) > 0:
        similarity1 = token_overlap1 / file1_data.token_coverage
    else:
        similarity1 = 0
    if len(file2_data.filtered_code) > 0:
        similarity2 = token_overlap2 / file2_data.token_coverage
    else:
        similarity2 = 0

    if len(file1_data.offsets) > 0:
        slices1 += file1_data.offsets[:,1][np.clip(
            np.searchsorted(file1_data.offsets[:,0], slices1),
            0, file1_data.offsets.shape[0] - 1)]
    if len(file2_data.offsets) > 0:
        slices2 += file2_data.offsets[:,1][np.clip(
            np.searchsorted(file2_data.offsets[:,0], slices2),
            0, file2_data.offsets.shape[0] - 1)]

    return token_overlap1, (similarity1,similarity2), (slices1,slices2)

class CopyDetector:
    """Main plagairism detection class. Searches provided directories
    and uses detection parameters to calculate similarity between all
    files found in the directories

    Parameters
    ----------
    test_dirs : list
        (test_directories) A list of directories to recursively search
        for files to check for plagiarism.
    ref_dirs: list
        (reference_directories) A list of directories to search for
        files to compare the test files to. This should generally be a
        superset of test_directories
    boilerplate_dirs : list
        (boilerplate_directories) A list of directories containing
        boilerplate code. Matches between fingerprints present in the
        boilerplate code will not be considered plagiarism.
    extensions : list
        A list of file extensions containing code the detector should
        look at.
    noise_t : int
        (noise_threshold) The smallest sequence of matching characters
        between two files which should be considered plagiarism. Note
        that tokenization and filtering replaces variable names with V,
        function names with F, object names with O, and strings with S
        so the threshold should be lower than you would expect from the
        original code.
    guarantee_t : int
        (guarantee_threshold) The smallest sequence of matching
        characters between two files for which the system is guaranteed
        to detect a match. This must be greater than or equal to the
        noise threshold. If computation time is not an issue, you can
        set guarantee_threshold = noise_threshold.
    display_t : float
        (display_threshold) The similarity percentage cutoff for
        displaying similar files on the detector report.
    same_name_only : bool
        If true, the detector will only compare files that have the
        same name
    ignore_leaf : bool
        If true, the detector will not compare files located in the
        same leaf directory.
    autoopen : bool
        If true, the detector will automatically open a webbrowser to
        display the results of generate_html_report
    disable_filtering : bool
        If true, the detector will not tokenize and filter code before
        generating file fingerprints.
    force_language : str
        If set, forces the tokenizer to use a particular programming
        language regardless of the file extension.
    truncate : bool
        If true, highlighted code will be truncated to remove non-
        highlighted regions from the displayed output
    out_file : str
        Path to output report file.
    css_files: list
        List of css files that will be linked within the generated html report
    silent : bool
        If true, all logging output will be supressed.
    encoding : str, default="utf-8"
        Text encoding to use for reading the file. If "DETECT", the
        chardet library will be used (if installed) to automatically
        detect file encoding
    """
    def __init__(self, test_dirs=None, ref_dirs=None,
                 boilerplate_dirs=None, extensions=None,
                 noise_t=defaults.NOISE_THRESHOLD,
                 guarantee_t=defaults.GUARANTEE_THRESHOLD,
                 display_t=defaults.DISPLAY_THRESHOLD,
                 same_name_only=False, ignore_leaf=False, autoopen=True,
                 disable_filtering=False, force_language=None,
                 truncate=False, out_file="./report.html", css_files=None,
                 silent=False, encoding: str = "utf-8"):
        conf_args = locals()
        conf_args = {
            key: val
            for key, val in conf_args.items()
            if key != "self" and val is not None
        }
        self.conf = CopydetectConfig(**conf_args)

        self.test_files = self._get_file_list(
            self.conf.test_dirs, self.conf.extensions
        )
        self.ref_files = self._get_file_list(
            self.conf.ref_dirs, self.conf.extensions
        )
        self.boilerplate_files = self._get_file_list(
            self.conf.boilerplate_dirs, self.conf.extensions
        )

        # before run() is called, similarity data should be empty
        self.similarity_matrix = np.array([])
        self.token_overlap_matrix = np.array([])
        self.slice_matrix = {}
        self.file_data = {}

    @classmethod
    def from_config(cls, config):
        """Initializes a CopyDetection object using the provided
        configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary using CLI parameter names.

        Returns
        -------
        CopyDetector
            Detection object initialized with config
        """
        params = CopydetectConfig.normalize_json(config)
        return cls(**params)

    def _get_file_list(self, dirs, exts):
        """Recursively collects list of files from provided
        directories. Used to search test_dirs, ref_dirs, and
        boilerplate_dirs
        """
        file_list = []
        for dir in dirs:
            print_warning = True
            for ext in exts:
                if ext == "*":
                    matched_contents = Path(dir).rglob("*")
                else:
                    matched_contents = Path(dir).rglob("*."+ext.lstrip("."))
                files = [str(f) for f in matched_contents if f.is_file()]

                if len(files) > 0:
                    print_warning = False
                file_list.extend(files)
            if print_warning:
                logging.warning("No files found in " + dir)

        # convert to a set to remove duplicates, then back to a list
        return list(set(file_list))

    def add_file(self, filename, type="testref"):
        """Adds a file to the list of test files, reference files, or
        boilerplate files.

        Parameters
        ----------
        filename : str
            Name of file to add.
        type : {"testref", "test", "ref", "boilerplate"}
            Type of file to add. "testref" will add the file as both a
            test and reference file.
        """
        if type == "testref":
            self.test_files.append(filename)
            self.ref_files.append(filename)
        elif type == "test":
            self.test_files.append(filename)
        elif type == "ref":
            self.ref_files.append(filename)
        elif type == "boilerplate":
            self.boilerplate_files.append(filename)

    def _get_boilerplate_hashes(self):
        """Generates a list of hashes of the boilerplate text. Returns
        a set containing all unique k-gram hashes across all files
        found in the boilerplate directories.
        """
        boilerplate_hashes = []
        for file in self.boilerplate_files:
            try:
                fingerprint = CodeFingerprint(
                    file,
                    k=self.conf.noise_t,
                    win_size=1,
                    filter=not self.conf.disable_filtering,
                    language=self.conf.force_language,
                    encoding=self.conf.encoding
                )
                boilerplate_hashes.extend(fingerprint.hashes)
            except UnicodeDecodeError:
                logging.warning(f"Skipping {file}: file not UTF-8 text")
                continue

        return np.unique(np.array(boilerplate_hashes))

    def _preprocess_code(self, file_list):
        """Generates a CodeFingerprint object for each file in the
        provided file list. This is where the winnowing algorithm is
        actually used.
        """
        boilerplate_hashes = self._get_boilerplate_hashes()
        for code_f in tqdm(file_list, bar_format= '   {l_bar}{bar}{r_bar}',
                           disable=self.conf.silent):
            if code_f not in self.file_data:
                try:
                    self.file_data[code_f] = CodeFingerprint(
                        code_f, self.conf.noise_t, self.conf.window_size,
                        boilerplate_hashes, not self.conf.disable_filtering,
                        self.conf.force_language, encoding=self.conf.encoding)

                except UnicodeDecodeError:
                    logging.warning(f"Skipping {code_f}: file not UTF-8 text")
                    continue

    def _comparison_loop(self):
        """The core code used to determine code overlap. The overlap
        between each test file and each compare file is computed and
        stored in similarity_matrix. Token overlap information and the
        locations of copied code are stored in slice_matrix and
        token_overlap_matrix, respectively.
        """

        self.similarity_matrix = np.full(
            (len(self.test_files), len(self.ref_files), 2),
            -1,
            dtype=np.float64,
        )
        self.token_overlap_matrix = np.full(
            (len(self.test_files), len(self.ref_files)), -1
        )
        self.slice_matrix = {}

        # this is used to track which files have been compared to avoid
        # unnecessary duplication when there is overlap between the
        # test and reference files
        comparisons = {}

        for i, test_f in enumerate(
            tqdm(self.test_files,
                 bar_format= '   {l_bar}{bar}{r_bar}',
                 disable=self.conf.silent)
        ):
            for j, ref_f in enumerate(self.ref_files):
                if (test_f not in self.file_data
                        or ref_f not in self.file_data
                        or test_f == ref_f
                        or (self.conf.same_name_only
                            and (Path(test_f).name != Path(ref_f).name))
                        or (self.conf.ignore_leaf
                            and (Path(test_f).parent == Path(ref_f).parent))):
                    continue

                if (ref_f, test_f) in comparisons:
                    ref_idx, test_idx = comparisons[(ref_f, test_f)]
                    overlap = self.token_overlap_matrix[ref_idx, test_idx]
                    sim2, sim1 = self.similarity_matrix[ref_idx, test_idx]
                else:
                    overlap, (sim1, sim2), (slices1, slices2) = compare_files(
                        self.file_data[test_f], self.file_data[ref_f]
                    )
                    comparisons[(test_f, ref_f)] = (i, j)
                    if slices1.shape[0] != 0:
                        self.slice_matrix[(test_f, ref_f)] = [slices1, slices2]

                self.similarity_matrix[i, j] = np.array([sim1, sim2])
                self.token_overlap_matrix[i, j] = overlap

    def run(self):
        """Runs the copy detection loop for detecting overlap between
        test and reference files. If no files are in the provided
        directories, the similarity matrix will remain empty and any
        attempts to generate a report will fail.
        """
        if len(self.test_files) == 0:
            logging.error("Copy detector failed: No files found in "
                          "test directories")
        elif len(self.ref_files) == 0:
            logging.error("Copy detector failed: No files found in "
                          "reference directories")
        else:
            start_time = time.time()

            if not self.conf.silent:
                print("  0.00: Generating file fingerprints")

            self._preprocess_code(self.test_files + self.ref_files)

            if not self.conf.silent:
                print(f"{time.time()-start_time:6.2f}: Beginning code comparison")

            self._comparison_loop()

            if not self.conf.silent:
                print(f"{time.time()-start_time:6.2f}: Code comparison completed")

    def get_copied_code_list(self):
        """Get a list of copied code to display on the output report.
        Returns a list of tuples containing the similarity score, the
        test file name, the compare file name, the highlighted test
        code, and the highlighted compare code.

        Returns
        -------
        list
            list of similarity data between each file pair which
            achieves a similarity score above the display threshold,
            ordered by percentage of copying in the test file. Each
            element of the list contains [test similarity, reference
            similarity, path to test file, path to reference file,
            highlighted test code, highlighted reference code, numer of
            overlapping tokens]
        """
        if len(self.similarity_matrix) == 0:
            logging.error("Cannot generate code list: no files compared")
            return []
        x,y = np.where(self.similarity_matrix[:,:,0] > self.conf.display_t)

        code_list = []
        file_pairs = set()
        for idx in range(len(x)):
            test_f = self.test_files[x[idx]]
            ref_f = self.ref_files[y[idx]]
            if (ref_f, test_f) in file_pairs:
                # if comparison is already in report, don't add it again
                continue
            file_pairs.add((test_f, ref_f))

            test_sim = self.similarity_matrix[x[idx], y[idx], 0]
            ref_sim = self.similarity_matrix[x[idx], y[idx], 1]
            if (test_f, ref_f) in self.slice_matrix:
                slices_test = self.slice_matrix[(test_f, ref_f)][0]
                slices_ref = self.slice_matrix[(test_f, ref_f)][1]
            else:
                slices_test = self.slice_matrix[(ref_f, test_f)][1]
                slices_ref = self.slice_matrix[(ref_f, test_f)][0]

            if self.conf.truncate:
                truncate = 10
            else:
                truncate = -1
            hl_code_1, _ = highlight_overlap(
                self.file_data[test_f].raw_code, slices_test,
                "<span class='highlight-red'>", "</span>",
                truncate=truncate, escape_html=True)
            hl_code_2, _ = highlight_overlap(
                self.file_data[ref_f].raw_code, slices_ref,
                "<span class='highlight-green'>", "</span>",
                truncate=truncate, escape_html=True)
            overlap = self.token_overlap_matrix[x[idx], y[idx]]

            code_list.append([test_sim, ref_sim, test_f, ref_f,
                              hl_code_1, hl_code_2, overlap])

        code_list.sort(key=lambda x: -x[0])
        return code_list

    def generate_html_report(self, output_mode="save"):
        """Generates an html report listing all files with similarity
        above the display_threshold, with the copied code segments
        highlighted.

        Parameters
        ----------
        output_mode : {"save", "return"}
            If "save", the output will be saved to the file specified
            by self.out_file. If "return", the output HTML will be
            directly returned by this function.
        """
        if len(self.similarity_matrix) == 0:
            logging.error("Cannot generate report: no files compared")
            return

        code_list = self.get_copied_code_list()

        plot_mtx = np.copy(self.similarity_matrix[:,:,0])
        plot_mtx[plot_mtx == -1] = np.nan
        plt.imshow(plot_mtx)
        plt.colorbar()
        plt.tight_layout()
        sim_mtx_buffer = io.BytesIO()
        plt.savefig(sim_mtx_buffer)
        sim_mtx_buffer.seek(0)
        sim_mtx_base64 = base64.b64encode(sim_mtx_buffer.read()).decode()
        plt.close()

        scores=self.similarity_matrix[:,:,0][self.similarity_matrix[:,:,0]!=-1]
        plt.hist(scores, bins=20)
        plt.tight_layout()
        sim_hist_buffer = io.BytesIO()
        plt.savefig(sim_hist_buffer)
        sim_hist_buffer.seek(0)
        sim_hist_base64 = base64.b64encode(sim_hist_buffer.read()).decode()
        plt.close()

        # render template with jinja and save as html
        with importlib.resources.open_text(
            data_files, "report.html", encoding="utf-8"
        ) as template_fp:
            template = Template(template_fp.read())

        flagged = self.similarity_matrix[:,:,0] > self.conf.display_t
        flagged_file_count = np.sum(np.any(flagged, axis=1))

        formatted_conf = json.dumps(self.conf.to_json(), indent=4)
        output = template.render(config_params=formatted_conf,
                                 css_files=self.conf.css_files,
                                 version=__version__,
                                 test_count=len(self.test_files),
                                 test_files=self.test_files,
                                 compare_count=len(self.ref_files),
                                 compare_files=self.ref_files,
                                 flagged_file_count=flagged_file_count,
                                 code_list=code_list,
                                 sim_mtx_base64=sim_mtx_base64,
                                 sim_hist_base64=sim_hist_base64)

        if output_mode == "save":
            with open(self.conf.out_file, "w", encoding="utf-8") as report_f:
                report_f.write(output)

            if not self.conf.silent:
                print(
                    f"Output saved to {self.conf.out_file.replace('//', '/')}"
                )
            if self.conf.autoopen:
                webbrowser.open(
                    'file://' + str(Path(self.conf.out_file).resolve())
                )
        elif output_mode == "return":
            return output
        else:
            raise ValueError("output_mode not supported")
