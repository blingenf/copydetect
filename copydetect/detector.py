"""This module contains functions for detecting overlap between
a set of test files (files to check for plagairism) and a set of
reference files (files that might have been plagairised from).
"""

from pathlib import Path
import time
import logging
import webbrowser
import pkg_resources
import io
import base64
import warnings

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from jinja2 import Template

from .utils import (filter_code, highlight_overlap, get_copied_slices,
                    get_document_fingerprints, find_fingerprint_overlap)
from . import defaults

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
    hashes : 1D array of ints
        List of fingerprints extracted from the filtered code.
    hash_idx : 1D array of ints
        List of indexes of the selected fingerprints. Used for
        translating hash indexes to indexes in the filtered code.
    k : int
        Value of provided k argument.
    language : str
        If set, will force the tokenizer to use the provided language
        rather than guessing from the file extension.
    """
    def __init__(self, file, k, win_size, boilerplate=[], filter=True,
                 language=None, fp=None):
        if fp is not None:
            code = fp.read()
        else:
            with open(file) as code_fp:
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
        similarity1 = len(idx1) / len(file1_data.hashes)
    else:
        similarity1 = 0
    if len(file2_data.filtered_code) > 0:
        similarity2 = len(idx2) / len(file2_data.hashes)
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
    config : dict
        Dictionary containing configuration parameters. Note that this
        uses the verbose version of each of the parameters listed
        below. If provided, parameters set in the configuration
        dictionary will overwrite default parameters and other
        parameters passed to the initialization function.

        Note that this parameter is deprecated and will be removed in a
        future version.
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
    silent : bool
        If true, all logging output will be supressed.
    """
    def __init__(self, config=None, test_dirs=[], ref_dirs=[],
                 boilerplate_dirs=[], extensions=["*"],
                 noise_t=defaults.NOISE_THRESHOLD,
                 guarantee_t=defaults.GUARANTEE_THRESHOLD,
                 display_t=defaults.DISPLAY_THRESHOLD,
                 same_name_only=False, ignore_leaf=False, autoopen=True,
                 disable_filtering=False, force_language=None,
                 truncate=False, out_file="./report.html", silent=False):
        if config is not None:
            # temporary workaround to ensure existing code continues
            # to work
            warnings.warn(
                "use CopyDetector.from_config to initialize CopyDetector from "
                "a config file. The config parameter is deprecated and will be"
                " removed in a future version.",
                DeprecationWarning, stacklevel=2
            )
            args = locals()
            del args["self"]
            del args["config"]
            self.__init__(**{**args, **self._read_config(config)})
            return

        self.silent = silent
        self.test_dirs = test_dirs
        if len(ref_dirs) == 0:
            self.ref_dirs = test_dirs
        else:
            self.ref_dirs = ref_dirs
        self.boilerplate_dirs = boilerplate_dirs
        self.extensions = extensions
        self.noise_t = noise_t
        self.guarantee_t = guarantee_t
        self.display_t = display_t
        self.same_name_only = same_name_only
        self.ignore_leaf = ignore_leaf
        self.autoopen = autoopen
        self.disable_filtering = disable_filtering
        self.force_language = force_language
        self.truncate = truncate
        self.out_file = out_file

        self._check_arguments()

        out_path = Path(self.out_file)
        if out_path.is_dir():
            self.out_file += "/report.html"
        elif out_path.suffix != ".html":
            self.out_file = str(out_path) + ".html"

        self.window_size = self.guarantee_t - self.noise_t + 1

        self.test_files = self._get_file_list(self.test_dirs, self.extensions)
        self.ref_files = self._get_file_list(self.ref_dirs, self.extensions)
        self.boilerplate_files = self._get_file_list(self.boilerplate_dirs,
                                                     self.extensions)

        # before run() is called, similarity data should be empty
        self.similarity_matrix = np.array([])
        self.token_overlap_matrix = np.array([])
        self.slice_matrix = np.array([])
        self.file_data = {}

    @staticmethod
    def _read_config(config: dict) -> dict:
        """Helper function for translating json configuration parameters
        to the arguments taken by __init__
        """
        config_param_mapping = {
            "noise_threshold": "noise_t",
            "guarantee_threshold": "guarantee_t",
            "display_threshold": "display_t",
            "test_directories": "test_dirs",
            "reference_directories": "ref_dirs",
            "boilerplate_directories": "boilerplate_dirs"
        }
        for conf_key, param_key in config_param_mapping.items():
            if conf_key in config:
                config[param_key] = config[conf_key]
                del config[conf_key]
        if "disable_autoopen" in config:
            config["autoopen"] = not config["disable_autoopen"]
            del config["disable_autoopen"]

        return config

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
        return cls(**cls._read_config(config))

    def _load_config(self, config):
        """Sets member variables according to a configuration
        dictionary.
        """
        self.noise_t = config["noise_threshold"]
        self.guarantee_t = config["guarantee_threshold"]
        self.display_t = config["display_threshold"]
        self.test_dirs = config["test_directories"]
        if "reference_directories" in config:
            self.ref_dirs = config["reference_directories"]
        if "extensions" in config:
            self.extensions = config["extensions"]
        if "boilerplate_directories" in config:
            self.boilerplate_dirs = config["boilerplate_directories"]
        if "force_language" in config:
            self.force_language = config["force_language"]
        if "same_name_only" in config:
            self.same_name_only = config["same_name_only"]
        if "ignore_leaf" in config:
            self.ignore_leaf = config["ignore_leaf"]
        if "disable_filtering" in config:
            self.disable_filtering = config["disable_filtering"]
        if "disable_autoopen" in config:
            self.autoopen = not config["disable_autoopen"]
        if "truncate" in config:
            self.truncate = config["truncate"]
        if "out_file" in config:
            self.out_file = config["out_file"]

    def _check_arguments(self):
        """type/value checking helper function for __init__"""
        if not isinstance(self.test_dirs, list):
            raise TypeError("Test directories must be a list")
        if not isinstance(self.ref_dirs, list):
            raise TypeError("Reference directories must be a list")
        if not isinstance(self.extensions, list):
            raise TypeError("extensions must be a list")
        if not isinstance(self.boilerplate_dirs, list):
            raise TypeError("Boilerplate directories must be a list")
        if not isinstance(self.same_name_only, bool):
            raise TypeError("same_name_only must be true or false")
        if not isinstance(self.ignore_leaf, bool):
            raise TypeError("ignore_leaf must be true or false")
        if not isinstance(self.disable_filtering, bool):
            raise TypeError("disable_filtering must be true or false")
        if not isinstance(self.autoopen, bool):
            raise TypeError("disable_autoopen must be true or false")
        if self.force_language is not None:
            if not isinstance(self.force_language, str):
                raise TypeError("force_language must be a string")
        if not isinstance(self.truncate, bool):
            raise TypeError("truncate must be true or false")
        if not isinstance(self.noise_t, int):
            if int(self.noise_t) == self.noise_t:
                self.noise_t = int(self.noise_t)
                self.window_size = int(self.window_size)
            else:
                raise TypeError("Noise threshold must be an integer")
        if not isinstance(self.guarantee_t, int):
            if int(self.guarantee_t) == self.guarantee_t:
                self.guarantee_t = int(self.guarantee_t)
                self.window_size = int(self.window_size)
            else:
                raise TypeError("Guarantee threshold must be an integer")

        # value checking
        if self.guarantee_t < self.noise_t:
            raise ValueError("Guarantee threshold must be greater than or "
                             "equal to noise threshold")
        if self.display_t > 1 or self.display_t < 0:
            raise ValueError("Display threshold must be between 0 and 1")
        if not Path(self.out_file).parent.exists():
            raise ValueError("Invalid output file path "
                "(directory does not exist)")

    def _get_file_list(self, dirs, exts):
        """Recursively collects list of files from provided
        directories. Used to search test_dirs, ref_dirs, and
        boilerplate_dirs
        """
        file_list = []
        for dir in dirs:
            for ext in exts:
                if ext == "*":
                    matched_contents = Path(dir).rglob("*")
                else:
                    matched_contents = Path(dir).rglob("*."+ext.lstrip("."))
                files = [str(f) for f in matched_contents if f.is_file()]

                if len(files) == 0:
                    logging.warning("No files found in " + dir)
                file_list.extend(files)

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
                fingerprint=CodeFingerprint(file, self.noise_t, 1,
                                            filter=not self.disable_filtering,
                                            language=self.force_language)
                boilerplate_hashes.extend(fingerprint.hashes)
            except UnicodeDecodeError:
                logging.warning(f"Skipping {file}: file not ASCII text")
                continue

        return np.unique(np.array(boilerplate_hashes))

    def _preprocess_code(self, file_list):
        """Generates a CodeFingerprint object for each file in the
        provided file list. This is where the winnowing algorithm is
        actually used.
        """
        boilerplate_hashes = self._get_boilerplate_hashes()
        for code_f in tqdm(file_list, bar_format= '   {l_bar}{bar}{r_bar}',
                           disable=self.silent):
            if code_f not in self.file_data:
                try:
                    self.file_data[code_f] = CodeFingerprint(
                        code_f, self.noise_t, self.window_size,
                        boilerplate_hashes, not self.disable_filtering,
                        self.force_language)

                except UnicodeDecodeError:
                    logging.warning(f"Skipping {code_f}: file not ASCII text")
                    continue

    def _comparison_loop(self):
        """The core code used to determine code overlap. The overlap
        between each test file and each compare file is computed and
        stored in similarity_matrix. Token overlap information and the
        locations of copied code are stored in slice_matrix and
        token_overlap_matrix, respectively.
        """
        start_time = time.time()
        if not self.silent:
            print("  0.00: Generating file fingerprints")
        self._preprocess_code(self.test_files + self.ref_files)

        self.similarity_matrix = np.full((
            len(self.test_files),len(self.ref_files),2), -1, dtype=np.float64)
        self.token_overlap_matrix = np.full((
            len(self.test_files), len(self.ref_files)), -1)
        self.slice_matrix = [[np.array([]) for _ in range(len(self.ref_files))]
                             for _ in range(len(self.test_files))]

        if not self.silent:
            print(f"{time.time()-start_time:6.2f}: Beginning code comparison")

        # this is used to track which files have been compared to avoid
        # unnecessary duplication when there is overlap between the
        # test and reference files
        comparisons = {}

        for i, test_f in enumerate(tqdm(self.test_files,
                bar_format= '   {l_bar}{bar}{r_bar}', disable=self.silent)):
            for j, ref_f in enumerate(self.ref_files):
                if (test_f not in self.file_data
                        or ref_f not in self.file_data
                        or test_f == ref_f
                        or (self.same_name_only
                            and (Path(test_f).name != Path(ref_f).name))
                        or (self.ignore_leaf
                            and (Path(test_f).parent == Path(ref_f).parent))):
                    continue

                if (ref_f, test_f) in comparisons:
                    ref_idx, test_idx = comparisons[(ref_f, test_f)]
                    overlap = self.token_overlap_matrix[ref_idx, test_idx]
                    sim2, sim1 = self.similarity_matrix[ref_idx, test_idx]
                    slices2, slices1 = self.slice_matrix[ref_idx][test_idx]
                else:
                    overlap, (sim1, sim2), (slices1, slices2) = compare_files(
                        self.file_data[test_f], self.file_data[ref_f])
                    comparisons[(test_f, ref_f)] = (i, j)

                self.similarity_matrix[i,j] = np.array([sim1, sim2])
                self.slice_matrix[i][j] = [slices1, slices2]
                self.token_overlap_matrix[i,j] = overlap

        if not self.silent:
            print(f"{time.time()-start_time:6.2f}: Code comparison completed")

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
            self._comparison_loop()

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
        x,y = np.where(self.similarity_matrix[:,:,0] > self.display_t)

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
            slices_test = self.slice_matrix[x[idx]][y[idx]][0]
            slices_ref = self.slice_matrix[x[idx]][y[idx]][1]

            if self.truncate:
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
        data_dir = pkg_resources.resource_filename('copydetect', 'data/')

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
        with open(data_dir + "report.html") as template_fp:
            template = Template(template_fp.read())

        flagged = self.similarity_matrix[:,:,0] > self.display_t
        flagged_file_count = np.sum(np.any(flagged, axis=1))

        output = template.render(test_count=len(self.test_files),
                                 compare_count=len(self.ref_files),
                                 flagged_file_count=flagged_file_count,
                                 code_list=code_list,
                                 sim_mtx_base64=sim_mtx_base64,
                                 sim_hist_base64=sim_hist_base64)

        if output_mode == "save":
            with open(self.out_file, "w") as report_f:
                report_f.write(output)

            if not self.silent:
                print(f"Output saved to {self.out_file.replace('//', '/')}")
            if self.autoopen:
                webbrowser.open('file://' + str(Path(self.out_file).resolve()))
        elif output_mode == "return":
            return output
        else:
            raise ValueError("output_mode not supported")
