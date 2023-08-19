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
import os.path
import collections
import json

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from jinja2 import Template

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
    token_coverage : int
        The number of tokens in the tokenized code which are considered
        for fingerprint comparison, after dropping duplicate k-grams and
        performing winnowing.
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
        Path to HTML report file (deprecated).
    html_file : str
        Path to HTML report file (replaces out_file).
    pdf_file : str
        Path to PDF report file.
    csv_file : str
        Path to CSV report file.
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
                 truncate=False, out_file=None,
                 html_file="./report.html", pdf_file=None,
                 csv_file=None, silent=False, encoding: str = "utf-8"):
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
        start_time = time.time()
        if not self.conf.silent:
            print("  0.00: Generating file fingerprints")
        self._preprocess_code(self.test_files + self.ref_files)

        self.similarity_matrix = np.full(
            (len(self.test_files), len(self.ref_files), 2),
            -1,
            dtype=np.float64,
        )
        self.token_overlap_matrix = np.full(
            (len(self.test_files), len(self.ref_files)), -1
        )
        self.slice_matrix = {}

        if not self.conf.silent:
            print(f"{time.time()-start_time:6.2f}: Beginning code comparison")

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

        if not self.conf.silent:
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
            by self.html_file. If "return", the output HTML will be
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
        with open(data_dir + "report.html", encoding="utf-8") as template_fp:
            template = Template(template_fp.read())

        flagged = self.similarity_matrix[:,:,0] > self.conf.display_t
        flagged_file_count = np.sum(np.any(flagged, axis=1))

        formatted_conf = json.dumps(self.conf.to_json(), indent=4)
        output = template.render(config_params=formatted_conf,
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
            with open(self.conf.html_file, "w", encoding="utf-8") as report_f:
                report_f.write(output)

            if not self.conf.silent:
                print(
                    f"HTML report saved to {self.conf.html_file.replace('//', '/')}"
                )
            if self.conf.autoopen:
                webbrowser.open(
                    'file://' + str(Path(self.conf.html_file).resolve())
                )
        elif output_mode == "return":
            return output
        else:
            raise ValueError("output_mode not supported")

    _basename = None
    def basename(self, p):
        """Generate a short name for test/ref files.

        Parameters
        ----------
        p : a path to be shortened

        Returns
        -------
        A non-ambiguous shortened name for p
        """
        if self._basename is None:
            prefix = os.path.commonpath(self.test_files + self.ref_files)
            def basename(p):
                return str(Path(p).relative_to(prefix))
            self._basename = basename
        return self._basename(p)

    def _get_sim(self, neg, nan):
        """Return the similarity matrix as a pandas.DataFrame whose
        index and columns are the shortened file names.

        Parameters
        ----------
        neg : float
            value to use instead of -1 in the returned matrix
        nan : float
            value to use instead of NaN in the returned matrix

        Returns
        -------
        similarity matrix self.similarity_matrix[:,:,0] as a `pd.DataFrame`
        where negative values and nans have been replaced as expected
        """
        rows = [self.basename(f) for f in self.test_files]
        cols = [self.basename(f) for f in self.ref_files]
        sim = pd.DataFrame(data=np.copy(self.similarity_matrix[:,:,0]),
                           index=rows, columns=cols)
        sim[sim < 0] = neg
        sim.fillna(nan, inplace=True)
        return sim

    def generate_csv_report(self, neg=1.0, nan=0.0):
        """Save similarity matrix to a CSV file.

        Parameters
        ----------
        neg : float
            value to use instead of -1 in the returned matrix
        nan : float
            value to use instead of NaN in the returned matrix
        """
        sim = self._get_sim(neg, nan)
        sim.to_csv(self.conf.csv_file)
        if not self.conf.silent:
            print(f"CSV report saved to {str(self.conf.csv_file).replace('//', '/')}")

    def generate_pdf_report(self, minsim=.33, split=None, groups=None, **args):
        """Generate a clickable heatmap in a PDF file,
        clicks link to the corresponding match in HTML report.

        Parameters
        ----------
        minsim : float
            minimal similarity that is considered significant for
            heatmap simplifications. Basically, rows/cols that only have
            values < minsim are removed. Further simplifications occur
            when using split or groups parameters
        split : None or int
            if not None, the generated heatmaps is split into parts of at
            most this width/height. Splitting occurs on most distant clusters
            first, descending the dendrograms. Then, smallest chunks are gathered
            together to minimize the number of split heatmaps
        groups: {None, "auto", dict, (str)->str}
            if groups is a dict, it must map group names to lists or sets of
            filenames, if groups is a function, it must map file names
            to group names and it is then used to build the dict as explained.
            File names are passed in their shortened version (from basename),
            if groups returns None, full version is tried, and if groups still
            return None, file is placed in its own group named as the short name.
            If groups is "auto", then groups names are generated from the directory
            names in which the analysed files have been collected.
            When groups is not None, the generated heatmap is split by groups:
            for each group name, a heatmap whose rows is limited the group members
            is generated. If split is also specified, these heatmaps are further
            into smaller parts.
        args :
            arguments to the functions that perform the drawing
            'sns_A=V' call 'seaborn.clustermap' with arg 'A=V'
            'plt_A=V' call 'matplotlib.pyplot.savefig' with 'A=V'
        """
        # late import to speedup program startup when PDF is not generated
        import seaborn as sns
        from scipy.cluster.hierarchy import to_tree
        if not self.conf.silent:
            start_time = time.time()
            print("  0.00: Generating heatmaps")
        # extract args
        kw = {"sns": {"vmax": 1.0,
                      "vmin": 0.0,
                      "cmap": "RdYlBu_r"},
              "plt": {}}
        for key, val in args.items():
            if key[:3] in kw and key[3:4] == "_":
                kw[key[:3]][key[4:]] = val
            else:
                raise TypeError(f"unexpected argument {key!r}")
        kw["sns"].setdefault("xticklabels", 1)
        kw["sns"].setdefault("yticklabels", 1)
        # extract meaningful similarities
        sim = self._get_sim(-1, 0)
        small = (sim < minsim)
        sim.drop(index=sim.index[small.all(axis="columns")],
                 columns=sim.columns[small.all(axis="index")],
                 inplace=True)
        sim[sim < 0] = 1
        if len(sim.index) <= 1 or len(sim.columns) <= 1:
            logging.error("not enough row/cols left after cleanup")
            return
        # compute clickable links
        anchors = {}
        for i, (_, _, t, r, *_) in enumerate(self.get_copied_code_list(), start=1):
            k = (self.basename(t), self.basename(r))
            anchors[k] = anchors[k[::-1]] = i
        # compute groups
        if not groups:
            _groups = {}
        elif callable(groups):
            _groups = collections.defaultdict(set)
            for f in set(self.test_files) | set(self.ref_files):
                b = self.basename(f)
                _groups[groups(b) or groups(f) or b].add(b)
            _groups = dict(_groups)
        elif groups == "auto":
            _groups = collections.defaultdict(set)
            all_files = self.test_files + self.ref_files
            for f in all_files:
                b = self.basename(f)
                for g in reversed([str(p) for p in Path(b).parents]):
                    if g != ".":
                        break
                else:
                    g = ""
                _groups[g].add(b)
            _groups = dict(_groups)
        else:
            _groups = {k: set(self.basename(f) for f in v)
                       for k, v in groups.items()}
        if _groups:
            pal = sns.color_palette(n_colors=len(_groups))
            g2c = dict(zip(_groups, pal))
            f2c = {f: g2c[k] for k, v in _groups.items() for f in v}
            rc = sim.index.map(f2c.get)
            cc = sim.columns.map(f2c.get)
        else :
            rc = cc = None
        # generate heatmaps
        pdf = Path(self.conf.pdf_file)
        todo = []
        with tqdm(total=0, bar_format= '   {l_bar}{bar}{r_bar}',
                  disable=self.conf.silent) as log:
            # global heatmap
            def _add(p, h, **cm):
                todo.append({"path": p, "heatmap": h, "cm": cm})
                log.total= len(todo)
                log.refresh()
            _add(pdf, sns.clustermap(sim, row_colors=rc, col_colors=cc, **kw["sns"]))
            # group heatmaps
            for grp, members in sorted(_groups.items()):
                if len(members) <= 1:
                    continue
                sub = sim[sim.index.isin(members)]
                if len(sub.index) <= 1:
                    continue
                sub = sub.drop(columns=sub.columns[(sub < minsim).all(axis="index")])
                if len(sub.columns) <= 1:
                    continue
                if _groups:
                    rc = sub.index.map(f2c.get)
                    cc = sub.columns.map(f2c.get)
                if split and any(d > split for d in sub.shape):
                    _add(pdf.with_name(pdf.stem + f"-{grp}" + pdf.suffix),
                         sns.clustermap(sub, row_colors=rc, col_colors=cc, **kw["sns"]))
                else :
                    _add(pdf.with_name(pdf.stem + f"-{grp}" + pdf.suffix), None,
                         data=sub, row_colors=rc, col_colors=cc, **kw["sns"])
            # split heatmaps
            if split:
                for args in todo[:]:
                    path, hm = args["path"], args["heatmap"]
                    if hm is None:
                        continue
                    xt = to_tree(hm.dendrogram_col.linkage)
                    yt = to_tree(hm.dendrogram_row.linkage)
                    _split = self._split_matrix(hm.data, split, minsim, xt, yt)
                    for num, sub in enumerate(_split, start=1):
                        report = (path.parent / (path.stem + f"-{num}"))
                        report = report.with_suffix(path.suffix)
                        if _groups:
                            rc = sub.index.map(f2c.get)
                            cc = sub.columns.map(f2c.get)
                        _add(report, None,
                             data=sub, row_colors=rc, col_colors=cc, **kw["sns"])
            # draw heatmaps
            for args in todo:
                log.update()
                cm = args.pop("cm")
                if cm :
                    args["heatmap"] = sns.clustermap(**cm)
                self._draw_heatmap(**args, anchors=anchors, plt_kw=kw["plt"])
        if not self.conf.silent:
            print(f"{time.time()-start_time:6.2f}: Heatmaps generation completed")

    def _draw_heatmap(self, path, heatmap, anchors, plt_kw):
        """Draw a clickable heatmap and save it to path

        Parameters
        ----------
        path : pathlib.Path or str
            where to save the generated heatmap. Note that it should have a `.pdf`
            suffix otherwise it will be saved but the clickable links will be
            discarded
        heatmap:
            the heatmap to be drawn as returned by seaborn.clustermap()
        anchors : dict
           map pairs of col/row values to collapse div numbers in `self.html_file`
           for the generation of clickable links
        plt_kw : dict
            additional arguments passed to matplotlib.pyplot.savefig()
        """
        # configure axes
        height, width = heatmap.data.shape
        xfont = max(2, 350/width)
        if width > 40:
            plt.setp(heatmap.ax_heatmap.xaxis.get_majorticklabels(), fontsize=xfont)
        yfont = max(2, 350/height)
        if height > 40:
            plt.setp(heatmap.ax_heatmap.yaxis.get_majorticklabels(), fontsize=xfont)
        plt.setp(heatmap.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
        plt.setp(heatmap.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
        ax = heatmap.ax_heatmap
        xl = [t.label1.get_text() for t in ax.xaxis.get_major_ticks()]
        yl = [t.label1.get_text() for t in ax.yaxis.get_major_ticks()]
        # add clickable links
        text = "|" * max(1, round(2.8 * height / width))
        report = Path(self.conf.html_file).name
        for i, x in enumerate(xl):
            for j, y in enumerate(yl):
                if x == y:
                    continue
                a = anchors.get((x, y), None)
                if a is None:
                    continue
                ax.annotate(text, xy=(i+.5,j+.5), ha="center", va="center", alpha=0.0,
                            size=yfont,
                            url=f"{report}#collapse-{a}",
                            bbox={"visible": False,
                                  "url": f"{report}#collapse-{a}"})
        # save PDF
        heatmap.savefig(path, **plt_kw)
        plt.close()

    def _split_matrix(self, sim, split, minsim, xt, yt):
        """Split a similarity matrix into chunks of limited width/height

        Splitting is performed as follows:
         - the dendograms from a complete heatmap are descended to extract clusters
           smaller than split arg
         - these smaller chunks are then gathered together to limit the number of
           generated split heatmaps, while keeping the in the size limit
         - sub-matrices are extracted and their cols/rows that only have values < minsim
           are discarded

        Parameters
        ----------
        sim : pandas.DataFrame
            the similarity matrix to be split
        split : int
            the maximum width/height of generated chunks
        minsim : float
            as for generate_pdf_report
        xt : tree generated from the cols dendrogram
        yt : tree generated from the rows dendrogram

        Returns
        -------
        An iterator yielding the sub-matrices extracted from sim
        """
        height, width = sim.shape
        xchunks = self._binpack(self._split_tree(xt, split, sim.columns), split)
        ychunks = self._binpack(self._split_tree(yt, split, sim.index), split)
        for x in xchunks:
            for y in ychunks:
                xy = sim.loc[y,x]
                small = (xy < minsim)
                sub = xy.drop(index=xy.index[small.all(axis="columns")],
                              columns=xy.columns[small.all(axis="index")])
                h, w = sub.shape
                if w > 1 and h > 1 and (w < width or h < height):
                    yield sub

    def _split_tree(self, tree, split, names):
        """Split a dendogram into chunks of limited size

        Parameters
        ----------
        tree : tree generated from the dendogram
        split : maximum size of chunks
        names : array of leaf names

        Returns
        -------
        An iterator yielding lists of leaf names
        """
        todo = [tree]
        while todo:
            node = todo.pop()
            count = node.get_count()
            if 1 < count <= split:
                yield self._get_leaves(node, names)
            else:
                for child in (node.left, node.right):
                    if child and child.get_count() > 1:
                        todo.append(child)

    def _get_leaves(self, node, names):
        """Return the names of a sub-tree leaves

        Parameters
        ----------
        node : a sub-tree
        names : as for _split_tree

        Returns
        -------
        A list of leaf names extracted from arg names
        """
        def _leave(node):
            if node.is_leaf():
                return names[node.get_id()]
        return [n for n in node.pre_order(_leave) if n is not None]

    def _binpack(self, chunks, binsize):
        """Gather chunks into bigger groups of at most binsize items

        Implements first-fit-decreasing bin packing algorithm that
        usually gives a good approximation

        Parameters
        ----------
        chunks : iterable of chunks
            each chunk itself should be an iterable of str
        binsize : target size for gathered chunks
        """
        bins = []
        for c in sorted(chunks, key=len, reverse=True):
            for b in bins:
                if len(b) + len(c) <= binsize:
                    b.extend(c)
                    break
            else:
                bins.append(list(c))
        return bins
