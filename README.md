# copydetect

![Screenshot of copydetect code comparison output](https://raw.githubusercontent.com/blingenf/copydetect/master/docs/_static/report_screenshot1.png)

## Overview
Copydetect is a code plagiarism detection tool based on the approach proposed in "[Winnowing: Local Algorithms for Document Fingerprinting](http://theory.stanford.edu/~aiken/publications/papers/sigmod03.pdf)" and used for the popular [MOSS](https://theory.stanford.edu/~aiken/moss/) platform. Copydetect takes a list of directories containing code as input, and generates an HTML report displaying copied slices as output. The implementation takes advantage of fast numpy functions for efficient generation of results. Code tokenization is handled by [Pygments](https://pygments.org/), so all 500+ languages which pygments can detect and tokenize are in turn supported by copydetect.

Note that, like MOSS, copydetect is designed to detect likely instances of plagiarism; it is not guaranteed to catch cheaters dedicated to evading it, and it does not provide a guarantee that plagiarism has occurred.

## Installation
Copydetect can be installed using `pip install copydetect`. Note that Python version 3.6 or greater is required. You can then generate a report using the `copydetect` command (`copydetect.exe` on Windows. If your scripts folder is not in your PATH the code can also be run using `py.exe -m copydetect`).

## Usage
The simplest usage is `copydetect -t DIRS`, where DIRS is a space-separated list of directories to search for input files. This will recursively search for all files in the provided directories and compare every file with every other file. To look only at specific file extensions, use `-e` followed by another space-separated list (for example, `copydetect -t student_code -e cc cpp h`)

If the files you want to compare to are different from the files you want to check for plagiarism (for example, if you want to also compare to submissions from previous semesters), use `-r` to provide a list of reference directories. For example, `copydetect -t PA01_F20 -r PA01_F20 PA01_S20 PA01_F19`. To avoid matches with code that was provided to students, use `-b` to specify a list of directories containing boilerplate code.

There are several options for tuning the sensitivity of the detector. The noise threshold, set with `-n`, is the minimum number of matching characters between two documents that is considered plagiarism. Note that this is AFTER tokenization and filtering, where variable names have been replaced with V, function names with F, etc. If you change `-n` (default value: 25), you will also have to change the guarantee threshold, `-g` (default value: 30). This is the number of matching characters for which the detector is guaranteed to detect the match. If speed isn't an issue, you can set this equal to the noise threshold. Finally, the display threshold, `-d` (default value: 0.33), is used to determine what percentage of code similarity is considered interesting enough to display on the output report. The distribution of similarity scores is plotted on the output report to assist selection of this value.

There are several other command line options for different use cases. If you only want to check for "lazy" plagiarism (direct copying without changing variable names or reordering code), `-f` can be used to disable code filtering. If you don't want to compare files in the same leaf directory (for example, if code is split into per-student directories and you don't care about self plagiarism), use `-l`. For a complete list of configuration options, see the following section.

## Configuration Options
Configuration options can be provided either by using the command line arguments or by using a JSON file. If a JSON file is used, specify it on the command line using `-c` (e.g., `copydetect -c configuration.json`). A sample configuration file is available [here](docs/_static/sample.json). The following list provides the names of each JSON configuration key along with its associated command line arguments.

- `test_directories` (`-t`, `--test-dirs`): a list of directories to recursively search for files to check for plagiarism.
- `reference_directories` (`-r`, `--ref-dirs`): a list of directories to search for files to compare the test files to. This should generally be a superset of `test_directories`. If not provided, the test directories are used as reference directories.
- `boilerplate_directories` (`-b`, `--boilerplate-dirs`): a list of directories containing boilerplate code. Matches between fingerprints present in the boilerplate code will not be considered plagiarism.
- `extensions` (`-e`, `--extensions`): a list of file extensions containing code the detector should look at.
- `noise_threshold` (`-n`, `--noise-thresh`): the smallest sequence of matching characters between two files which should be considered plagiarism. Note that tokenization and filtering replaces variable names with `V`, function names with `F`, object names with `O`, and strings with `S` so the threshold should be lower than you would expect from the original code.
- `guarantee_threshold` (`-g`, `--guarantee-thresh`): the smallest sequence of matching characters between two files for which the system is guaranteed to detect a match. This must be greater than or equal to the noise threshold. If computation time is not an issue, you can set `guarantee_threshold = noise_threshold`.
- `display_threshold` (`-d`, `--display-thresh`): the similarity percentage cutoff for displaying similar files on the detector report.
- `same_name_only` (`-s`, `--same-name`): if `true`, the detector will only compare files that have the same name (for example, `decision_tree.py` will not be compared to `k_nn.py`). Note that this also means that, for example, `bryson_k_nn.py` will not be compared to `sara_k_nn.py`.
- `ignore_leaf` (`-l`, `--ignore-leaf`):  if `true`, the detector will not compare files located in the same leaf directory.
- `disable_filtering` (`-f`, `--disable-filter`):  if `true`, the detector will not tokenize and filter code before generating file fingerprints.
- `disable_autoopen` (`-a`, `--disable-autoopen`):  if `true`, the detector will not automatically open a browser window to display the report.

## API
For advanced use cases, copydetect provides an API for performing comparisons between files. An example of basic usage is provided below. API documentation is available [here](https://copydetect.readthedocs.io/en/latest/api.html).
```
>>> import copydetect
>>> fp1 = copydetect.CodeFingerprint("sample1.py", 25, 1)
>>> fp2 = copydetect.CodeFingerprint("sample2.py", 25, 1)
>>> token_overlap, similarities, slices = copydetect.compare_files(fp1, fp2)
>>> token_overlap # number of tokens in overlapping code
53
>>> similarities[0] # percentage of tokens in document 1 which were flagged
0.828125
>>> similarities[1] # percentage of tokens in document 2 which were flagged
0.8412698412698413
>>> code1, _ = copydetect.utils.highlight_overlap(fp1.raw_code, slices[0], ">>", "<<")
>>> code2, _ = copydetect.utils.highlight_overlap(fp2.raw_code, slices[1], ">>", "<<")
>>> print(code1)
def hashed_kgrams(string, k):
    """Return hashes of all k-grams in string"""
    >>hashes = [hash(string[offset:offset+k])
              for offset in range(len(string) - k + 1)]
    return np.array(hashes)<<

>>> print(code2)
def hash_f(s, k):
    >>h = [hash(s[o:o+k]) for o in range(len(s)-k+1)]
    return np.array(h)<<
```
