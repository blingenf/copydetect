"""This module contains the command line interface for copydetect. User
parameters are used to initialize a CopyDetector object, then the
detector is run and used to generate an output HTML report.
"""

import json
import argparse

from .detector import CopyDetector
from . import __version__
from . import defaults

def main():
    """main function for parsing command line arguments and running the
    detector
    """
    parser = argparse.ArgumentParser(prog="copydetect",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--conf", metavar="CONFIGURATION.JSON",
                        help="path to the JSON configuration file, if "
                        "using file configuration rather than command "
                        "line flags")
    parser.add_argument("-t", "--test-dirs", nargs='+',
                        metavar="TEST-DIRECTORY",
                        help="list of directories to recursively search for "
                             "files to check for plagairism")
    parser.add_argument("-r", "--ref-dirs", nargs='+',
                        metavar="REFERENCE-DIRECTORY",
                        help="list of directories to recursively search for "
                        "files to compare the test files to. If left empty, "
                        "the test directories themselves are used")
    parser.add_argument("-b", "--boilerplate-dirs", nargs='+',
                        metavar="BOILERPLATE-DIRECTORY", default=[],
                        help="list of directories to recursively search for "
                        "files containing boilerplate code")
    parser.add_argument("-e", "--extensions", nargs='+', default=["*"],
                        metavar="EXTENSIONS",
                        help="list of file extensions to collect code from")
    parser.add_argument("-n", "--noise-thresh",
                        default=defaults.NOISE_THRESHOLD, type=int,
                        metavar="NOISE-THRESHOLD",
                        help="length of minimum number of matching characters "
                        "which should be considered copied")
    parser.add_argument("-g", "--guarantee-thresh",
                        default=defaults.GUARANTEE_THRESHOLD, type=int,
                        metavar="GUARANTEE-THRESHOLD",
                        help="length of minimum number of matching characters "
                        "which the parser is guaranteed to detect as copied")
    parser.add_argument("-d", "--display-thresh",
                        default=defaults.DISPLAY_THRESHOLD, type=float,
                        metavar="DISPLAY-THRESHOLD",
                        help="percentage of copied code considered interesting"
                        " enough to display on the report")
    parser.add_argument("-o", '--force-language', dest='language',
                        default=None,
                        help="language to use for tokenization (if not "
                        "provided, the tokenizer guesses based on file "
                        "extensions)")
    parser.add_argument("-s", '--same-name', dest='same_name',
                        action='store_true', default=False,
                        help="only compare files which have the same name")
    parser.add_argument("-l", '--ignore-leaf', dest='ignore_leaf',
                        action='store_true', default=False,
                        help="don't compare files located in the same "
                        "leaf directory")
    parser.add_argument("-f", '--disable-filter', dest='filter',
                        action='store_true', default=False,
                        help="disable code tokenization and filtering")
    parser.add_argument("-a", '--disable-autoopen', dest='autoopen',
                        action='store_true', default=False,
                        help="disable browser autoopen")
    parser.add_argument("-T", '--truncate', dest='truncate',
                        action='store_true', default=False,
                        help="truncate non-copied portions of highlighted "
                        "code")
    parser.add_argument("-O", '--out-file', dest='out_file',
                        default="./report.html",
                        help="path to save output report to. A '.html' "
                        "extension will be added to the path if not provided. "
                        "If a directory is provided instead of a file, the "
                        "report will be saved  to that directory as "
                        "report.html.")
    parser.add_argument('--css', nargs="+",
                        metavar="CSS-FILE", default=[], dest="css_files",
                        help="Optional list of CSS files that will be linked "
                        "in the generated HTML report file. These will "
                        "overwrite the styling of the default report.")
    parser.add_argument('--version', action='version',
                        version="copydetect v" + __version__,
                        help="print version number and exit")
    parser.add_argument("--encoding", default="utf-8",
                        help="encoding to use for reading files. If files use "
                        "varying encodings, --encoding DETECT can be used to "
                        "detect the encoding of all files (requires the "
                        "chardet package)")
    args = parser.parse_args()

    if args.conf:
        with open(args.conf, encoding="utf-8") as json_fp:
            config = json.load(json_fp)
    elif args.test_dirs:
        if not args.ref_dirs:
            args.ref_dirs = args.test_dirs
        config = {
          "test_directories" : args.test_dirs,
          "reference_directories" : args.ref_dirs,
          "boilerplate_directories" : args.boilerplate_dirs,
          "extensions" : args.extensions,
          "noise_threshold" : args.noise_thresh,
          "guarantee_threshold" : args.guarantee_thresh,
          "display_threshold" : args.display_thresh,
          "force_language" : args.language,
          "same_name_only" : args.same_name,
          "ignore_leaf" : args.ignore_leaf,
          "disable_filtering" : args.filter,
          "disable_autoopen" : args.autoopen,
          "truncate" : args.truncate,
          "out_file" : args.out_file,
          "css_files": args.css_files,
          "encoding": args.encoding,
        }
    else:
        parser.error("either a path to a configuration file (-c) or a "
                     "list of test directories (-t) must be provided.")

    # get overlapping code
    detector = CopyDetector.from_config(config)
    detector.run()
    detector.generate_html_report()

if __name__ == "__main__":
    main()
