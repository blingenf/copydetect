from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, ClassVar
from pathlib import Path

from copydetect import defaults


@dataclass
class CopydetectConfig:
    """Utility class for providing a full list of configuration
    parameters with type/value checking and simple conversion to and from
    the JSON format used by the copydetect CLI.
    """

    test_dirs: List[str] = field(default_factory=lambda: [])
    ref_dirs: Optional[List[str]] = field(default_factory=lambda: [])
    boilerplate_dirs: Optional[List[str]] = field(default_factory=lambda: [])
    extensions: Optional[List[str]] = field(default_factory=lambda: ["*"])
    noise_t: int = defaults.NOISE_THRESHOLD
    guarantee_t: int = defaults.GUARANTEE_THRESHOLD
    display_t: float = defaults.DISPLAY_THRESHOLD
    same_name_only: bool = False
    ignore_leaf: bool = False
    autoopen: bool = True
    disable_filtering: bool = False
    force_language: Optional[str] = None
    truncate: bool = False
    out_file: str = "./report.html"
    css_files: List[str] = field(default_factory=lambda: [])
    silent: bool = False
    encoding: str = "utf-8"

    window_size: int = field(init=False, default=guarantee_t - noise_t + 1)
    short_names: ClassVar[Dict[str, str]] = {
        "noise_threshold": "noise_t",
        "guarantee_threshold": "guarantee_t",
        "display_threshold": "display_t",
        "test_directories": "test_dirs",
        "reference_directories": "ref_dirs",
        "boilerplate_directories": "boilerplate_dirs",
    }

    def _check_arguments(self):
        """Checks type/value of all parameters"""
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
        if not isinstance(self.css_files, list):
            raise TypeError("Linked CSS entries must be a list")
        
        # value checking
        if self.guarantee_t < self.noise_t:
            raise ValueError(
                "Guarantee threshold must be greater than or "
                "equal to noise threshold"
            )
        if self.display_t > 1 or self.display_t < 0:
            raise ValueError("Display threshold must be between 0 and 1")
        if not Path(self.out_file).parent.exists():
            raise ValueError(
                "Invalid output file path (directory does not exist)"
            )

    @staticmethod
    def normalize_outfile(file_path: str) -> str:
        """Ensures that the outfile has an html suffix. If the provided
        out file is a directory, append report.html to the path.
        """
        out_path = Path(file_path)
        if out_path.is_dir():
            file_path += "/report.html"
        elif out_path.suffix != ".html":
            file_path = str(out_path) + ".html"
        return str(file_path)

    def to_json(self) -> dict:
        """Converts the parameters of this configuration to the JSON
        format used for copydetect config files
        """
        dict_params = asdict(self)
        for long_name, short_name in self.short_names.items():
            dict_params[long_name] = dict_params[short_name]
            del dict_params[short_name]
        dict_params["disable_autoopen"] = not dict_params["autoopen"]
        del dict_params["autoopen"]
        if self.force_language is None:
            del dict_params["force_language"]
        return dict_params

    @staticmethod
    def normalize_json(config: dict) -> dict:
        """Converts the longer names used by the JSON configuration
        format to the arguments used by the CopyDetector class.
        """
        for long_name, short_name in CopydetectConfig.short_names.items():
            if long_name in config:
                config[short_name] = config[long_name]
                del config[long_name]
        if "disable_autoopen" in config:
            config["autoopen"] = not config["disable_autoopen"]
            del config["disable_autoopen"]
        return config

    def __post_init__(self):
        """Sets reference directories to test directories if needed and
        performs argument checking.
        """
        if len(self.ref_dirs) == 0:
            self.ref_dirs = self.test_dirs
        self.out_file = self.normalize_outfile(self.out_file)
        self.window_size = self.guarantee_t - self.noise_t + 1
        self._check_arguments()
