from setuptools import setup, Extension

with open("README.md", "r") as readme_fp:
    readme = readme_fp.read()

# NOTE
# the C extension is not currently built for versions uploaded to PyPI.
# Speedup is not meaningful and it makes cross-platform support quite
# a bit more painful

setup(name="copydetect",
      author="Bryson Lingenfelter",
      author_email="blingenfelter@nevada.unr.edu",
      version="0.4.4",
      description="Code plagiarism detection tool",
      long_description=readme,
      long_description_content_type="text/markdown",
      url="https://github.com/blingenf/copydetect",
      packages=["copydetect"],
      ext_modules=[Extension("copydetect.winnow",
                             sources=["copydetect/winnow/winnow.c"],
                             optional=True)],
      install_requires=["numpy", "matplotlib", "jinja2", "pygments", "tqdm"],
      package_data={"copydetect" : ["data/*"]},
      python_requires=">=3.6",
      entry_points={"console_scripts" : [
          "copydetect = copydetect.__main__:main"]},
      classifiers=[
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Programming Language :: Python :: 3.9",
          "Programming Language :: Python :: 3.10",
          "Programming Language :: Python :: 3 :: Only",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
          "Development Status :: 4 - Beta",
          "Intended Audience :: Education"
      ])
