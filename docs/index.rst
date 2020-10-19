Copydetect
======================================

========
Contents
========

.. toctree::
   :maxdepth: 2

   self
   cmdline
   api

========
Overview
========
Copydetect is a code plagiarism detection tool based on the approach proposed in `Winnowing: Local Algorithms for Document Fingerprinting <http://theory.stanford.edu/~aiken/publications/papers/sigmod03.pdf>`_ and used for the popular `MOSS <https://theory.stanford.edu/~aiken/moss/>`_ platform. Copydetect takes a list of directories containing code as input, and generates an HTML report displaying copied slices as output. The implementation takes advantage of fast numpy functions for efficient generation of results. Code tokenization is handled by `Pygments <https://pygments.org>`_, so all 500+ languages which pygments can detect and tokenize are in turn supported by copydetect.

Note that, like MOSS, copydetect is designed to detect likely instances of plagiarism; it is not guaranteed to catch cheaters dedicated to evading it, and it does not provide a guarantee that plagiarism has occurred.

============
Installation
============
Copydetect can be installed using ``pip install copydetect``. Note that Python version 3.6 or greater is required. You can then generate a report using the ``copydetect`` command (``copydetect.exe`` on Windows. If your scripts folder is not in your PATH the code can also be run using ``py.exe -m copydetect``).

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
