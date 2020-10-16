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
Copydetect is a code plagiarism detection tool based the approach proposed in `Winnowing: Local Algorithms for Document Fingerprinting <http://theory.stanford.edu/~aiken/publications/papers/sigmod03.pdf>`_ and used for the popular `MOSS <https://theory.stanford.edu/~aiken/moss/>`_ platform. Unlike MOSS, copydetect is designed to run entirely on the user's machine rather than an external server. The implementation takes advantage of fast numpy functions for efficient generation of results. Code tokenization is handled by `Pygments <https://pygments.org>`_, so all 500+ languages which pygments can detect and tokenize are in turn supported by copydetect.

Note that, like MOSS, copydetect is designed to detect likely instances of plagiarism; it is not guaranteed to catch cheaters dedicated to evading it, and it does not provide a guarantee that plagiarism has occurred.

============
Installation
============
Copydetect can be installed via pip: ``pip install copydetect``. You can then generate a report using the ``copydetect`` command.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
