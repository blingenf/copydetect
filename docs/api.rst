API
======================================

For advanced use cases, copydetect provides an API for performing comparisons between files. An example of basic usage is provided below.

.. code-block:: python

    >>> import copydetect
    >>> fp1 = copydetect.CodeFingerprint("sample1.py", 25, 1)
    >>> fp2 = copydetect.CodeFingerprint("sample2.py", 25, 1)
    >>> token_overlap, similarities, slices = copydetect.compare_files(
    ...     fp1, fp2)
    >>> token_overlap
    53
    >>> similarities[0]
    0.828125
    >>> similarities[1]
    0.8412698412698413
    >>> code1, _ = copydetect.utils.highlight_overlap(
    ...     fp1.raw_code, slices[0], ">>", "<<")
    >>> code2, _ = copydetect.utils.highlight_overlap(
    ...     fp2.raw_code, slices[1], ">>", "<<")
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

========
Detector
========

.. automodule:: copydetect.detector
   :members:

=====
Utils
=====

.. automodule:: copydetect.utils
  :members:
