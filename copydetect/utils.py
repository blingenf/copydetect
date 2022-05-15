"""This module contains functions for tokenizing/filtering code
as well as generic functions for detecting overlap between two
documents.
"""

import logging

from pygments import lexers, token
import pygments.util
import numpy as np
from markupsafe import escape

# if the C extention is available, use it. For almost all use cases
# the speed difference is not significant so if the C extention isn't
# found copydetect will silenty switch to the python implementation.
try:
    from .winnow import _winnow
except (ModuleNotFoundError, ImportError):
    from .pywinnow import _winnow

def filter_code(code, filename, language=None):
    """Tokenize and filter a code document. Replace variable names with
    V, function names with F, object names with O, and strings with S.
    Return the filtered document and a list of offsets indicating how
    many characters were removed by filtering at each index in the
    resulting document where filtering occured (this is used later to
    highlight the original code using plagiarism detection results on
    the filtered code)
    """
    try:
        if language is not None:
            lexer = lexers.get_lexer_by_name(language)
        else:
            lexer = lexers.get_lexer_for_filename(filename)
        tokens = lexer.get_tokens(code)
    except pygments.util.ClassNotFound:
        logging.warning(f"{filename} not tokenized: unknown file extension")
        return code, np.array([])

    if lexer == pygments.lexers.TextLexer:
        logging.warning(f"did not tokenize plaintext file {filename}")
        return code, np.array([])

    out_code = ""
    offset = 0
    offsets = [[0,0]]
    variable_tokens = {token.Name, token.Name.Variable, token.Name.Attribute}
    for t in tokens:
        if t[0] in variable_tokens:
            out_code += "V"
            offsets.append([len(out_code) - 1, offset])
            offset += len(t[1]) - 1
        elif t[0] in token.Name.Function:
            out_code += "F"
            offsets.append([len(out_code) - 1, offset])
            offset += len(t[1]) - 1
        elif t[0] in token.Name.Class:
            out_code += "O"
            offsets.append([len(out_code) - 1, len(t[1]) - 1])
            offset += len(t[1]) - 1
        elif t[0] == token.Comment.Preproc or t[0] == token.Comment.Hashbang:
            out_code += "P"
            offsets.append([len(out_code) - 1, offset])
            offset += len(t[1]) - 1
        elif t[0] in token.Text or t[0] in token.Comment:
            offsets.append([len(out_code) - 1, offset])
            offset += len(t[1])
        elif t[0] in token.Literal.String:
            if t[1] == "'" or t[1] == '"':
                out_code += '"'
            else:
                out_code += "S"
                offsets.append([len(out_code) - 1, offset])
                offset += len(t[1]) - 1
        else:
            out_code += t[1]
    return out_code, np.array(offsets)

def hashed_kgrams(string, k):
    """Return hashes of all k-grams in a string"""
    hashes = [hash(string[offset:offset+k])
              for offset in range(len(string) - k + 1)]
    return np.array(hashes)

def winnow(hashes, window_size, remove_duplicates=True):
    """implementation of the robust winnowing algorithm decribed in
    https://theory.stanford.edu/~aiken/publications/papers/sigmod03.pdf
    Returns a list of selected hashes and the indexes of those hashes.
    """
    if window_size < 1:
        raise ValueError("window_size must be greater than 0")

    # window size of 1 will just select all hashes
    if window_size == 1:
        selected_hashes = hashes
        selected_idx = np.arange(len(hashes))
    else:
        selected_idx = _winnow(hashes, window_size)
        selected_hashes = hashes[selected_idx]

    if remove_duplicates:
        selected_hashes, unique_idx = np.unique(selected_hashes,
                                                return_index=True)
        selected_idx = selected_idx[unique_idx]

    return selected_hashes, selected_idx

def get_copied_slices(idx, k):
    """Given k and a list of indexes detected by
    find_fingerprint_overlap, generates a list of slices where the
    copied code begins and ends. Returns a 2D array where the first
    dimension is slice start locations and the second dimension is
    slice end locations.
    """
    if len(idx) == 0:
        return np.array([[],[]])

    # determine the gaps between slices (called skips)
    sorted_idx = np.sort(idx)
    next_idx = np.concatenate([sorted_idx[1:], [0]])
    skips = np.where(next_idx - sorted_idx > k - 1)[0]

    # use the elements around the gaps to compute slice start/ends
    slice_starts = np.concatenate([[sorted_idx[0]], sorted_idx[skips + 1]])
    slice_ends = np.concatenate([sorted_idx[skips]+k, [sorted_idx[-1]+k]])

    return np.array([slice_starts, slice_ends])

def get_document_fingerprints(doc, k, window_size, boilerplate=[]):
    """Given a document, computes all k-gram hashes and uses the
    winnowing algorithm to reduce their number. Optionally takes a
    list of boilerplate hashes to remove from the winnowed list.
    Returns the selected hashes and their indexes in the original list
    """
    hashes, idx = winnow(hashed_kgrams(doc, k=k), window_size=window_size)
    if len(boilerplate) > 0:
        _, overlap_idx, _ = np.intersect1d(hashes, boilerplate,
                                           return_indices=True,
                                           assume_unique=True)
        idx = np.delete(idx, overlap_idx)
        hashes = np.delete(hashes, overlap_idx)
    return hashes, idx

def find_fingerprint_overlap(hashes1, hashes2, idx1, idx2):
    """Finds the indexes of overlapping values between two lists of
    hashes. Returns two lists of indexes, one for the first hash list
    and one for the second. The indexes of the original hashes are
    provided in case boilerplate results in gaps.
    """
    overlap, ol_idx1, ol_idx2 = np.intersect1d(hashes1, hashes2,
        return_indices=True, assume_unique=True)
    return idx1[ol_idx1], idx2[ol_idx2]

def highlight_overlap(doc, slices, left_hl, right_hl,
                      truncate=-1, escape_html=False):
    """Highlights copied code in a document given the slices containing
    copied code and strings to use for the highlight start and end.
    Returns the document annoted with the highlight strings as well as
    the percentage of code which was highlighted. If truncate is set to
    an integer, everything not within that many lines of highlighted
    code will be replaced with "..."
    """
    hl_percent = np.sum(slices[1] - slices[0])/len(doc)

    new_doc = ""
    current_idx = 0
    for slice_idx in range(slices.shape[1]):
        start_idx = slices[0,slice_idx]
        end_idx = slices[1,slice_idx]

        if escape_html:
            pre_highlight = str(escape(doc[current_idx:start_idx]))
            highlighted = left_hl+str(escape(doc[start_idx:end_idx]))+right_hl
        else:
            pre_highlight = doc[current_idx:start_idx]
            highlighted = left_hl + doc[start_idx:end_idx] + right_hl

        if truncate >= 0:
            lines = pre_highlight.split("\n")
            if slice_idx != 0 and len(lines) > truncate*2:
                pre_highlight = ("\n".join(lines[:truncate+1]) + "\n\n...\n\n"
                                 + "\n".join(lines[-truncate - 1:]))
            elif len(lines) > truncate:
                pre_highlight = "\n".join(lines[-truncate - 1:])

        new_doc += pre_highlight + highlighted
        current_idx = end_idx

    if escape_html:
        post_highlight = str(escape(doc[current_idx:]))
    else:
        post_highlight = doc[current_idx:]

    if truncate >= 0:
        lines = post_highlight.split("\n")
        if len(lines) > truncate:
            post_highlight = "\n".join(lines[:truncate])
    new_doc += post_highlight

    return new_doc, hl_percent
