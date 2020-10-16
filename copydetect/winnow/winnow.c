#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject* winnow(PyObject *self, PyObject *args) {
    PyObject *input_arg = NULL;
    PyObject *input_arr = NULL;
    PyObject *output_idx = PyList_New(0);
    PyObject *output_arr = NULL;
    long window_size;

    if (!PyArg_ParseTuple(args, "O!l", &PyArray_Type,&input_arg, &window_size))
        return NULL;

    if (PyArray_DIMS(input_arg)[0] == 0) {
        output_arr = PyArray_FROM_OTF(output_idx, NPY_LONG,NPY_ARRAY_IN_ARRAY);
        Py_DECREF(output_idx);
        return output_arr;
    }

    #if NPY_API_VERSION >= 0x0000000c
        input_arr = PyArray_FROM_OTF(input_arg, NPY_LONG,
                                     NPY_ARRAY_INOUT_ARRAY2);
    #else
        input_arr = PyArray_FROM_OTF(input_arg, NPY_LONG,
                                     NPY_ARRAY_INOUT_ARRAY);
    #endif

    int len_list = PyArray_DIMS(input_arr)[0];
    long* data_ptr = (long*)PyArray_DATA(input_arr);

    // code adapted from page 9 of the winnowing paper
    // https://theory.stanford.edu/~aiken/publications/papers/sigmod03.pdf
    long* hash_buffer = (long*) malloc(window_size * sizeof(long));
    for (int i = 0; i < window_size; i++) {
        hash_buffer[i] = 9223372036854775807; // max_long
    }
    int r = 0; // right end of window
    int min_idx = 0; // index of minimum hash
    for (long list_idx = 0; list_idx < len_list; list_idx++) {
        r = (r + 1) % window_size; // shift window to right
        hash_buffer[r] = data_ptr[list_idx];

        if (min_idx == r) {
            for (int i = (r - 1 + window_size) % window_size; i != r;
                 i = (i - 1 + window_size) % window_size) {
                if (hash_buffer[i] < hash_buffer[min_idx]) {
                    min_idx = i;
                }
            }
            PyObject *next_element = PyLong_FromLong(
                list_idx - ((r + window_size - min_idx) % window_size));
            PyList_Append(output_idx, next_element);
            Py_DECREF(next_element);
        }
        else {
            if (hash_buffer[r] < hash_buffer[min_idx]) {
                min_idx = r;
                PyObject *next_element = PyLong_FromLong(list_idx);
                PyList_Append(output_idx, next_element);
                Py_DECREF(next_element);
            }
        }
    }
    free(hash_buffer);
    Py_DECREF(input_arr);

    output_arr = PyArray_FROM_OTF(output_idx, NPY_LONG, NPY_ARRAY_IN_ARRAY);
    Py_DECREF(output_idx);

    return output_arr;
}

static PyMethodDef WinnowMethods[] = {
    {"_winnow",  winnow, METH_VARARGS,
    "Winnow an input list, returning the indexes of the selected hashes"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyDoc_STRVAR(winnow_doc, "C extention for winnowing a list of hashes.");

static struct PyModuleDef winnowmodule = {
    PyModuleDef_HEAD_INIT,
    "winnow",
    winnow_doc,
    -1,
    WinnowMethods
};

PyMODINIT_FUNC PyInit_winnow(void) {
    import_array();
    return PyModule_Create(&winnowmodule);
}
