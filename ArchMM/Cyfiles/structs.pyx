# -*- coding: utf-8 -*-
#cython: boundscheck=False, wraparound=False, initializedcheck=True
#@PydevCodeAnalysisIgnore

from libc.stdio cimport *
cimport libc.math
cimport numpy as cnp
import numpy as np

from libc.stdio cimport *
from libc.stdlib cimport *

from cpython.object cimport PyObject, PyTypeObject
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cpython.method cimport PyMethod_Check
from cpython.buffer cimport *
from cpython.ref cimport Py_INCREF, Py_XDECREF
from cpython.tuple cimport PyTuple_Check


# https://github.com/cython/cython/blob/master/Cython/Includes/cpython/object.pxd
# https://jakevdp.github.io/blog/2014/05/05/introduction-to-the-python-buffer-protocol/
# https://github.com/cython/cython/wiki/enhancements-numpy-getitem

ctypedef object generic
cdef class LogStable_matrix

cdef __lsmatrix_csetitem__(LogStable_matrix self, object key, object item):
    if PyTuple_Check(key):
        self.data[key[0] * self.n_cols + key[1]] = item
    else:
        NotImplementedError()

cdef __lsmatrix_cgetitem__(LogStable_matrix this, object key):
    if PyTuple_Check(key):
        return this.data[key[0] * this.n_cols + key[1]]
    else:
        NotImplementedError()
        
cdef __lsmatrix_cadd__(LogStable_matrix matrix_A, LogStable_matrix matrix_B):
    assert(matrix_A.n_rows == matrix_B.n_rows and matrix_A.n_cols == matrix_B.n_cols)
    cdef LogStable_matrix new_matrix = LogStable_matrix(matrix_A.n_cols, matrix_A.n_rows)
    cdef Py_ssize_t i, j, position
    for i in range(matrix_A.n_rows):
        for j in range(matrix_A.n_cols):
            position = i * matrix_A.n_cols + j
            new_matrix.data[position] = matrix_A.data[position] + matrix_B.data[position]
    return new_matrix

cdef __lsmatrix_csub__(LogStable_matrix matrix_A, LogStable_matrix matrix_B):
    assert(matrix_A.n_rows == matrix_B.n_rows and matrix_A.n_cols == matrix_B.n_cols)
    cdef LogStable_matrix new_matrix = LogStable_matrix(matrix_A.n_cols, matrix_A.n_rows)
    cdef Py_ssize_t i, j, position
    for i in range(matrix_A.n_rows):
        for j in range(matrix_A.n_cols):
            position = i * matrix_A.n_cols + j
            new_matrix.data[position] = matrix_A.data[position] + matrix_B.data[position]
    return new_matrix

cdef class LogStable_matrix:
    cdef Py_ssize_t n_cols, n_rows
    cdef Py_ssize_t shape[2]
    cdef Py_ssize_t strides[2]
    cdef float* data
    cdef type dtype
    cdef __cythonbufferdefaults__ = {"ndim" : 2, "mode" : "c"}

    def __cinit__(self, Py_ssize_t n_cols, Py_ssize_t n_rows):
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.data = <float*>malloc(sizeof(float) * n_cols * n_rows)
        self.shape[0] = self.n_rows
        self.shape[1] = self.n_cols
        self.strides[1] = <Py_ssize_t>(<char*>&(self.data[1]) - <char*>&(self.data[0]))
        self.strides[0] = self.n_cols * self.strides[1]
        self.dtype = np.float32
        assert(PyObject_CheckBuffer(self))

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        
        cdef Py_ssize_t itemsize = sizeof(self.data[0])

        buffer.buf = <char*>&(self.data[0])
        buffer.format = 'f' # float
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = self.n_cols * self.n_rows * itemsize
        buffer.ndim = 2
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL
        
        if not PyBuffer_IsContiguous(buffer, 'C'): # or PyBUF_F_CONTIGUOUS
            raise BufferError("Buffer is not C contiguous")
        if PyBUF_ND | PyBUF_SIMPLE:
            raise BufferError("Error while getting buffer")
        
        Py_INCREF(self)

    def __releasebuffer__(self, Py_buffer *buffer):
        Py_XDECREF(<PyObject*>self)
        
    def __getitem__(generic self, object key):
        return __lsmatrix_cgetitem__(self, key)
    
    def __setitem__(generic self, object key, object item):
        __lsmatrix_csetitem__(self, key, item)  
    
    def copy(self, other):
        PyObject_CopyData(self, other)
    
alpha = LogStable_matrix(100, 100)
__lsmatrix_csetitem__(alpha, (54, 54), 78) 
beta  = LogStable_matrix(100, 100)
__lsmatrix_csetitem__(beta, (54, 54), 8) 
gamma = __lsmatrix_cadd__(alpha, beta)
print(__lsmatrix_cgetitem__(gamma, (54, 54))) 