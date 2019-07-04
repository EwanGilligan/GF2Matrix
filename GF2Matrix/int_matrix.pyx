from functools import reduce
from typing import List

import numpy as np
import cython
cimport numpy as np

DTYPE = np.uint64
ctypedef np.uint64_t DTYPE_t

cdef int rank(np.ndarray[DTYPE_t, ndim=1] rows, ncols):
    cdef int rank = 0
    cdef DTYPE_t pivot_row
    cdef np.ndarray[DTYPE_t, ndim=1] old_rows
    for col_mask in (1 << col for col in range(ncols)):
        pivot_row = 0
        old_rows = rows
        rows = np.ndarray(1, dtype=DTYPE)
        for row in old_rows:
            if not row & col_mask:
                np.append(rows, row)
            elif pivot_row:
                np.append(rows, row ^ pivot_row)
            else:
                pivot_row = row
                rank += 1
    return rank


class IntMatrix:

    # cdef DTYPE_t rows
    # cdef DTYPE_t columns

    def __init__(self, size=(64, 64)):
        if size[0] > 64 or size[1] > 64:
            raise ValueError("Maximum matrix size is 64*64")
        self.rows = size[0]
        self.columns = size[1]
        self.data = np.zeros(self.rows, dtype=DTYPE)

    def __repr__(self):
        raise NotImplemented()

    def __add__(self, other):
        if self.rows != other.rows or self.columns != other.columns:
            raise ValueError("Dimensions don't match.")
        result = IntMatrix(self.size())
        for i in range(self.rows):
            result.data[i] = self.data[i] ^ other.data[i]
        return result

    def __mul__(self, other):
        if self.columns != other.rows:
            raise ValueError("Dimension mismatch.")
        result = IntMatrix((self.rows, other.columns))
        for i in range(self.rows):
            for j in range(other.columns):
                print(self.get_row(i))
                print(other.get_column(j))
                result[i, j] = reduce(lambda x, y: x ^ y, map(lambda x, y: x & y, self.get_row(i), other.get_column(j)))
        return result

    def __sub__(self, other):
        # subtraction is the same as addition in GF(2)
        return self + other

    def __setitem__(self, coords, value):
        value = DTYPE(value)
        if not (value == 1 or value == 0):
            raise ValueError("Value must be from {1,0}.")
        row, column = coords
        column = DTYPE(column)
        self.data[row] |= value << column

    def __getitem__(self, coords):
        row, column = coords
        return (self.data[row] >> column) & 1

    def get_row(self, i):
        if i > self.rows:
            raise IndexError("Index out of range.")
        return [DTYPE(int(n)) for n in reversed(np.binary_repr(self.data[i], self.columns))]

    def set_row(self, i, bitstring):
        if len(bitstring) != self.columns:
            raise ValueError("Bitstring too not correct length.")
        if i > self.rows:
            raise IndexError("Index out of range.")
        if not isinstance(bitstring, List):
            raise ValueError("List not given.")
        if bitstring.count(1) + bitstring.count(0) != len(bitstring):
            raise ValueError("Values must be from {0,1}")
        value = 0
        for bit in reversed(bitstring):
            value = (value << 1) | bit
        self.data[i] = value

    def get_column(self, i):
        if i > self.columns:
            raise IndexError("Index out of range.")
        i = DTYPE(i)
        return [(self.data[row] >> i) & DTYPE(1) for row in range(self.rows)]

    def __str__(self):
        string = '<'
        separator = ''
        for i in range(self.rows):
            # making sure to reverse, as the least significant bit is at position 0.
            string += separator + " ".join(np.binary_repr(self.data[i], self.columns)[::-1])
            separator = '\n '
        return string + '>'

    def __copy__(self):
        copy = IntMatrix(self.size)
        for i in range(self.rows):
            copy.data[i] = self.data[i]
        return copy

    def size(self):
        return self.rows, self.columns

    def rank(self):
        rank = 0
        rows = self.data
        for col_mask in (DTYPE(1) << col for col in np.arange(0, self.columns, dtype=DTYPE)):
            pivot_row = None
            rows, old_rows = [], rows
            for row in old_rows:
                if not row & col_mask:
                    rows.append(row)
                elif pivot_row:
                    rows.append(row ^ pivot_row)
                else:
                    pivot_row = row
                    rank += 1
        return rank
