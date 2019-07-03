import numpy as np
import cython
cimport numpy as np

DTYPE = np.uint64
ctypedef np.uint64_t DTYPE_t

class IntMatrix:

    # cdef DTYPE_t rows
    # cdef DTYPE_t columns


    def __init__(self, size=(64, 64)):
        self.rows = size[0]
        self.columns = size[1]
        self.data = np.zeros(self.rows, dtype=DTYPE)

    def __repr__(self):
        raise NotImplemented()


    def __add__(self, other):
        if self.rows != other.rows or self.columns != other.columns:
            raise ValueError("Dimensions don't match.")
        result = IntMatrix((self.rows, self.columns))
        for i in range(self.rows):
            result.data[i] = self.data[i] ^ self.data[i]
        return result

    def __mul__(self, other):
        pass

    def __sub__(self, other):
        # subtraction is the same as addition in GF(2)
        return self + other

    def __setitem__(self, coords, value):
        if value != 1 or value != 0:
            raise ValueError("Value must be from {1,0}.")
        row, column = coords
        self.data[row] |= value << column

    def __getitem__(self, coords):
        row, column = coords
        return (self.data[row] >> column) & 1

    def get_row(self, i) -> str:
        if i > self.rows:
            raise IndexError("Index out of range.")
        return [int(n) for n in np.binary_repr(self.data[i], self.columns)]

    def get_column(self, i):
        if i > self.columns:
            raise IndexError("Index out of range.")
        return [self.data[row] >> i for row in range(self.columns)]

    def __str__(self):
        string = ''
        for i in range(self.rows):
            string += np.binary_repr(self.data[i], self.columns) + "\n"
        return string

    def rank(self):
        rank = 0
        for col_mask in (1 << col for col in range(self.columns)):
            pivot_row = None
            rows, old_rows = [], self.data
            for row in old_rows:
                if not row & col_mask:
                    rows.append(row)
                elif pivot_row:
                    rows.append(row ^ pivot_row)
                else:
                    pivot_row = row
                    rank += 1
        return rank

