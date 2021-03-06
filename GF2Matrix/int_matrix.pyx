from functools import reduce
from typing import List

import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.uint64
ctypedef np.uint64_t DTYPE_t



class IntMatrix:
    """
    Class to represent a matrix with entries from GF(2).

    This implementation has a maximum size of 64*64 matrices. This is done for efficiency, as the rows of the matrices are stored as unsigned 64 bit integers.
    The 0th entry of each row corresponds to the least significant bit of the integer. Storing as integers allows for efficient operations using bitwise operators.
    """

    # cdef DTYPE_t rows
    # cdef DTYPE_t columns

    def __init__(self, size=(64, 64)):
        if not (0 < size[0] <= 64 and 0 < size[1] <= 64):
            raise ValueError("Matrix dimensions must be positive values less than or equal to 64.")
        self.rows = size[0]
        self.columns = size[1]
        self.data = np.zeros(self.rows, dtype=DTYPE)

    def __repr__(self):
        raise NotImplemented()

    def __add__(self, other):
        """
        Adds two matrices together.

        As this matrix is from GF2, the componentwise addition is performed by exclusive or (XOR).
        XOR is performed on each of the integers representing each row.

        :param other: Second operand of the addition operation.
        :return: IntMatrix representing the result of the addition.
        :rtype: IntMatrix
        """
        if self.rows != other.rows or self.columns != other.columns:
            raise ValueError("Dimensions don't match.")
        result = IntMatrix(self.size())
        _add(self.data, other.data, result.data)
        # for i in range(self.rows):
        #     result.data[i] = self.data[i] ^ other.data[i]
        return result

    def __mul__(self, other):
        """
        Multiplies two matrices together.

        :param other: Second operand of the multiplication operation.
        :return: IntMatrix representing the result of the addition.
        :rtype: IntMatrix
        """

        if self.columns != other.rows:
            raise ValueError("Dimension mismatch.")
        result = IntMatrix((self.rows, other.columns))
        for i in range(self.rows):
            for j in range(other.columns):
                result[i, j] = reduce(lambda x, y: x ^ y, map(lambda x, y: x & y, self.get_row(i), other.get_column(j)))
        return result

    def __sub__(self, other):
        """
        Subtracts one matrix from another.

        Since subtraction in GF(2) is the same as addition, then this is just a wrapper for addition; that is provided for convenience.
        :param other: Matrix to subtract from the first one.
        :return: IntMatrix representing the result of the addition.
        :rtype: IntMatrix
        """
        # subtraction is the same as addition in GF(2)
        return self + other

    def __setitem__(self, coords, value):
        """
        Sets the value at the specified coordinates to the value provided.


        :param coords: Tuple of positions (i,j) to set the element a_ij
        :param value: value to set a_ij to, which must be either 0 or 1.
        """

        if not (0 <= coords[0] < self.rows and 0 <= coords[1] < self.columns):
            raise IndexError()
        value = DTYPE(value)
        if not (value == 1 or value == 0):
            raise ValueError(value + " is not from {1,0}.")
        row, column = coords
        column = DTYPE(column)
        self.data[row] |= value << column

    def __getitem__(self, coords):
        """
        Returns the element located at the coordinated provided.

        :param coords: Tuple of positions (i,j) to return the element a_ij
        :return: The value at position (i,j), which will either be 0 or 1.
        """

        if not (0 <= coords[0] < self.rows and 0 <= coords[1] < self.columns):
            raise IndexError()
        row, column = coords
        return (self.data[row] >> DTYPE(column)) & DTYPE(1)

    def get_row(self, i):
        """
        Returns the ith row of the matrix.

        :param i: The index of the row to return.
        :return: List containing the values of the row.
        """

        if not 0 <= i < self.rows:
            raise IndexError("Index out of range.")
        return [DTYPE(int(n)) for n in reversed(np.binary_repr(self.data[i], self.columns))]

    def set_row(self, i, bitstring):
        """
        Set the ith row to the bitstring provided.
        :param i: The index of the row to set.
        :param bitstring: List of 1 and 0 to set the row to.
        """

        if len(bitstring) != self.columns:
            raise ValueError("List is of incorrect length.")
        if not 0 <= i < self.rows:
            raise IndexError()
        if bitstring.count(1) + bitstring.count(0) != len(bitstring):
            raise ValueError("Values must be from {0,1}")
        value = 0
        for bit in reversed(bitstring):
            value = (value << 1) | bit
        self.data[i] = value

    def get_column(self, i):
        """
        Returns the ith column of the matrix.

        :param i: The index of the column to return.
        :return: List containing the values of the column.
        """
        if not 0 <= i < self.columns:
            raise IndexError()
        i = DTYPE(i)
        return [(self.data[row] >> i) & DTYPE(1) for row in range(self.rows)]

    def set_column(self, i, bitstring):
        """
        Set the ith column to the bitstring provided.

        :param i: The index of the column to set.
        :param bitstring: List of 1 and 0 to set the column to.
        """

        if len(bitstring) != self.columns:
            raise ValueError("Bitstring too not correct length.")
        if not 0 <= i < self.columns:
            raise IndexError("Index out of range.")
        if bitstring.count(1) + bitstring.count(0) != len(bitstring):
            raise ValueError("Values must be from {0,1}")

        for j in range(self.rows):
            self[j, i] = bitstring[j]

    def __str__(self):
        """
        Get the String representation of the matrix.

        :return: String representation of the matrix.
        """
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
        """
        Returns a tuple containing the size of the matrix.

        :return: Tuple of the form (row count, column count).
        """
        return self.rows, self.columns

    def rank(self):
        """
        Calculate the rank of the matrix.

        This function modifies the input array.

        :return: The rank of the matrix.
        """
        return _rank(self.data)
        # rank = 0
        # rows = self.data
        # for col_mask in (DTYPE(1) << col for col in np.arange(0, self.columns, dtype=DTYPE)):
        #     pivot_row = None
        #     rows, old_rows = [], rows
        #     for row in old_rows:
        #         if not row & col_mask:
        #             rows.append(row)
        #         elif pivot_row:
        #             rows.append(row ^ pivot_row)
        #         else:
        #             pivot_row = row
        #             rank += 1
        # return rank

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef size_t _rank(DTYPE_t[:] rows):
    """
    Find rank of a matrix over GF2.

    This function modifies the input array.
    """
    cdef size_t i, j, nrows, rank
    cdef DTYPE_t pivot_row, row, lsb
    nrows = rows.shape[0]
    rank = 0
    for i in range(nrows):
        pivot_row = rows[i]
        if pivot_row:
            rank += 1
            lsb = pivot_row & -pivot_row
            for j in range(i + 1, nrows):
                row = rows[j]
                if row & lsb:
                    rows[j] = row ^ pivot_row
    return rank

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef  _add(DTYPE_t[:] m1, DTYPE_t[:] m2, DTYPE_t[:] result):
    nrows = result.shape[0]
    for i in range(nrows):
        result[i] = m1[i] ^ m2[i]


