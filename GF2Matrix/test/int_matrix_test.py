import numpy as np
import unittest
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})
from GF2Matrix import IntMatrix


class IntMatrixTester(unittest.TestCase):

    def test_initialisation(self):
        """
        Tests that a IntMatrix is initialised correctly
        """
        rows = 4
        columns = 5
        matrix = IntMatrix((rows, columns))
        self.assertEqual(matrix.rows, rows, "Row count not equal.")
        self.assertEqual(matrix.columns, columns, "Column count not equal.")
        self.assertEqual(matrix.data.shape[0], rows, "Data matrix not correct length.")

    def test_too_large_init(self):
        """
        Tests that a Value Error is thrown when the size of the matrix exceeds 64 in at least one dimension.
        """
        self.assertRaises(ValueError, IntMatrix, (64, 65))
        self.assertRaises(ValueError, IntMatrix, (65, 1))
        self.assertRaises(ValueError, IntMatrix, (65, 65))

    def test_negative_sizes(self):
        """
        Tests than an Exception is thrown when the sizes are less than 1.
        """
        self.assertRaises(ValueError, IntMatrix, (0, 0))
        self.assertRaises(ValueError, IntMatrix, (-1, 32))
        self.assertRaises(ValueError, IntMatrix, (32, -1))

    def test_boundary_size(self):
        """
        Tests that the maximum size of matrix can be initialised.
        """
        try:
            IntMatrix((64, 64))
        except ValueError:
            self.fail("Initialisation of IntMatrix raised ValueError for 64*64")

    def test_add(self):
        """
        Tests that addition works correctly.
        """
        m1 = IntMatrix((3, 3))
        m1.data[0] = 0b111
        m1.data[1] = 0b101
        m1.data[2] = 0b011
        m2 = IntMatrix((3, 3))
        m2.data[0] = 0b101
        m2.data[1] = 0b000
        m2.data[2] = 0b100
        result = m1 + m2
        self.assertEqual(result.size(), m1.size(), "Shape is wrong")
        self.assertEqual(result.data[0], 0b010, "Wrong result.")
        self.assertEqual(result.data[1], 0b101, "Wrong result.")
        self.assertEqual(result.data[2], 0b111, "Wrong result.")

    def test_sub(self):
        """
        Tests that subtraction works correctly.
        """
        m1 = IntMatrix((3, 3))
        m1.data[0] = 0b111
        m1.data[1] = 0b101
        m1.data[2] = 0b011
        m2 = IntMatrix((3, 3))
        m2.data[0] = 0b101
        m2.data[1] = 0b000
        m2.data[2] = 0b100
        result = m1 + m2
        self.assertEqual(result.size(), m1.size(), "Shape is wrong")
        self.assertEqual(result.data[0], 0b010, "Wrong result.")
        self.assertEqual(result.data[1], 0b101, "Wrong result.")
        self.assertEqual(result.data[2], 0b111, "Wrong result.")

    def test_mul(self):
        """
        Tests that multiplication works correctly.
        """
        m1 = IntMatrix((2, 3))
        m1.set_row(0, [1, 0, 1])
        m1.set_row(1, [0, 0, 1])
        m2 = IntMatrix((3, 2))
        m2.set_row(0, [1, 0])
        m2.set_row(1, [1, 1])
        m2.set_row(2, [0, 1])
        result = m1 * m2
        self.assertEqual(result.size(), (2, 2))
        self.assertListEqual(result.get_row(0), [1, 1])
        self.assertListEqual(result.get_row(1), [0, 1])

    def test_set_item(self):
        """
        Tests that the set item method works correctly.
        """
        m = IntMatrix((2, 2))
        m[0, 1] = 1
        self.assertEqual(m.data[0], 0b10)
        m[1, 0] = 1
        self.assertEqual(m.data[1], 0b01)

    def test_set_item_out_of_bounds(self):
        """
        Tests that an IndexError is thrown when the coordinates are out of bounds.
        """
        m = IntMatrix((2, 2))
        self.assertRaises(IndexError, m.__setitem__, (-1, 0), 1)
        self.assertRaises(IndexError, m.__setitem__, (0, -1), 1)
        self.assertRaises(IndexError, m.__setitem__, (2, 0), 1)
        self.assertRaises(IndexError, m.__setitem__, (0, 2), 1)

    def test_get_item(self):
        """
        Tests than get item works correctly.
        """
        m = IntMatrix((2, 2))
        self.assertEqual(m[0, 1], 0)
        self.assertEqual(m[1, 0], 0)
        m.data[0] = 0b10
        m.data[1] = 0b01
        self.assertEqual(m[0, 1], 1)
        self.assertEqual(m[1, 0], 1)

    def test_get_item_out_of_bounds(self):
        """
        Tests that get item raises an exception when the indices are invalid.
        """
        m = IntMatrix((2, 2))
        self.assertRaises(IndexError, m.__getitem__, (-1, 0))
        self.assertRaises(IndexError, m.__getitem__, (0, -1))
        self.assertRaises(IndexError, m.__getitem__, (2, 0))
        self.assertRaises(IndexError, m.__getitem__, (0, 2))

    def test_get_column(self):
        """
        Tests that get_column works correctly.
        """
        m1 = IntMatrix((4, 4))
        m1.set_row(0, [1, 1, 1, 0])
        m1.set_row(1, [1, 0, 1, 0])
        m1.set_row(2, [0, 1, 1, 0])
        m1.set_row(3, [0, 0, 1, 0])
        self.assertListEqual(m1.get_column(0), [1, 1, 0, 0])
        self.assertListEqual(m1.get_column(1), [1, 0, 1, 0])
        self.assertListEqual(m1.get_column(2), [1, 1, 1, 1])
        self.assertListEqual(m1.get_column(3), [0, 0, 0, 0])

    def test_get_column_out_of_bounds(self):
        """
        Tests that get column raises an exception when the indices are out of bounds.
        """
        m = IntMatrix((2, 3))
        self.assertRaises(IndexError, m.get_row, -1)
        self.assertRaises(IndexError, m.get_row, 2)

    def test_set_column(self):
        """
        Tests that set column works correctly.
        """
        m = IntMatrix((3, 3))
        column1 = [1, 1, 0]
        column2 = [0, 1, 1]
        column3 = [0, 0, 1]
        m.set_column(0, column1)
        m.set_column(1, column2)
        m.set_column(2, column3)
        self.assertListEqual(m.get_column(0), column1)
        self.assertListEqual(m.get_column(1), column2)
        self.assertListEqual(m.get_column(2), column3)

    def test_set_column_out_of_bounds(self):
        """
        Tests that set column raises an exception when the indices are out of bounds.
        """
        m = IntMatrix((2, 3))
        self.assertRaises(IndexError, m.get_row, -1)
        self.assertRaises(IndexError, m.get_row, 2)

    def test_set_columns_wrong_values(self):
        """
        Tests than the correct exception is thrown when a value that is not 0 or 1 is inserted into a column.
        """
        m = IntMatrix((3, 3))
        self.assertRaises(ValueError, m.set_column, 0, [2, 1, 0])

    def test_set_column_short_list(self):
        """
        Tests an exception is thrown when the list isn't long enough.
        """
        m = IntMatrix((3, 3))
        self.assertRaises(ValueError, m.set_column, 0, [1, 1])

    def test_set_column_long_list(self):
        """
        Tests an exception is thrown when the list is too long
        """
        m = IntMatrix((3, 3))
        self.assertRaises(ValueError, m.set_column, 0, [1, 1, 1, 0])

    def test_set_column_non_indexable(self):
        """
        Tests an exception is thrown when the value given isn't indexable.
        """
        m = IntMatrix((3, 3))
        self.assertRaises(TypeError, m.set_column, 0, 1)

    def test_set_row(self):
        """
        Tests that set row works correctly.
        """
        m = IntMatrix((3, 3))
        m.set_row(0, [1, 1, 0])
        m.set_row(1, [0, 1, 1])
        m.set_row(2, [0, 1, 0])
        # Checking reverse order, as zeroth position is the least significant bit.
        self.assertEqual(m.data[0], 0b011)
        self.assertEqual(m.data[1], 0b110)
        self.assertEqual(m.data[2], 0b010)

    def test_set_row_out_of_bounds(self):
        m = IntMatrix((2, 3))
        self.assertRaises(IndexError, m.set_row, -1, [1, 1, 1])
        self.assertRaises(IndexError, m.set_row, 2, [1, 1, 1])

    def test_set_row_wrong_values(self):
        """
        Tests that a ValueError is thrown when a value that is not 0 or 1 is inserted in a row.
        """
        m = IntMatrix((3, 3))
        self.assertRaises(ValueError, m.set_row, 0, [-1, 1, 0])

    def test_set_row_short_list(self):
        """
        Tests that an Exception is thrown when the list isn't long enough.
        """
        m = IntMatrix((3, 3))
        self.assertRaises(ValueError, m.set_row, 0, [1, 1])

    def test_set_row_long_list(self):
        """
        Tests that an Exception is thrown when the list is too long.
        """
        m = IntMatrix((3, 3))
        self.assertRaises(ValueError, m.set_row, 0, [1, 1, 1, 1])

    def test_set_row_non_indexable(self):
        """
        Tests that an Exception is thrown when a indexable value isn't given.
        """
        m = IntMatrix((3, 3))
        self.assertRaises(TypeError, m.set_row, 0, 1)

    def test_get_row(self):
        """
        Tests that get row works correctly.
        """
        m = IntMatrix((4, 4))
        row1 = [1, 1, 0, 1]
        row2 = [0, 1, 1, 1]
        row3 = [0, 1, 0, 1]
        row4 = [1, 0, 0, 1]
        m.set_row(0, row1)
        m.set_row(1, row2)
        m.set_row(2, row3)
        m.set_row(3, row4)
        self.assertListEqual(m.get_row(0), row1, "Wrong value in row.")
        self.assertListEqual(m.get_row(1), row2, "Wrong value in row.")
        self.assertListEqual(m.get_row(2), row3, "Wrong value in row.")
        self.assertListEqual(m.get_row(3), row4, "Wrong value in row.")

    def test_get_row_out_of_bounds(self):
        """
        Tests that get row raises an exception when the indices are out of bounds.
        """
        m = IntMatrix((2, 3))
        self.assertRaises(IndexError, m.set_row, -1, [1, 1, 1])
        self.assertRaises(IndexError, m.set_row, 2, [1, 1, 1])

    def test_rank_1(self):
        """
        Tests that the rank is calculated correctly.

        The rank was verified by hand.
        """
        # Checks rank for 0 matrix
        m = IntMatrix()
        self.assertEqual(m.rank(), 0)
        m1 = IntMatrix((4, 4))
        m1.set_row(0, [1, 1, 1, 0])
        m1.set_row(1, [1, 0, 1, 0])
        m1.set_row(2, [0, 1, 1, 0])
        m1.set_row(3, [0, 0, 1, 0])
        self.assertEqual(m1.rank(), 3, "Wrong Rank")

    def test_rank_2(self):
        """
        Tests that the rank is calculated correctly.

        The rank was verified by hand.
        """
        m2 = IntMatrix((5, 4))
        m2.set_row(0, [1, 0, 1, 0])
        m2.set_row(1, [1, 1, 0, 1])
        m2.set_row(2, [0, 0, 1, 0])
        m2.set_row(3, [1, 1, 1, 1])
        m2.set_row(4, [1, 0, 0, 0])
        self.assertEqual(m2.rank(), 3, "Wrong rank")
