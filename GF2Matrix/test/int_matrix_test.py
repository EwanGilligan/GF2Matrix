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

    def test_boundary_size(self):
        try:
            IntMatrix()
        except ValueError:
            self.fail("Initialisation of IntMatrix raised ValueError for 64*64")

    def test_add(self):
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

    def test_set_row(self):
        m = IntMatrix((3, 3))
        m.set_row(0, [1, 1, 0])
        m.set_row(1, [0, 1, 1])
        m.set_row(2, [0, 1, 0])
        # Checking reverse order, as zeroth position is the least significant bit.
        self.assertEqual(m.data[0], 0b011)
        self.assertEqual(m.data[1], 0b110)
        self.assertEqual(m.data[2], 0b010)

    def test_get_row(self):
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

    def test_rank_1(self):
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
        m2 = IntMatrix((5, 4))
        m2.set_row(0, [1, 0, 1, 0])
        m2.set_row(1, [1, 1, 0, 1])
        m2.set_row(2, [0, 0, 1, 0])
        m2.set_row(3, [1, 1, 1, 1])
        m2.set_row(4, [1, 0, 0, 0])
        self.assertEqual(m2.rank(), 3, "Wrong rank")
