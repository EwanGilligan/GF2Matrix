import numpy as np
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})
from GF2Matrix import IntMatrix
import unittest


class IntMatrixTester(unittest.TestCase):
    def test_initialisation(self):
        rows = 4
        columns = 5
        matrix = IntMatrix((rows, columns))
        self.assertEqual(matrix.rows, rows, "Row count not equal.")
        self.assertEqual(matrix.columns, columns, "Column count not equal.")
        self.assertEqual(matrix.data.shape[0], rows, "Data matrix not correct length.")


