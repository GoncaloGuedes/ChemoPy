""" Test the Perkin Elmer data loader module. """

import os
import unittest

import pandas as pd

# replace with the actual module name
from chemopy.dataloader import load_texas_instruments_data


class TestLoadTexasInstrumentsData(unittest.TestCase):
    """Test the load_texas_instruments_data function.

    Parameters
    ----------
    unittest :
        test case class that is used to create new test cases.
    """

    def setUp(self):
        self.pathname = "chemopy/dataloader/tests/test_files/texas_instruments"
        self.excel_name = "Perkin_Elmer_Dataframe"
        self.expected_columns = ["Name", 900.837839, 904.733104, 909.920066]
        self.expected_values = [
            ["TI_1", 0.804932, 0.796678, 0.789913],
            ["TI_2", 1.034885, 1.035648, 1.024181],
        ]

    def test_load_data(self):
        """Test the load_texas_instruments_data function."""
        df = load_texas_instruments_data(self.pathname)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertListEqual(list(df.columns[:4]), self.expected_columns[:4])
        self.assertListEqual(df.iloc[:2, :4].values.tolist(), self.expected_values)

    def test_load_data_with_excel(self):
        """Test the load_texas_instruments_data function and save excel_name."""
        df = load_texas_instruments_data(self.pathname, self.excel_name)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertListEqual(list(df.columns[:4]), self.expected_columns)
        self.assertTrue(
            os.path.exists(os.path.join(self.pathname, f"{self.excel_name}.xlsx"))
        )

    def tearDown(self):
        if os.path.exists(os.path.join(self.pathname, f"{self.excel_name}.xlsx")):
            os.remove(os.path.join(self.pathname, f"{self.excel_name}.xlsx"))


if __name__ == "__main__":
    unittest.main()
