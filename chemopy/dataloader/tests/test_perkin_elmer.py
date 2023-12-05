""" Test the Perkin Elmer data loader module. """
import os
import unittest

import pandas as pd

# replace with the actual module name
from chemopy.dataloader import load_perkin_elmer_data


class TestLoadPerkinElmerData(unittest.TestCase):
    """ Test the load_perkin_elmer_data function.

    Parameters
    ----------
    unittest : 
        test case class that is used to create new test cases.
    """

    def setUp(self):
        self.pathname = "chemopy/dataloader/tests/test_files/perkin_elmer"
        self.excel_name = "Perkin_Elmer_Dataframe"
        self.expected_columns = ['Names', 1000.0, 1000.5, 1001.0]
        self.expected_values = [['PE_1', 0.4428, 0.4398, 0.4384],
                                ['PE_2', 0.4367, 0.4384, 0.4403]]

    def test_load_data(self):
        """ Test the load_perkin_elmer_data function.
        """
        df = load_perkin_elmer_data(self.pathname)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertListEqual(list(df.columns[:4]), self.expected_columns[:4])
        self.assertListEqual(
            df.iloc[:2, :4].values.tolist(), self.expected_values)

    def test_load_data_with_excel(self):
        """ Test the load_perkin_elmer_data function with excel_name.
        """
        df = load_perkin_elmer_data(self.pathname, self.excel_name)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertListEqual(list(df.columns[:4]), self.expected_columns)
        self.assertTrue(os.path.exists(os.path.join(
            self.pathname, f"{self.excel_name}.xlsx")))

    def tearDown(self):
        if os.path.exists(os.path.join(self.pathname, f"{self.excel_name}.xlsx")):
            os.remove(os.path.join(self.pathname, f"{self.excel_name}.xlsx"))


if __name__ == "__main__":
    unittest.main()
