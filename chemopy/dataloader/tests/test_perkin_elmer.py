import os
import unittest

import pandas as pd

from chemopy.dataloader import load_perkin_elmer_data


class TestLoadPerkinElmerData(unittest.TestCase):
    def test_load_perkin_elmer_data(self):
        # Test data directory and files
        test_data_path = r"C:\Goncalo\ChemomPy\chemopy\dataloader\tests\test_data"
        csv_files = ["file1.csv", "file2.csv"]

        # Generate sample data files for testing
        for csv_file in csv_files:
            file_path = os.path.join(test_data_path, csv_file)
            df = pd.DataFrame({"wavelength": [0, 100, 101], "data": [0, 1, 2]})
            df.to_csv(file_path, index=False)

        # Call the function to load data and save CSV
        output_csv_name = "output"
        df = load_perkin_elmer_data(test_data_path, csv_name=output_csv_name)

        # Assert DataFrame column names and data
        self.assertEqual(df.columns.tolist(), ["Name", 100, 101])
        self.assertEqual(df["Name"].tolist(), ["file1", "file2"])
        self.assertEqual(df[100].tolist(), [1, 1])
        self.assertEqual(df[101].tolist(), [2, 2])

        # Assert that the output CSV file was saved
        output_csv_path = os.path.join(test_data_path, f"{output_csv_name}.csv")
        self.assertTrue(os.path.exists(output_csv_path))

        # Clean up the generated test data files and output CSV
        for csv_file in csv_files:
            file_path = os.path.join(test_data_path, csv_file)
            os.remove(file_path)
        os.remove(output_csv_path)


if __name__ == "__main__":
    unittest.main()
