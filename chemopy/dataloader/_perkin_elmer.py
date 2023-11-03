""" Module to import data from Perkin Elmer CSV files."""
import glob
import os
from typing import Optional

import pandas as pd


def load_perkin_elmer_data(pathname: str, excel_name: Optional[str] = None) -> pd.DataFrame:
    """ Load data from Perkin Elmer CSV files.

    Parameters
    ----------
    pathname : str
        Path to the folder containing the CSV files.
        Note: The CSV files must be all from perkin elmer.
    excel_name : str, optional
       If given, the data will be saved as an excel file with the given name.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the data from the CSV files. The first column is the name of the file.
        The other columns are the data from the CSV files.
    """
    csv_files = glob.glob(os.path.join(pathname, "*.csv"))
    data = []
    names = []
    df_aux = pd.DataFrame()
    for csv_file in csv_files:
        names.append(os.path.basename(csv_file).replace(".csv", ""))
        df_aux = pd.read_csv(csv_file, skiprows=1)
        data.append(df_aux.iloc[:, 1].to_list())

    df = pd.DataFrame(data, columns=df_aux.iloc[:, 0].to_list())
    df.insert(0, "Name", names)

    if excel_name:
        csv_path = os.path.join(pathname, f"{excel_name}.xlsx")
        df.to_excel(csv_path, index=False)
    return df
