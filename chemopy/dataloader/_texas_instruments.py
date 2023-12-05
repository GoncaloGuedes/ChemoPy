""" Module to import data from Texas Instruments CSV files."""
import glob
import os
from typing import Optional

import numpy as np
import pandas as pd


def load_texas_instruments_data(
    pathname: str, excel_name: Optional[str] = None, factory_reference: Optional[list] = None
) -> pd.DataFrame:
    """ Import data from Texas Instruments CSV files.

    Parameters
    ----------
    pathname : str
        Path to the folder containing the CSV files.
        Note: The CSV files must be all from Texas Instruments.
    excel_name : Optional[str], optional
        If given, the data will be saved as an excel file with the given name, 
        default is None.
    factory_reference : Optional[list], optional
        if given, the data will be converted to absorbance using the given reference, 
        default is None.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the data from the CSV files. The first column is the name of the file.
        The other columns are the data from the CSV files.
    """
    csv_files = glob.glob(os.path.join(pathname, "*.csv"))
    csv_files.sort()
    data = []
    names = []
    df_aux = pd.DataFrame()

    for csv_file in csv_files:
        names.append(os.path.basename(csv_file).replace(".csv", ""))
        df_aux = pd.read_csv(csv_file, skiprows=21, encoding="cp1252")

        if factory_reference is not None:
            reference = np.array(factory_reference)
            intensity = df_aux["Sample Signal (unitless)"].to_numpy()
            absorbance = np.log10((reference / intensity))
            absorbance = list(absorbance)
        else:
            absorbance = df_aux["Absorbance (AU)"].to_list()
        data.append(absorbance)
    df = pd.DataFrame(data, columns=df_aux["Wavelength (nm)"].to_list())
    df.insert(0, "Name", names)

    if excel_name:
        csv_path = os.path.join(pathname, f"{excel_name}.xlsx")
        df.to_excel(csv_path, index=False)
    return df
