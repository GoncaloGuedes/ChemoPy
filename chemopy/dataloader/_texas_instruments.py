import os
import glob
import pandas as pd


def load_texas_instruments_data(pathname: str, excel_name: str = None) -> pd.DataFrame:
    """
    Import data from Texas Instruments CSV files and return a pandas DataFrame.

    Args:
        pathname (str): Path to the directory containing the CSV files.
        csv_name (str, optional): Name of the output CSV file. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing the imported data.
    """
    csv_files = glob.glob(os.path.join(pathname, '*.csv'))
    data = []
    names = []

    for csv_file in csv_files:
        names.append(os.path.basename(csv_file).replace('.csv', ''))
        df_aux = pd.read_csv(csv_file, skiprows=21, encoding='cp1252')
        data.append(df_aux["Absorbance (AU)"].to_list())

    df = pd.DataFrame(data, columns=df_aux["Wavelength (nm)"].to_list())
    df.insert(0, "Name", names)

    if excel_name:
        csv_path = os.path.join(pathname, f'{excel_name}.xlsx')
        df.to_excel(csv_path, index=False)
    return df
        