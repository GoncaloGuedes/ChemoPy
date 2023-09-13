import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from sklearn.pipeline import make_pipeline

from chemopy.dataloader import load_texas_instruments_data
from chemopy.preprocessing import *

df = load_texas_instruments_data(
    r"C:\Users\DP\OneDrive - Cork Supply Portugal\Ambiente de Trabalho\Grapes\Day_1",
    excel_name="Dataframe_FactoryRef",
    factory_reference=True,
)
