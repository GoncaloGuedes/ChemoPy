from chemopy.dataloader import load_texas_instruments_data
from chemopy.preprocessing import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
import scipy.signal


df = load_texas_instruments_data(r"C:\Users\DP\OneDrive - Cork Supply Portugal\Ambiente de Trabalho\Grapes\Day_3",
                                 )

x = df.iloc[:, 1:].to_numpy(dtype=np.float32)

x_preprocessed = scipy.signal.detrend(x)

preprocessing_pipeline = make_pipeline(ConvolutionSmoothing(kernel_size=5, keep_dims=False),
                                       Detrend()
                                       )
x_preprocessed = preprocessing_pipeline.fit_transform(x)
#plt.plot(x.T, 'b')
plt.plot(x_preprocessed.T)
plt.show()
print(x_preprocessed.shape)