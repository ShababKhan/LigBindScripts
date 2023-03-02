# a python script to generate dose-response curves according to ll4 model

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# define the ll4 model
def ll4(x, a, b, c, d):
    '''x: dose
       a: minimum response
       b: maximum response
       c: EC50
       d: Hill slope'''
    return a + (b - a) / (1 + (x / c) ** d)

# fit dose-response curves to ll4 model
def fit_ll4(x, y):
    '''x: dose
       y: response
       '''
    try:
        popt, pcov = curve_fit(ll4, x, y, maxfev=10000)
    # popt: optimal values for the parameters so that the sum of the squared error of ll4() and y is minimized
    # pcov: the estimated covariance of popt
    except:
        print('Error: curve fit failed - probably because the data does not fit the ll4 model.')

    return popt

# example data
df = pd.read_csv('example-data.csv')
x = df['Dose/uM']
y = df['Activity/%']
print(df)



# fit dose-response curves to ll4 model
popt = fit_ll4(x, y)

print(popt)

xz = np.linspace(np.min(x), np.max(y), 100)
yz = ll4(xz, popt[0], popt[1], popt[2], popt[3])
plt.plot(xz, yz, '-'), plt.plot(x, y, 'o')
plt.title('Dose-response curve - EC50: {}'.format(np.round(popt[2])) )
plt.show()