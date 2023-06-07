# a python script to generate dose-response curves according to ll4 model

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
        # calculate R2 value
        residuals = y - ll4(x, popt[0], popt[1], popt[2], popt[3])
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

    # popt: optimal values for the parameters so that the sum of the squared error of ll4() and y is minimized
    # pcov: the estimated covariance of popt
    except:
        print('Error: curve fit failed - probably because the data does not fit the ll4 model.')

    return popt, r_squared

# example data
cwd = os.getcwd()
filename = input("Enter the name of the file (with extension): ")
try:
    df = pd.read_csv(cwd + '/' + filename, sep = ',')
except:
    df = pd.read_csv(cwd + '/' + filename, sep = '\t')


dose_column = input("Which one is the dose column? 1 is {} and 2 is {}: ".format(list(df.columns)[0], list(df.columns)[1]))
activity_column = input("Which one is the activity column? 1 is {} and 2 is {}: ".format(list(df.columns)[0], list(df.columns)[1]))

# from df, drop all rows with zero in the dose column
df = df[df[df.columns[int(dose_column)-1]] != 0]
# transform dose column to log10
df[df.columns[int(dose_column)-1]] = np.log10(df[df.columns[int(dose_column)-1]])

x = df[df.columns[int(dose_column)-1]]
y = df[df.columns[int(activity_column)-1]]


# fit dose-response curves to ll4 model
popt, r2 = fit_ll4(x, y)
hill_slope = popt[3]

xz = np.linspace(np.min(x), np.max(x), 100)
yz = ll4(xz, popt[0], popt[1], popt[2], popt[3])
plt.plot(xz, yz, '-'), plt.plot(x, y, 'o')
locs, labels = plt.xticks()
plt.xticks(locs, np.round(10 ** locs, 2))
plt.xlabel('Dose')
plt.ylabel('Response')
plt.title('Dose-response curve - EC50: {}, \nHills slope: {}, R\u00b2 of fit: {}'.format(np.round(10 ** popt[2], 2), np.round(hill_slope, 2), np.round(r2, 3)))
plt.show()
