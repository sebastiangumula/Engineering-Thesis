# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 18:57:31 2022

@author: Sebastian Gumula

Stationarity test of shampoo dataset
"""

# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
df = pd.read_csv("cars.csv", header=0, index_col=0)

df_values = df.values     
  
diff = list()

for i in range(1, len(df_values)):
    value = df_values[i] - df_values[i-1]
    diff.append(value)

#Perform augmented Dickey-Fuller test
adf_test = adfuller(diff)

autocorrelation_plot(df)
plot_acf(df, adjusted=True)
plot_pacf(df, method="ols")


#Plot differentiated dataset
fig, ax = plt.subplots(1)

ax.set_title("Różnicowanie pierwszego stopnia")
ax.plot(diff)
for i, t in enumerate(ax.get_xticklabels()):
    if (i % 3) != 0:
        t.set_visible(False)
plt.xlabel("Miesiąc")
plt.ylabel("Ilość")

plt.show()

#Print ADF results
print("ADF Statistic:" ,adf_test[0])
print("p-value:" ,adf_test[1])
print("Critical values :" ,adf_test[4])
"""
result = seasonal_decompose(diff_2, model = 'additive', period=12)

fig, ((ax1, ax2,ax3, ax4)) = plt.subplots(4)
ax1.plot(result.observed)
ax1.set_title('Obserwacja')
ax2.plot(result.trend, color='green')
ax2.set_title('Składowa Trend-Cykl')
ax3.plot(result.seasonal, color='orange')
ax3.set_title('Składowa Sezonowa')
ax4.plot(result.resid, color='magenta')
ax4.set_title('Składowa Rezydualna')


for i, t in enumerate(ax1.get_xticklabels()):
    if (i % 30) != 0:
        t.set_visible(False)
for i, t in enumerate(ax2.get_xticklabels()):
    if (i % 30) != 0:
        t.set_visible(False)
for i, t in enumerate(ax3.get_xticklabels()):
    if (i % 20) != 0:
        t.set_visible(False)
for i, t in enumerate(ax4.get_xticklabels()):
    if (i % 20) != 0:
        t.set_visible(False)     
        
fig.tight_layout(pad=1.0)

plt.show()

"""



