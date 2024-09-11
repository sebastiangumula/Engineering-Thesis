# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 15:30:33 2021

@author: Sebastian Gumula

Decomposition of AirPassenger dataset

"""

from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("shampoo.csv", header=0, index_col=0)
result = seasonal_decompose(df, model = 'additive', period=12)

print(result.seasonal)

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
    if (i % 4) != 0:
        t.set_visible(False)
for i, t in enumerate(ax2.get_xticklabels()):
    if (i % 2) != 0:
        t.set_visible(False)
for i, t in enumerate(ax3.get_xticklabels()):
    if (i % 4) != 0:
        t.set_visible(False)
for i, t in enumerate(ax4.get_xticklabels()):
    if (i % 2) != 0:
        t.set_visible(False)     
        
fig.tight_layout(pad=1.0)
plt.show()