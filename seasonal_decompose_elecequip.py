# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 21:28:48 2021

@author: Sebastian Gumula

STL decomposition and naive prediction of monthly manufacture of 
electrical equipment: computer, electronic and optical products.

"""

from statsmodels.tsa.seasonal import seasonal_decompose, STL
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot()
"""
result = seasonal_decompose(df, model = 'additive', period=12)


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
    if (i % 50) != 0:
        t.set_visible(False)
for i, t in enumerate(ax2.get_xticklabels()):
    if (i % 50) != 0:
        t.set_visible(False)
for i, t in enumerate(ax3.get_xticklabels()):
    if (i % 50) != 0:
        t.set_visible(False)
for i, t in enumerate(ax4.get_xticklabels()):
    if (i % 50) != 0:
        t.set_visible(False)     
        
fig.tight_layout(pad=1.0)

plt.show()
"""





