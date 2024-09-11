# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

load = pd.read_excel("commodity_prices.xlsx", sheet_name=0)

df = load[0:679]

headers = list(df.columns.values)

y = df.iloc[:,4:5]
x = df.iloc[:, 5:6]

ax = plt.gca()

df.plot(kind='line',x='Date',y='Value',ax=ax)



plt.show()

