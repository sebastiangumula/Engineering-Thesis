# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("AirPassengers.csv")

ax = plt.gca()

df.plot(kind='line',x='Month',y="#Passengers",ax=ax)



plt.show()

