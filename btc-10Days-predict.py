import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import math
import datetime


#BTC üzerinden yaptığımız analizde son 10 günlük verileri çıkartıp bu 10 gün olmadan tahmin yapın. Ondan sonra çıkarttığınız 10 günü ve tahmin ettiğiniz günleri iki ayrı grafik üstünde çizdirerek aradaki farkları inceleyin.
style.use('ggplot')

api_key='GyU-dbp-xstNxU35y6XD'
quandl.ApiConfig.api_key = api_key

df=quandl.get('BITFINEX/BTCUSD')
df.dropna(inplace=True)

df['HL_PCT'] = (df["High"] - df['Low']) / df['Last'] * 100.0  # High Low oranı
df['ASKBID_PCT'] = (df['Ask'] - df['Bid']) / df['Ask'] * 100.0  # Ask Bid oranı
df = df[['High', 'Low', 'Last', 'HL_PCT', 'ASKBID_PCT', 'Volume']]  # DataFrame görünecek parametreler.

dfPr=df.iloc[:-10,:].copy()
f_out= 10

dfPr['label']=dfPr['Last'].shift(-f_out)

x=dfPr.drop(columns='label')
x=scale(x)
y=dfPr.iloc[:,-1]

x_toPredict=x[-f_out:]
x=x[:-f_out]
y=y[:-f_out]

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)

regressor=LinearRegression()
regressor.fit(x_train,y_train)
Accuary=regressor.score(x_test,y_test)
print(Accuary)


prediction_set=regressor.predict(x_toPredict)
dfPr['prediction']=np.nan

last_date=dfPr.iloc[-1].name
last_datetime=last_date.timestamp()
one_day=86400
next_datetime=last_datetime+one_day

for i in prediction_set:
    next_date=datetime.datetime.fromtimestamp(next_datetime)
    next_datetime+=one_day
    dfPr.loc[next_date]=[np.nan for q in range(len(dfPr.columns)-1 )]+[i]

dfLast['Last'].plot(color='b')
dfPr['prediction'].plot(color='r')
plt.axis('auto') # grafiğe göre orlamak iiçin
plt.xlabel('Date')
plt.ylabel('USD/Price')
plt.legend(loc=4)
plt.show()