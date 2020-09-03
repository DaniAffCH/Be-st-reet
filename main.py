import pandas as pd
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import OneHotEncoder
import time 
import datetime
import re
import urllib
import seaborn as sns

def OneHotToDf(df, field):
  onehot_encoder = OneHotEncoder(sparse=False)
  arr = np.array(df[field])
  arrReshaped = arr.reshape(len(arr), 1)
  field_OH = onehot_encoder.fit_transform(arrReshaped)

  new_columns=list()
  for col, values in zip(df[field], onehot_encoder.categories_):
    new_columns.extend([field + '_' + str(value) for value in values])

  new_df= pd.concat([df, pd.DataFrame(field_OH, columns=new_columns)], axis='columns')
  del new_df[field]

  return new_df

def fromDataToGoniometric(date):
  minutiSettimana = 7*24*60
  data = re.search("(\d{2})/(\d{2})/(\d{4}) (\d{2}):(\d{2})", date)
  unix_timestamp = datetime.datetime(int(data.group(3)),int(data.group(2)),int(data.group(1)),int(data.group(4)),int(data.group(5))).timestamp()
  myTimestamp = unix_timestamp - (24 * 4 * 60 * 60)
  myTimestamp /= 60
  return [np.sin( 2 * np.pi * myTimestamp / minutiSettimana ), np.cos( 2 * np.pi * myTimestamp / minutiSettimana )]

def spazio():
  print("\n======================================================================\n")
  
df = pd.read_csv("http://hoffnung.altervista.org/streetData.csv")
print("Suddivisione strade:")
sns.set_style('whitegrid')
sns.countplot(x = 'StradaID',hue="StradaID", data=df)
plt.legend(['Porta Romana', 'Ipercoop', 'Centro'])
plt.show()

spazio()

df = OneHotToDf(df, "StradaID")

# Encoding tempo come combinazione di seno/coseno (periodicità)

newCol = list()

for row in df["Data"]:
  newCol.append(fromDataToGoniometric(row))

df= pd.concat([df, pd.DataFrame(newCol, columns=["sin_periodo", "cos_periodo"])], axis='columns')
del df["Data"]

print(df.head())
spazio()
print(df.describe())
spazio()
print("Distribuzione periodo:")
df.plot.scatter('cos_periodo','sin_periodo').set_aspect('equal');
plt.show()

df.sin_periodo.plot()
df.cos_periodo.plot()

feature = ["StradaID_0", "StradaID_1", "StradaID_2", "sin_periodo", "cos_periodo"]
label = "Tempo"
def buildModel(my_learning_rate):
  model = keras.models.Sequential()
  layer = keras.layers.Dense(units=1)
  model.add(layer)
  model.compile(optimizer=keras.optimizers.RMSprop(lr=my_learning_rate),
                loss="mean_squared_error",
                metrics=[keras.metrics.RootMeanSquaredError()])
  return model

def fitModel(model, df, feature, label, batch_size, epochs):
  history = model.fit(x=df[feature],
                      y=df[label],
                      batch_size=batch_size,
                      epochs=epochs)

  trained_weight = model.get_weights()[0]
  trained_bias = model.get_weights()[1]

  epochs = history.epoch
  
  hist = pd.DataFrame(history.history)
 
  rmse = hist["root_mean_squared_error"]

  return trained_weight, trained_bias, epochs, rmse

lr = 1
batch = 1
epochs = 150
model = buildModel(lr)
# Side effect su model??
tr_weight, tr_bias, epochs, rmse = fitModel(model, df, feature, label, batch, epochs)
def plot_the_loss_curve(epochs, rmse):

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Root Mean Squared Error")

  plt.plot(epochs, rmse, label="Loss")
  plt.legend()
  plt.ylim([rmse.min()*0.97, rmse.max()])
  plt.show()  
plot_the_loss_curve(epochs, rmse)
print("Pesi stimati: {}\nBias stimato:{}".format(tr_weight, tr_bias))
street = {
    0:"Verità-Porta Romana-Cassia",
    1:"Ipercoop-Ponte Sodo",
    2:"Centro-Ponte Sodo"
}

sns.set_style('white')

streetImg=["http://hoffnung.altervista.org/cassia.png","http://hoffnung.altervista.org/iper.png","http://hoffnung.altervista.org/centro.png"]

date = str()
flag = True

while(flag):
  date = input("Data: ")
  if(re.search("(\d{2})/(\d{2})/(\d{4}) (\d{2}):(\d{2})", date) == None):
    print("Formato data errato [gg/mm/aaaa hh:mm]")
  else:
    flag = False

out = dict()

for c in range(3):
  data = [[0, 0, 0]+fromDataToGoniometric(date)]
  data[0][c] = 1
  print("DATA PER IL PREDICT: ")
  predict = model.predict(data)[0][0]
  predictFormatted = str(datetime.timedelta(seconds=int(predict)))[-5:]
  print(data[0])

  img = mpimg.imread(urllib.request.urlopen(streetImg[c]))
  imgplot = plt.imshow(img)
  plt.show()

  print(street[c]+", tempo stimato: "+predictFormatted+"\n")
  out[predict] = street[c]

keyOfBest = sorted(list(out.keys()))[0]
print("Strada migliore: {}".format(out[keyOfBest]))
