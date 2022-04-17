

import pandas as pd
import numpy as np
import sklearn
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

filepath = r"C:\Users\syafi\Downloads\diamonds\diamonds.csv"
diamonds = pd.read_csv(filepath,sep=',',header=0)

#%%

diamonds = diamonds.drop(['Unnamed: 0'],axis = 1)
diamonds_features = diamonds.copy()
diamonds_labels = diamonds_features.pop('price')

#%%
cut_categories=['Fair','Good','Very Good','Premium','Ideal']
color_categories=['J','I','H','G','F','E','D']
clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
ordinal_encoder = OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])
diamonds_features[['cut','color','clarity']] = ordinal_encoder.fit_transform(diamonds_features[['cut','color','clarity']])

#%%
from sklearn.model_selection import train_test_split

SEED = 12345
x_train,x_val,y_train,y_val = train_test_split(diamonds_features,diamonds_labels,test_size=0.2,random_state=SEED)
x_val,x_test,y_val,y_test = train_test_split(x_val,y_val,test_size=0.2,random_state=SEED)

standardizer = sklearn.preprocessing.StandardScaler()
standardizer.fit(x_train)
x_train = standardizer.transform(x_train)
x_val = standardizer.transform(x_val)
x_test = standardizer.transform(x_test)

# Data preparation is done

#%%

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape = x_train.shape[1]),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation="linear")
    ])

model.summary()

#%%

model.compile(optimizer='adam',loss='mse',metrics=['mse','mae'])

history = model.fit(x_train,y_train,validation_data=(x_val,y_val),batch_size=300,epochs=30)

#%%

import matplotlib.pyplot as plt

training_loss = history.history['loss']
val_loss = history.history['val_loss']
training_mae = history.history['mae']
val_mae = history.history['val_mae']
epochs = history.epoch

plt.plot(epochs,training_loss,label='Training loss')
plt.plot(epochs,val_loss,label='Validation loss')
plt.title('Training Loss vs Validation Loss')
plt.legend()
plt.figure()

plt.plot(epochs,training_mae,label='Training MAE')
plt.plot(epochs,val_mae,label='Validation MAE')
plt.title('Training MAE vs Validation MAE')
plt.legend()
plt.figure()

plt.show()

#%%

#Evaluate with test data for wild testing
test_result = model.evaluate(x_test,y_test,batch_size=300)
print(f"Test loss = {test_result[0]}")
print(f"Test MAE = {test_result[1]}")
print(f"Test MSE = {test_result[2]}")

#%%

predictions = np.squeeze(model.predict(x_test))

labels = np.squeeze(y_test)
plt.plot(predictions,labels,".")
plt.xlabel("Predictions")
plt.ylabel("labels")
plt.title(" Graph of Predictions vs Labels with Test Data")

plt.show()