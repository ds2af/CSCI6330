dataFolder = 'data/stockData/SPX_10yr.csv'
inputColumns = ['Open', 'High', 'Low', 'Adj Close', 'Volume','Close'] # keep target column at last
target=['Close']

import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Choose a model type for training.')
parser.add_argument('--useModel', '-m', type=str, required=True, choices=['LSTM', 'BiLSTM', 'GRU', 'Hybrid'],
                    help='Model type to use for training (LSTM, BiLSTM, GRU, or Hybrid)')

# Parse arguments from the command line
args = parser.parse_args()

# Get the model type from arguments
useModel = args.useModel



time_step_predict = 15 #for 15 day ahead prediction
epochs=1000
# useModel = "LSTM" # LSTM, BiLSTM, GRU or Hybrid
batch_size = 16

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error
import datetime
import time
from models import model_GRU, model_LSTM, model_Hybrid, model_BiLSTM

dfOriginal = pd.read_csv(dataFolder)
dfOriginal['Date'] = pd.to_datetime(dfOriginal['Date'])
dfOriginal = dfOriginal.sort_values('Date')



scaler = StandardScaler()
columns_to_scale = inputColumns
df = pd.DataFrame(scaler.fit_transform(dfOriginal[columns_to_scale]), columns=columns_to_scale)

X= []
y=[]
for i in range(len(df) - time_step_predict):
  X.append(df[columns_to_scale].iloc[i:i + time_step_predict].values)
  y.append(df[target].iloc[i+time_step_predict])

X = np.array(X)
y = np.array(y)

print("Input Shape:", X.shape)
print("Output Shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = False)
print("Training Shape:",X_train.shape, X_test.shape)
print("Testing Shape:",y_train.shape, y_test.shape)

y_train_original = (y_train *scaler.scale_[-1])+scaler.mean_[-1] # inverse transform of standard scalar
y_test_original = (y_test *scaler.scale_[-1])+scaler.mean_[-1] # inverse transform of standard scalar
start_time = time.time()

tf.keras.backend.clear_session()

if useModel == "LSTM":
    model = model_LSTM(input_shape =(X_train.shape[1], X_train.shape[2]) ,units=50)
elif useModel == "BiLSTM":
    model = model_BiLSTM(input_shape =(X_train.shape[1], X_train.shape[2]) ,units=50)
elif useModel == "GRU":
    model = model_GRU(input_shape =(X_train.shape[1], X_train.shape[2]) ,units=50)
elif useModel == "Hybrid":
    model = model_Hybrid(input_shape =(X_train.shape[1], X_train.shape[2]) ,units=50)
else:
    raise ValueError("Incorrect model selection. Choose LSTM, GRU, or Hybrid.")

# model.summary()
start_time = time.time()

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, shuffle = False,verbose=2)

end_time = time.time()

# Calculate elapsed time in seconds
elapsed_time = end_time - start_time

y_train_predict = model.predict(X_train)
y_train_predict_original = (y_train_predict *scaler.scale_[-1])+scaler.mean_[-1]

y_test_predict = model.predict(X_test)
y_test_predict_original = (y_test_predict *scaler.scale_[-1])+scaler.mean_[-1]

dates = dfOriginal['Date']
dates_train = dates[time_step_predict :len(y_train_original)+time_step_predict]
dates_test = dates[len(y_train_original)+time_step_predict:]

train_mses = history.history['loss']  # For training loss (MSE)
val_mses = history.history['val_loss']  # For validation loss (MSE)

test_mse = mean_squared_error(y_test, y_test_predict)


# Print final MSE and TIME
print(f"Model:Serial_'{useModel}")
print(f"Training Time: {elapsed_time:.2f} seconds")
print(f"Final Training MSE: {train_mses[-1]:.4f}")
print(f"Final Validation MSE: {val_mses[-1]:.4f}")
print(f"Test MSE: {test_mse:.4f}")


plt.figure(figsize=(10,6))
# Plot the actual training data
plt.plot(dates_train,y_train_original, color='blue', label='Actual Close Prices')

# Plot the predicted training data
plt.plot(dates_train,y_train_predict_original, color='green', label='Predicted Train Close Prices')

# Plot the actual test data - immediately following the train
plt.plot(dates_test, y_test_original, color='blue')

# Plot the predicted test data - immediately following the train
plt.plot(dates_test, y_test_predict_original, color='red', label='Predicted Test Close Prices')

plt.title('Actual vs Predicted Close Prices')
plt.xlabel('Date')
# plt.x_ticks(dates = dfOriginal['Date'])
plt.ylabel('Close Price')
plt.legend()
plt.savefig('plots/Serial_'+useModel+'_ActualVsPredicted.png')
plt.close()




# Plotting the MSE over epochs
plt.figure(figsize=(10, 6))

# Training MSE
plt.plot(train_mses, label='Training MSE', color='blue')

# Validation MSE
plt.plot(val_mses, label='Validation MSE', color='red')

# Setting the title and labels
plt.title('Training and Validation MSE over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.ylim(0,0.1)
plt.savefig('plots/Serial_'+useModel+'_MSE.png')
plt.close()
