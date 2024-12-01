import os

# Suppress TensorFlow logs except for errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = info, 2 = warnings, 3 = errors

import warnings
# Suppress specific warnings, like Keras RNN input warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Do not pass an `input_shape`.*")

import argparse
from mpi4py import MPI
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from models import model_GRU, model_LSTM, model_Hybrid, model_BiLSTM
import time

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Set up argument parser (only on rank 0)
if rank == 0:
    parser = argparse.ArgumentParser(description='Choose a model type for training.')
    parser.add_argument('--useModel', '-m', type=str, required=True, choices=['LSTM', 'BiLSTM', 'GRU', 'Hybrid'],
                        help='Model type to use for training (LSTM, BiLSTM, GRU, or Hybrid)')
    args = parser.parse_args()
else:
    args = None

# Broadcast parsed arguments
args = comm.bcast(args, root=0)
useModel = args.useModel

# Define hyperparameters
time_step_predict = 15
epochs = 1000
batch_size = 16

# Load dataset and preprocess (each rank does its own loading to avoid data broadcasting)
# dataFolder = 'data/stockData/SPX_14-24.csv'
# inputColumns = ['Open', 'High', 'Low', 'Close']

dataFolder = 'data/stockData/SPX_14-20.csv'
inputColumns = ['Open', 'High', 'Low', 'Adj Close', 'Volume', 'Close']

target = ['Close']

dfOriginal = pd.read_csv(dataFolder)
dfOriginal['Date'] = pd.to_datetime(dfOriginal['Date'])
dfOriginal = dfOriginal.sort_values('Date')

scaler = StandardScaler()
columns_to_scale = inputColumns
df = pd.DataFrame(scaler.fit_transform(dfOriginal[columns_to_scale]), columns=columns_to_scale)

X, y = [], []
for i in range(len(df) - time_step_predict):
    X.append(df[columns_to_scale].iloc[i:i + time_step_predict].values)
    y.append(df['Close'].iloc[i + time_step_predict])

X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Split the training data among processes
subset_size = len(X_train) // size
start = rank * subset_size
end = (rank + 1) * subset_size if rank != size - 1 else len(X_train)
X_train_subset = X_train[start:end]
y_train_subset = y_train[start:end]

# Clear previous model and set up new model
tf.keras.backend.clear_session()
if useModel == "LSTM":
    model = model_LSTM(input_shape=(X_train.shape[1], X_train.shape[2]), units=50)
elif useModel == "BiLSTM":
    model = model_BiLSTM(input_shape=(X_train.shape[1], X_train.shape[2]), units=50)
elif useModel == "GRU":
    model = model_GRU(input_shape=(X_train.shape[1], X_train.shape[2]), units=50)
elif useModel == "Hybrid":
    model = model_Hybrid(input_shape=(X_train.shape[1], X_train.shape[2]), units=50)
else:
    comm.Abort(1)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model and collecting MSE histories
training_history = []
validation_history = []

# Start training timer
if rank == 0:
    start_time = time.time()

# Synchronize training parameters every `sync_every` epochs
sync_every = 5

for epoch in range(epochs):
    history = model.fit(X_train_subset, y_train_subset,
                        batch_size=batch_size,
                        epochs=1,
                        validation_split=0.1 if (epoch + 1) % sync_every == 0 else 0,  # Reduce validation frequency
                        verbose=2 if rank == 0 else 0,
                        shuffle=False)

    # Store training loss for the current epoch
    train_loss = history.history['loss'][0]

    # Collect training loss from all ranks
    all_train_losses = comm.gather(train_loss, root=0)

    if (epoch + 1) % sync_every == 0 or (epoch + 1) == epochs:
        # Synchronize weights every `sync_every` epochs
        local_weights = model.get_weights()
        avg_weights = [np.zeros_like(w) for w in local_weights]
        for i in range(len(local_weights)):
            comm.Allreduce(local_weights[i], avg_weights[i], op=MPI.SUM)
            avg_weights[i] /= size
        model.set_weights(avg_weights)

        # Collect validation loss if validated
        if 'val_loss' in history.history:
            val_loss = history.history['val_loss'][0]
            all_val_losses = comm.gather(val_loss, root=0)
        else:
            all_val_losses = None

        # Rank 0 collects and averages MSE values
        if rank == 0:
            avg_train_loss = np.mean(all_train_losses)
            training_history.append(avg_train_loss)

            if all_val_losses:
                avg_val_loss = np.mean(all_val_losses)
                validation_history.append(avg_val_loss)

# After all epochs, plot the MSE history on rank 0
if rank == 0:
    end_time = time.time()
    elapsed_time = end_time - start_time

    y_train_predict = model.predict(X_train)
    y_test_predict = model.predict(X_test)
    y_train_original = (y_train * scaler.scale_[-1]) + scaler.mean_[-1]
    y_test_original = (y_test * scaler.scale_[-1]) + scaler.mean_[-1]
    y_train_predict_original = (y_train_predict * scaler.scale_[-1]) + scaler.mean_[-1]
    y_test_predict_original = (y_test_predict * scaler.scale_[-1]) + scaler.mean_[-1]

    # Plot MSE over epochs
    test_mse = mean_squared_error(y_test, y_test_predict)

    # Print final MSE and time
    print(f"Model: MPI_{size}_{useModel}")
    print(f"Training Time: {elapsed_time:.2f} seconds")
    if len(training_history) > 0:
        print(f"Final Training MSE: {training_history[-1]:.4f}")
    if len(validation_history) > 0:
        print(f"Final Validation MSE: {validation_history[-1]:.4f}")
    print(f"Test MSE: {test_mse:.4f}")

    # Plot training results
    dates = dfOriginal['Date']
    dates_train = dates[time_step_predict: len(y_train) + time_step_predict]
    dates_test = dates[len(y_train) + time_step_predict:]

    plt.figure(figsize=(10, 6))
    plt.plot(dates_train, y_train_original, color='blue', label='Actual Close Prices (Train)')
    plt.plot(dates_train, y_train_predict_original, color='green', label='Predicted Train Close Prices')
    plt.plot(dates_test, y_test_original, color='blue')
    plt.plot(dates_test, y_test_predict_original, color='red', label='Predicted Test Close Prices')
    plt.title('Actual vs Predicted Close Prices')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.savefig(f"plots/MPI_{size}_{useModel}_ActualVsPredicted.png")
    plt.close()

    # Plot training and validation MSE
    plt.figure(figsize=(10, 6))
    plt.plot(training_history, label='Training MSE', color='blue')
    if len(validation_history) > 0:
        plt.plot(validation_history, label='Validation MSE', color='red')
    plt.title('Training and Validation MSE Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend()
    plt.savefig(f"plots/MPI_{size}_{useModel}_MSE.png")
    plt.close()
