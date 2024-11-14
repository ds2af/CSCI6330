# model_parallel_2lstm-gru-lstm.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import argparse
import datetime
import os
import traceback  # For exception traceback

from mpi4py import MPI
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, GRU
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Ensure we have exactly 2 processes
if size != 2:
    if rank == 0:
        print("This program requires exactly 2 MPI processes.")
    comm.Abort()

# Set up argument parser
# if rank == 0:
#     parser = argparse.ArgumentParser(description='Choose a model type for training.')
#     parser.add_argument('--useModel', '-m', type=str, required=True, choices=['LSTM', 'BiLSTM', 'GRU', 'Hybrid'],
#                         help='Model type to use for training (LSTM, BiLSTM, GRU, or Hybrid)')
#     args = parser.parse_args()
#     useModel = args.useModel
# else:
#     useModel = None
useModel = "2L-GL"

# Broadcast useModel to all ranks
useModel = comm.bcast(useModel, root=0)
# print(f"Rank {rank}: useModel = {useModel}", flush=True)  # Debugging

# Hyperparameters
dataFolder = 'data/stockData/SPX_10yr.csv'
inputColumns = ['Open', 'High', 'Low', 'Adj Close', 'Volume', 'Close']  # Keep target column at last
target = ['Close']
time_step_predict = 15  # For 15-day ahead prediction
epochs = 1000  # Reduced for debugging
batch_size = 16
learning_rate = 0.001

# Data preprocessing (only on rank 0)
if rank == 0:
    try:
        dfOriginal = pd.read_csv(dataFolder)
        dfOriginal['Date'] = pd.to_datetime(dfOriginal['Date'])
        dfOriginal = dfOriginal.sort_values('Date')

        # Standardizing the data
        scaler = StandardScaler()
        columns_to_scale = inputColumns
        df = pd.DataFrame(scaler.fit_transform(dfOriginal[columns_to_scale]), columns=columns_to_scale)

        # Preparing data
        X = []
        y = []
        for i in range(len(df) - time_step_predict):
            X.append(df[columns_to_scale].iloc[i:i + time_step_predict].values)
            y.append(df[target].iloc[i + time_step_predict])

        X = np.array(X)
        y = np.array(y)

        # print(f"Rank {rank}: Input Shape: {X.shape}", flush=True)
        # print(f"Rank {rank}: Output Shape: {y.shape}", flush=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        # print(f"Rank {rank}: Training Shape: {X_train.shape}, {y_train.shape}", flush=True)
        # print(f"Rank {rank}: Testing Shape: {X_test.shape}, {y_test.shape}", flush=True)

        y_train_original = (y_train * scaler.scale_[-1]) + scaler.mean_[-1]  # Inverse transform
        y_test_original = (y_test * scaler.scale_[-1]) + scaler.mean_[-1]  # Inverse transform
    except Exception as e:
        print(f"Rank {rank}: Exception during data preprocessing: {e}", flush=True)
        traceback.print_exc()
        comm.Abort()
else:
    X_train = X_test = y_train = y_test = None
    y_train_original = y_test_original = None
    scaler = None
    dfOriginal = None

# Broadcast data to all ranks
try:
    X_train = comm.bcast(X_train, root=0)
    X_test = comm.bcast(X_test, root=0)
    y_train = comm.bcast(y_train, root=0)
    y_test = comm.bcast(y_test, root=0)
    y_train_original = comm.bcast(y_train_original, root=0)
    y_test_original = comm.bcast(y_test_original, root=0)
    scaler = comm.bcast(scaler, root=0)
    dfOriginal = comm.bcast(dfOriginal, root=0)
except Exception as e:
    print(f"Rank {rank}: Exception during data broadcasting: {e}", flush=True)
    traceback.print_exc()
    comm.Abort()

# Define the model parts for each rank
try:
    if rank == 0:
        # Process 0: First LSTM layer
        model_part = Sequential()
        model_part.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
        model_part.add(LSTM(units=50, return_sequences=True))
        model_part.add(Dropout(0.2))
        model_part.add(LSTM(units=50, return_sequences=True))
        model_part.add(Dropout(0.2))
        # print(f"Rank {rank}: Model part defined.", flush=True)
    elif rank == 1:
        # Process 1: Second LSTM layer and output layers
        model_part = Sequential()
        model_part.add(Input(shape=(X_train.shape[1], 50)))
        model_part.add(GRU(units=50, return_sequences=True))
        model_part.add(Dropout(0.2))
        model_part.add(LSTM(units=50, return_sequences=False))
        model_part.add(Dropout(0.2))
        model_part.add(Dense(units=1))
        # print(f"Rank {rank}: Model part defined.", flush=True)
except Exception as e:
    print(f"Rank {rank}: Exception during model definition: {e}", flush=True)
    traceback.print_exc()
    comm.Abort()

# Define optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
mse_loss_fn = tf.keras.losses.MeanSquaredError()

# Training loop
history = {'loss': [], 'val_loss': []}
start_time = time.time()
for epoch in range(epochs):
    if rank == 0:
        print(f"Rank {rank}: Starting epoch {epoch+1}/{epochs}", flush=True)
    epoch_loss = []
    for batch_start in range(0, len(X_train), batch_size):
        batch_end = min(batch_start + batch_size, len(X_train))
        batch_size_actual = batch_end - batch_start

        # Prepare batch data
        X_batch = X_train[batch_start:batch_end]
        y_batch = y_train[batch_start:batch_end]

        # Forward pass
        try:
            if rank == 0:
                with tf.GradientTape() as tape:
                    output = model_part(X_batch, training=True)
                # Save output shape for backward pass
                output_shape = output.shape
                # print(f"Rank {rank}: Output shape: {output_shape}", flush=True)
                # Send output to rank 1
                comm.Send([output.numpy(), MPI.FLOAT], dest=1, tag=11)
                # print(f"Rank {rank}: Sent output to rank 1", flush=True)
                # Receive gradient from rank 1
                grad_output = np.empty_like(output.numpy())
                comm.Recv([grad_output, MPI.FLOAT], source=1, tag=12)
                # print(f"Rank {rank}: Received grad_output from rank 1", flush=True)
                # Backward pass
                gradients = tape.gradient(
                    output,
                    model_part.trainable_variables,
                    output_gradients=tf.convert_to_tensor(grad_output)
                )
                # Update weights
                optimizer.apply_gradients(zip(gradients, model_part.trainable_variables))
                # print(f"Rank {rank}: Updated weights", flush=True)
            elif rank == 1:
                # Receive output from rank 0
                input_shape = (batch_size_actual, X_train.shape[1], 50)
                input_data = np.empty(input_shape, dtype='float32')
                comm.Recv([input_data, MPI.FLOAT], source=0, tag=11)
                # print(f"Rank {rank}: Received input_data from rank 0 with shape {input_data.shape}", flush=True)
                input_data = tf.convert_to_tensor(input_data)
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(input_data)
                    predictions = model_part(input_data, training=True)
                    loss = mse_loss_fn(y_batch, predictions)
                # Compute gradients with respect to model variables
                gradients = tape.gradient(loss, model_part.trainable_variables)
                # Compute gradients with respect to input data
                grad_input = tape.gradient(loss, input_data)
                # Delete the tape to release resources
                del tape
                # Send gradient back to rank 0
                comm.Send([grad_input.numpy(), MPI.FLOAT], dest=0, tag=12)
                # print(f"Rank {rank}: Sent grad_input to rank 0", flush=True)
                # Update weights
                optimizer.apply_gradients(zip(gradients, model_part.trainable_variables))
                # print(f"Rank {rank}: Updated weights", flush=True)
                epoch_loss.append(loss.numpy())
                # print()
        except Exception as e:
            print(f"Rank {rank}: Exception during training loop: {e}", flush=True)
            traceback.print_exc()
            comm.Abort()

    # Synchronize before validation
    comm.Barrier()

    # **Add Validation Code Here on Rank 0**
    if rank == 0:
        try:
            # Validation data
            val_indices = int(len(X_train) * 0.9)
            X_val = X_train[val_indices:]
            # Forward pass through rank 0's model
            val_output = model_part(X_val, training=False)
            # Send val_output to rank 1
            comm.Send([val_output.numpy(), MPI.FLOAT], dest=1, tag=21)
            # print(f"Rank {rank}: Sent validation output to rank 1", flush=True)
        except Exception as e:
            print(f"Rank {rank}: Exception during validation: {e}", flush=True)
            traceback.print_exc()
            comm.Abort()

    # **Add Validation Code Here on Rank 1**
    if rank == 1:
        try:
            # Validation data
            val_indices = int(len(X_train) * 0.9)
            y_val = y_train[val_indices:]
            # Receive val_output from rank 0
            val_output_shape = (X_train.shape[0] - val_indices, X_train.shape[1], 50)
            input_data_val = np.empty(val_output_shape, dtype='float32')
            comm.Recv([input_data_val, MPI.FLOAT], source=0, tag=21)
            # print(f"Rank {rank}: Received validation input_data from rank 0 with shape {input_data_val.shape}", flush=True)
            input_data_val = tf.convert_to_tensor(input_data_val)
            # Forward pass through rank 1's model
            predictions_val = model_part(input_data_val, training=False)
            val_loss = mse_loss_fn(y_val, predictions_val).numpy()
            history['val_loss'].append(val_loss)
            # Average epoch loss
            avg_epoch_loss = np.mean(epoch_loss)
            history['loss'].append(avg_epoch_loss)
            print(f"Epoch: {epoch + 1}, Validation Loss: {avg_epoch_loss}")
            # print(f"Rank {rank}: Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.6f}, Val Loss: {val_loss:.6f}", flush=True)
        except Exception as e:
            print(f"Rank {rank}: Exception during validation: {e}", flush=True)
            traceback.print_exc()
            comm.Abort()

    # Synchronize after validation
    comm.Barrier()

end_time = time.time()
elapsed_time = end_time - start_time

# Evaluation (only on rank 1)
if rank == 1:
    try:
        # Predictions on training data
        # **Process training data through both models**

        # Rank 0 processes X_train and sends output to Rank 1
        if rank == 1:
            # Receive training output from rank 0
            train_output_shape = (X_train.shape[0], X_train.shape[1], 50)
            train_output = np.empty(train_output_shape, dtype='float32')
            comm.Recv([train_output, MPI.FLOAT], source=0, tag=31)
            # print(f"Rank {rank}: Received training data output from rank 0", flush=True)
            train_output = tf.convert_to_tensor(train_output)
            y_train_predict = model_part(train_output, training=False).numpy()
            y_train_predict_original = (y_train_predict * scaler.scale_[-1]) + scaler.mean_[-1]

        # Predictions on test data
        # Rank 0 processes X_test and sends output to Rank 1
        test_output_shape = (X_test.shape[0], X_test.shape[1], 50)
        test_output = np.empty(test_output_shape, dtype='float32')
        comm.Recv([test_output, MPI.FLOAT], source=0, tag=32)
        # print(f"Rank {rank}: Received test data output from rank 0", flush=True)
        test_output = tf.convert_to_tensor(test_output)
        y_test_predict = model_part(test_output, training=False).numpy()
        y_test_predict_original = (y_test_predict * scaler.scale_[-1]) + scaler.mean_[-1]

        # Compute test MSE
        test_mse = mean_squared_error(y_test, y_test_predict)
        # Print results
        print(f"Model: ModelParallel_{useModel}")
        print(f"Training Time: {elapsed_time:.2f} seconds")
        print(f"Final Training MSE: {history['loss'][-1]:.6f}")
        print(f"Final Validation MSE: {history['val_loss'][-1]:.6f}")
        print(f"Test MSE: {test_mse:.6f}")
        # Plotting results
        dates = dfOriginal['Date']
        dates_train = dates[time_step_predict:len(y_train_original) + time_step_predict]
        dates_test = dates[len(y_train_original) + time_step_predict:]

        # Plot actual vs predicted
        plt.figure(figsize=(10, 6))
        plt.plot(dates_train, y_train_original, color='blue', label='Actual Train Close Prices')
        plt.plot(dates_train, y_train_predict_original, color='green', label='Predicted Train Close Prices')
        plt.plot(dates_test, y_test_original, color='blue')
        plt.plot(dates_test, y_test_predict_original, color='red', label='Predicted Test Close Prices')
        plt.title('Actual vs Predicted Close Prices')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        if not os.path.exists('plots'):
            os.makedirs('plots')
        plt.savefig(f'plots/ModelHybrid_{useModel}_ActualVsPredicted.png')
        plt.close()

        # Plot training and validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(history['loss'], label='Training MSE', color='blue')
        plt.plot(history['val_loss'], label='Validation MSE', color='red')
        plt.title('Training and Validation MSE over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.legend()
        plt.savefig(f'plots/ModelHybrid_{useModel}_MSE.png')
        plt.close()
    except Exception as e:
        print(f"Rank {rank}: Exception during evaluation: {e}", flush=True)
        traceback.print_exc()
        comm.Abort()

# **Evaluation code on Rank 0 to send data to Rank 1**
if rank == 0:
    try:
        # Send processed training data to Rank 1
        train_output = model_part(X_train, training=False)
        comm.Send([train_output.numpy(), MPI.FLOAT], dest=1, tag=31)
        print(f"Rank {rank}: Sent training data output to rank 1", flush=True)

        # Send processed test data to Rank 1
        test_output = model_part(X_test, training=False)
        comm.Send([test_output.numpy(), MPI.FLOAT], dest=1, tag=32)
        print(f"Rank {rank}: Sent test data output to rank 1", flush=True)
    except Exception as e:
        print(f"Rank {rank}: Exception during evaluation data sending: {e}", flush=True)
        traceback.print_exc()
        comm.Abort()
