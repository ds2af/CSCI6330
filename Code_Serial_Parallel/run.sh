#!/bin/sh

python3 TS_Prediction_Serial.py -m LSTM > outputs/Serial_LSTM.out
python3 TS_Prediction_Serial.py -m BiLSTM > outputs/Serial_BiLSTM.out
python3 TS_Prediction_Serial.py -m GRU > outputs/Serial_GRU.out
python3 TS_Prediction_Serial.py -m Hybrid > outputs/Serial_Hybrid.out

mpirun -np 2 python3 TS_Prediction_DataParallel.py -m LSTM > outputs/MPI_2_LSTM.out
mpirun -np 2 python3 TS_Prediction_DataParallel.py -m BiLSTM > outputs/MPI_2_BiLSTM.out
mpirun -np 2 python3 TS_Prediction_DataParallel.py -m GRU > outputs/MPI_2_GRU.out
mpirun -np 2 python3 TS_Prediction_DataParallel.py -m Hybrid > outputs/MPI_2_Hybrid.out

mpirun -np 4 python3 TS_Prediction_DataParallel.py -m LSTM > outputs/MPI_4_LSTM.out
mpirun -np 4 python3 TS_Prediction_DataParallel.py -m BiLSTM > outputs/MPI_4_BiLSTM.out
mpirun -np 4 python3 TS_Prediction_DataParallel.py -m GRU > outputs/MPI_4_GRU.out
mpirun -np 4 python3 TS_Prediction_DataParallel.py -m Hybrid > outputs/MPI_4_Hybrid.out

mpirun -np 8 python3 TS_Prediction_DataParallel.py -m LSTM > outputs/MPI_8_LSTM.out
mpirun -np 8 python3 TS_Prediction_DataParallel.py -m BiLSTM > outputs/MPI_8_BiLSTM.out
mpirun -np 8 python3 TS_Prediction_DataParallel.py -m GRU > outputs/MPI_8_GRU.out
mpirun -np 8 python3 TS_Prediction_DataParallel.py -m Hybrid > outputs/MPI_8_Hybrid.out

mpirun -np 16 python3 TS_Prediction_DataParallel.py -m LSTM > outputs/MPI_16_LSTM.out
mpirun -np 16 python3 TS_Prediction_DataParallel.py -m BiLSTM > outputs/MPI_16_BiLSTM.out
mpirun -np 16 python3 TS_Prediction_DataParallel.py -m GRU > outputs/MPI_16_GRU.out
mpirun -np 16 python3 TS_Prediction_DataParallel.py -m Hybrid > outputs/MPI_16_Hybrid.out

mpirun -np 2 python3 TS_Prediction_ModelParallel_2LSTM-GRULSTM.py> outputs/MPI_2LSTM-GRULSTM.out