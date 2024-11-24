# CSCI6330
Parallel Processing
### Introduction
Training neural networks on large datasets often requires a lot of time and computational resources. This is especially true for complex RNN models, which are widely used for time series prediction. In this paper, our aim is to reduce the computational overhead in training these networks by applying parallel computing techniques using the mpi4py library. By distributing the training process across multiple processors, our goal is to speed up model training while maintaining accuracy. We will evaluate our approach on wind farm data obtained from seven wind farms in Europe and stock prices from the S\&P 500 index using Mean Square Error (MSE) and Training Time metrics to observe how the parallel computing approach can reduce training time without compromising performance.

Runnign the Code:
The Latest updated codes are in "Updated Codes" Folder. There are two subfolders for 10 years and 6 years data. Although Codes and approaches are the same, we have different folders just for ease of running.
To run the code in any of the subfolders, bash script files are available namely run.sh.

run.sh will run the entire set of models- GRU, LSM, BiLSTM and Hybrid in Series, Data Parallel and Model parallel approaches and plot results are saved in plots folder. Output files are saved in outputs folders.

Individual setting can also be run in similar way as done in run.sh file.
For example, to run GRU model in Dataparallel mode with 4 cores and save the output in MPI_4_GRU.out file, we can run the following command in terminal:
mpirun -np 4 python3 TS_Prediction_DataParallel.py -m GRU > outputs/MPI_4_GRU.out
