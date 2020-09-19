from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from numpy import concatenate
from math import sqrt
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
import time

'''This script is used to train RNN based time series forecasting model using Selego (ot other variate selection technique) selected variates.
Go to __main__ module and define the parameters and necessary inputs to training and test the model forecasting. '''

class TimeHistory(tf.keras.callbacks.Callback):
    times = []
    epoch_time_start = 0

    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def series_to_supervised(data, n_in, n_out, count, dropnan=True,):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    drop_column = []
    j = count + 1
    k = j + (n_in - 1) * j + count
    for m in range(j, k):
        drop_column.append(m)

    # drop columns we don't want to predict
    agg.drop(agg.columns[drop_column], axis=1, inplace=True)

    return agg


def dataSplit(normalized_data, timestamps, batch_train, batch_valid, batch_test):

    train_end = timestamps * batch_train
    valid_end = train_end + timestamps * batch_valid
    test_end = valid_end + timestamps * batch_test

    train = normalized_data[:train_end, :]

    valid = normalized_data[train_end:valid_end, :]

    test = normalized_data[valid_end:test_end]

    train_X, train_y = train[:, :-1], train[:, -1]
    valid_X, valid_y = valid[:, :-1], valid[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((batch_train, timestamps, train_X.shape[1]))
    valid_X = valid_X.reshape((batch_valid, timestamps, valid_X.shape[1]))
    test_X = test_X.reshape((batch_test, timestamps, test_X.shape[1]))
    train_y = train_y.reshape((batch_train, timestamps, 1))
    valid_y = valid_y.reshape((batch_valid, timestamps, 1))
    test_y = test_y.reshape((batch_test, timestamps, 1))

    return train_X, train_y, valid_X, valid_y, test_X, test_y

def getTopXPercentVariatesNames(VariateRankingFilePath, TopPercent_variates):

    # read the variate ranking file into DataFrame object and extracting variates name into list
    variateRanking_df = pd.read_csv(VariateRankingFilePath, header=None)
    Ranked_variates_name = variateRanking_df.iloc[:, 0].tolist()
    total_variates = len(Ranked_variates_name)

    # calculating the # of variates for given top% from total # of variates
    # and then appending their names in the columns list by extracting those from Ranked_variates_name
    VariateCount = (total_variates * TopPercent_variates) // 100
    columns = []
    columns.extend(Ranked_variates_name[:VariateCount])
    columns.append(TargetVariate)

    return columns, VariateCount


if __name__ == "__main__":

    # specify the dataset file path
    dataset_path = "../../Datasets/Nasdaq/NasdaqDataset.csv"

    # specify the variate rankings file path of the given lead, here we consider to load csv file
    VariateRankingFilePath = "../../Nasdaq_lead5_variate_names.csv"

    # specify the directory path to store model forecasting Errors
    predictionErrorFilePath = "../../prediction/error/directory/path/"

    # specify directory path to save the model forecasting on test data
    prediction_path = "../../directory/path/to/save/forecasting/results"

    # provide a directory path to save the best model state checkpoint and model training timings
    dest_path = "../../directory/path/to/save/modelCheckpoint/"

    # specify the exact name target variate given in the dataset Example: TargetVariate = "NDX"
    TargetVariate = "NDX"

    # specify the Data Normalization technique. for example Norm = "NoNorm" or "Z-Norm" or "MinMax-Norm"
    Norm = "Z-Norm"

    # specify top X% variates you want to consider. For example: TopPercent_variates = 2 --> 2% input variates
    TopPercent_variates = 10

    # specify of lead you want to predict Example: Lead = 1 or Lead = 5
    # here in example Lead = 1 --> 1-timestamp ahead forecasting, Lead = 5 --> 5-timesteps ahead forecasting
    Lead = 5

    # Number of timestamps per batch size. Refer Table 1 in the paper to get the Number_of_timestamps.
    Number_of_timestamps = 210

    # input data split (into batches) for training, validation and testing.
    batch_train = 70
    batch_valid = 10
    batch_test = 25

    # creating a pandas dataframe object to store the model errors results for different top% variates and given lead
    dfObj = pd.DataFrame(columns=['Top%', 'Lead', 'Run', 'MAE'])

    # read the input dataset into pandas DataFrame object
    dataset_df = pd.read_csv(dataset_path)

    columnsNames, VariateCount = getTopXPercentVariatesNames(VariateRankingFilePath, TopPercent_variates)

    # select the top X% variates data for the experiment
    dataset = dataset_df[columnsNames]
    values = dataset.values

    # ensure all data is float
    values = values.astype('float32')

    # frame as supervised learning
    reframed = series_to_supervised(values,Lead, 1, VariateCount, True)
    supervised_values = reframed.values

    # normalize the input data before training the model
    if Norm == "Z-Norm":
        scaler = StandardScaler().fit(supervised_values)
        normalized_data = scaler.transform(supervised_values)
    elif Norm == "MinMax-Norm":
        scaler = MinMaxScaler(feature_range=(0, 1)).fit(supervised_values)
        normalized_data = scaler.transform(supervised_values)
    else:
        normalized_data = supervised_values


    # split the normalized data into train, valid and test
    train_X, train_y, valid_X, valid_y, test_X, test_y = dataSplit(normalized_data, Number_of_timestamps, batch_train,
                                                                   batch_valid, batch_test)

    # following loop consider multiple runs of model training for the
    # given top x% variates and Lead. In the experiments, we considered 5 runs per configuration
    for run in range(1, 6):

        # we considered, batch_size = 1 as a default for all the experiments
        batch_size = 1

        # model configuration
        model = Sequential()
        model.add(SimpleRNN(200, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='RMSprop', metrics=['accuracy'])

        # Saving best model given the epcoh performance
        bestpath = dest_path + "/Models.best_"+str(run)+".h5"
        model_weight_best = ModelCheckpoint(bestpath, save_best_only=True)

        # Creating Model Timinig callback
        model_time_callback = TimeHistory()

        callbacks_list = [model_weight_best, model_time_callback]

        # fit network
        history = model.fit(train_X, train_y, epochs=200, batch_size=batch_size,
                            validation_data=(valid_X, valid_y), callbacks=callbacks_list, verbose=2,
                            shuffle=False)

        # Saving model tmining callback output
        timepath = dest_path + "/times"+str(run)+".csv"
        np.savetxt(timepath, model_time_callback.times, delimiter=",", fmt="%1.05f")

        # loading the model state
        model1 = load_model(bestpath)

        # make a prediction
        copy_test_X = test_X
        copy_test_Y = test_y

        yhat = model1.predict(copy_test_X)

        # reshaping forecasted values and the test data from 3d to 2d
        yhat = yhat.reshape(yhat.shape[0] * yhat.shape[1], yhat.shape[2])
        copy_test_X = copy_test_X.reshape((copy_test_X.shape[0] * copy_test_X.shape[1], copy_test_X.shape[2]))

        # invert scaling for forecast
        inv_yhat = concatenate((copy_test_X[:, :], yhat), axis=1)
        if Norm != "NoNorm":
            inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:, -1]

        # invert scaling for actual
        copy_test_Y = copy_test_Y.reshape((copy_test_Y.shape[0] * copy_test_Y.shape[1], 1))
        inv_y = concatenate((copy_test_X, copy_test_Y), axis=1)
        if Norm != "NoNorm":
            inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:, -1]

        output = prediction_path + "/Model_prediction"+str(run)+".xlsx"
        predicted_df = pd.DataFrame({"Actual": inv_y, "Predicted": inv_yhat})
        predicted_df.to_excel(output)

        # calculate MAE
        mae = mean_absolute_error(inv_y, inv_yhat)

        dfObj = dfObj.append({'Top%': TopPercent_variates, 'Lead': Lead, 'Run': run, 'MAE': mae},ignore_index=True)


    ErrorOutput = predictionErrorFilePath + '/ModelForecastingErrors.csv'
    dfObj.to_csv(ErrorOutput, index=None)

