# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# from jsonschema.compat import urlopen
import math
import string
import os
import matplotlib.pyplot as plt
import numpy
import tensorflow as tf

# from jsonschema.compat import urlopen
from keras import Model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.models import model_from_json
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Input
from tensorflow.python.layers.core import dense
from user_agent import generate_user_agent
import quandl
import unicodedata as U1
from finviz.screener import Screener
from finviz.portfolio import Portfolio
import finviz

#########################
#   Examples Section    #
#########################

#########################
#   Project Section     #
#########################
"""
Creates the Model
numEpochs: number of iterations of the neural net 
stockName: Name of equity
look_back: Domain Days of function. The set of days per training set
look_forward: Range Day of function. The n-th day forward from the end of the training set you're predicting
FunctionalAPI: 2nd choice of prediction adds volume
"""
def LSTM_model_10A(numEpochs, stockName, look_back, look_forward, FunctionalAPI):
    """
    Creates multivalue time series with price and volume
    Parameters:
        numEpochs;integer
        stockName:string
        look_back:integer
        FuncitonalAPI:boolean
    """
    # 12/8/19 Use this instead for time series: BRZY84UZ9H66SK72
    # https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol=TCNB&apikey=BRZY84UZ9H66SK72&datatype=csv
    # https://www.alphavantage.co/documentation/#time-series-data
    # NEW alphaVantage Python API : https://alpha-vantage.readthedocs.io/en/latest/
    # https://rapidapi.com/patrick.collins/api/alpha-vantage

    numpy.random.seed(7)
    import quandl
    quandl.ApiConfig.api_key = "6X-hxfxiPcSb3uTS2UyB"
    # df = quandl.get("WIKI/KR")
    # df = quandl.get("EOD/MSFT")
    # df = quandl.get("SHARADAR/SEP",ticker="DDS")
    df = quandl.get(stockName)
    # START HERE and include Volume! 8/24/19
    # ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividend', 'Split', 'Adj_Open', 'Adj_High', 'Adj_Low', 'Adj_Close', 'Adj_Volume']
    print(df.tail(4))

    for index, row in df.head().iterrows():
        print(index, row['Open'], row['High'], row['Adj_Close'])  # HERE
    import pandas as pd
    # New code 9/21/2019
    dataframeAdj_Close = df[['Adj_Close', 'Volume']]
    dataframePriceOnly = df[['Adj_Close']]
    Debug = False

    # This actually splits upa nd creates your
    # Independent and Dependent Variables
    # But it's complicated and you need to get back to it.
    # look_back = 3
    def makeTimeSeries(dataset, look_back, look_forward, factor1, factor2):
        print("begin makeTimeSeries")
        # IN: dataset=  [[Price1,Volume1]..[PriceN,VolumeN]]
        # Checked comes in price and volume viz:
        # [[3.1752134e-05 6.0479403e-03]
        idx = 0
        l = 0
        print("***********")
        print("* Price,Volume *")
        print("***********")
        # dataX, dataY = [[],[]], [[],[]]
        dataX, dataXInter, dataY = [], [], []
        aSuperList = []
        aIntermediateList = []
        for i in range(len(dataset) - (look_back + look_forward)):  # -1
            a1 = []
            a2 = []
            for timeSteps in range(look_back):
                a1 = dataset[
                    i + timeSteps, 0]  # 5 or 6? element slice... Stops at (len-1)th element e.g. if i=0 and look_back=6 so 5th element    goes from (len-k-1 to len -1)_ and e.g. a[1:2],a[2:3],a[3:4]  goes from
                a2 = dataset[
                    i + timeSteps, 1]  # 5 or 6? element slice... Stops at (len-1) e.g. if i=0 and look_back=6 so 5th element   goes from (len-k-1 to len -1)_ and e.g. a[1:2],a[2:3],a[3:4]  goes from
                #            a1 = dataset[i:(i + look_back),0]  # 5 or 6? element slice... Stops at (len-1)th element e.g. if i=0 and look_back=6 so 5th element    goes from (len-k-1 to len -1)_ and e.g. a[1:2],a[2:3],a[3:4]  goes from
                #            a2 = dataset[i:(i + look_back),1] # 5 or 6? element slice... Stops at (len-1) e.g. if i=0 and look_back=6 so 5th element   goes from (len-k-1 to len -1)_ and e.g. a[1:2],a[2:3],a[3:4]  goes from
                dataX.clear()
                dataX.append(a1 * factor1)  # Price 1 elements
                dataX.append(a2 * factor2)  # Volume 1 elements
                dataXInter.append(list(dataX))
            b1 = []
            # not used b2 = []
            # To get a forward looking y do a for x = 1 to lookforward(append to b1)
            if i + look_back + look_forward <= len(dataset):
                b1 = numpy.float32(
                    dataset[
                        i + look_back + look_forward, 0]) * factor1  # Y=(i+look_back)th element e.g. i=0, look_back=6 so 6th element
                b2 = numpy.float32(
                    dataset[i + look_back + look_forward, 1]) * factor2
            dataY.append(b1)  # Y=Price
            aIntermediateList.append(dataXInter.copy())  # 6 X's
            # print("DATAXINTER=")
            # print(dataXInter)
            dataXInter.clear()
        # Diagnostics
        # print("Superlist={}".format(numpy.array(aIntermediateList).shape))
        # print("Datay={}".format(numpy.array(dataY).shape))
        # print(aIntermediateList[0:6])
        # OUT:  dataX: [[[Price1,Volume1]...[Price6],[Volume6]]...[[Price,Volume1]...[Price6,Volume6]]]
        # dataX Shape:(5,000,6,2)
        # in Fact if we follow Jason Brownlee at this url:
        # https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
        # print("IntermediateList")
        # print(aIntermediateList)
        # print("dataY")
        # print(dataY)
        # exit(0)
        return numpy.array(aIntermediateList), numpy.array(dataY)

    def makeTimeSeries2(dataset, look_back, factor1, factor2):
        print("begin makeTimeSeries")
        # IN: dataset=  [[Price1,Volume1]..[PriceN,VolumeN]]
        # Checked comes in price and volume viz:
        # [[3.1752134e-05 6.0479403e-03]
        idx = 0
        l = 0
        print("***********")
        print("* Price,Volume *")
        print("***********")
        # dataX, dataY = [[],[]], [[],[]]
        dataX, dataXInter, dataY = [], [], []
        aSuperList = []
        aIntermediateList = []
        for i in range(len(dataset) - look_back):  # -1
            a1 = []
            a2 = []
            for timeSteps in range(look_back):
                a1 = dataset[
                    i + timeSteps, 0]  # 5 or 6? element slice... Stops at (len-1)th element e.g. if i=0 and look_back=6 so 5th element    goes from (len-k-1 to len -1)_ and e.g. a[1:2],a[2:3],a[3:4]  goes from
                a2 = dataset[
                    i + timeSteps, 1]  # 5 or 6? element slice... Stops at (len-1) e.g. if i=0 and look_back=6 so 5th element   goes from (len-k-1 to len -1)_ and e.g. a[1:2],a[2:3],a[3:4]  goes from
                #            a1 = dataset[i:(i + look_back),0]  # 5 or 6? element slice... Stops at (len-1)th element e.g. if i=0 and look_back=6 so 5th element    goes from (len-k-1 to len -1)_ and e.g. a[1:2],a[2:3],a[3:4]  goes from
                #            a2 = dataset[i:(i + look_back),1] # 5 or 6? element slice... Stops at (len-1) e.g. if i=0 and look_back=6 so 5th element   goes from (len-k-1 to len -1)_ and e.g. a[1:2],a[2:3],a[3:4]  goes from
                dataX.clear()
                dataX.append(a1 * factor1)  # Price 1 elements
                dataX.append(a2 * factor2)  # Volume 1 elements
                dataXInter.append(list(dataX))
            b1 = []
            # start with two elements and scale up the Y
            b1[0] = numpy.float32(
                dataset[
                    i + look_back - 1, 0]) * factor1  # Y=(i+look_back)th element e.g. i=0, look_back=6 so 6th element
            b1[1] = numpy.float32(
                dataset[i + look_back, 0]) * factor1  # Y=(i+look_back)th element e.g. i=0, look_back=6 so 6th element
            dataY.append(b1[0])  # Y=Price
            dataY.append(b1[1])
            aIntermediateList.append(dataXInter.copy())
            dataXInter.clear()
        print("Superlist={}".format(numpy.array(aIntermediateList).shape))
        print("Datay={}".format(numpy.array(dataY).shape))
        print(aIntermediateList[0:6])
        # OUT:  dataX: [[[Price1,Volume1]...[Price6],[Volume6]]...[[Price,Volume1]...[Price6,Volume6]]]
        # dataX Shape:(5,000,6,2)3
        # in Fact if we follow Jason Brownlee at this url:
        # https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
        return numpy.array(aIntermediateList), numpy.array(dataY)

    numpy.random.seed(7)
    # dataframeAdj_Close = dffinalAdj_Close
    # dataframeVolume=dffinalVolume
    print("*********************************\n")
    print("Tail Values: \n", dataframeAdj_Close.tail(4))
    print("*********************************\n")
    datasetPrice = dataframeAdj_Close.values
    datasetPrice = datasetPrice.astype('float32')
    datasetPriceOnly = dataframePriceOnly.values
    datasetPriceOnly = dataframePriceOnly.astype('float32')
    if FunctionalAPI == True:
        datasetVolume = datasetPrice.values
        datasetVolume = datasetVolume.astype('float32')
    # normalize the datasetPrice
    scaler = MinMaxScaler(feature_range=(0, 1))
    # FIT_TRANSFORM CLEANS UP THE NULLS!!!
    # AND NORMALIZES!!!!
    # Error expected 2D error got 1D array
    Debug = True
    if Debug == True:
        print("**************************" + "\n")
        print("* Print the datasetPrice      *" + "\n")
        print("**************************" + "\n")
        print(datasetPrice)
        print(numpy.expand_dims(datasetPrice, axis=2).shape)
        print(numpy.array(datasetPrice).shape)
    datasetPrice = scaler.fit_transform(datasetPrice)

    datasetPriceOnly = scaler.fit_transform(datasetPriceOnly)
    if FunctionalAPI == True:
        datasetVolume = scaler.fit_transform(datasetVolume)
    # split into train and test sets
    train_size = int(len(datasetPrice) * 0.67)
    test_size = len(datasetPrice) - train_size
    if FunctionalAPI == True:
        train_sizeVolume = int(len(datasetVolume) * 0.67)
        test_sizeVolume = len(datasetVolume) - train_sizeVolume  # train and test are both 2 Dimensional Arrays
    # Break 67% into train and the rest of it into test
    # train gets both price and volume here!!
    train, test = datasetPrice[0:train_size, :], datasetPrice[train_size:len(datasetPrice),
                                                 :]  # 2 - 2 Dimensional Arrays
    if FunctionalAPI == True:
        trainVolume, testVolume = datasetVolume[0:train_sizeVolume, :], datasetVolume[
                                                                        train_sizeVolume:len(datasetVolume),
                                                                        :]  # 2 - 2 Dimensional Arrays
    if Debug == True:
        print("**************************")
        print("* Print out train & test *")
        print("**************************")
        for i in range(0, train_size):
            for j in range(0, 1):
                print(train[i, j])
        print("\r\n")
        print("**************************")
        print("Train={}".format(train))
        print("Train shape={}".format(train.shape))
    factor1 = 1;
    factor2 = .34
    trainX, trainY = makeTimeSeries(train, look_back, look_forward, factor1, factor2)
    print(scaler.inverse_transform(numpy.array(trainY).reshape(-1, 1)[-10:]))
    if FunctionalAPI == True:
        trainXVolume, trainYVolume = makeTimeSeries(trainVolume, look_back, look_forward, factor1, factor2)
    trainX, trainY = makeTimeSeries(train, look_back, look_forward, factor1, factor2)
    testX, testY = makeTimeSeries(test, look_back, look_forward, factor1, factor2)
    if FunctionalAPI == True:
        testXVolume, testYVolume = makeTimeSeries(testVolume, look_back, look_forward, factor1, factor2)
    if Debug == True:
        print("trainX", trainX)
        print("trainY", trainY)
        print("Shape", trainX.shape)
        print("Shape", testX.shape)
    oldtestX = testX
    trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], trainX.shape[2]))
    testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], testX.shape[2]))

    if FunctionalAPI == True:
        trainXVolume = numpy.reshape(trainXVolume,
                                     (trainXVolume.shape[0], trainXVolume.shape[1], trainXVolume.shape[2]))
        testXVolume = numpy.reshape(testXVolume, (testXVolume.shape[0], testXVolume.shape[1], testXVolume.shape[2]))
    if Debug == True:
        print("**********CHECKING LAST 2 VALUES IN DATASET****************")
        # YES WE DO GET TO THE LAST 2 VALUES IN THE DATASET!!!!!!!!!!!
        print("New TrainX=", scaler.inverse_transform(numpy.reshape(trainX[-1:], (1, -1))))
        print("New TrainX Shape=", trainX.shape)
        print("New TestX=", scaler.inverse_transform(numpy.reshape(testX[-1:], (1, -1))))
        print("New TestX Shape=", testX.shape)
        print("New TestY=", scaler.inverse_transform(numpy.reshape(testY[-1:], (1, -1))))
        print("New TestY Shape=", testY.shape)
        print("***********************************************************")
        print("New Test last value=", scaler.inverse_transform(numpy.reshape(test[-1:], (1, -1))))
        print("dataset last value={}".format(
            scaler.inverse_transform(numpy.reshape(datasetPrice[len(datasetPrice) - 1], (1, -1)))))
        print("You is here!")

    score = 0
    # http://www.itdaan.com/blog/2017/11/09/c2525310531416583200a3e14fbb8965.html
    if FunctionalAPI == True:
        # ***************************************************************
        # Beginning of Will Teslers Code

        import keras as ks
        # No activation functions for Input Layer
        price = Input(shape=(look_back, 1), name='price')
        volume = Input(shape=(look_back, 1), name='volume')
        HiddenSize = (look_back + 1) * 2
        priceLayers = [[0]]
        volumeLayers = [[0]]
        # priceLayers = LSTM(64, return_sequences=False)(price)
        # volumeLayers = LSTM(64, return_sequences=False)(volume)
        # tanh for hidden layer
        priceLayers = LSTM(HiddenSize, return_sequences=False, activation='tanh')(price)
        volumeLayers = LSTM(HiddenSize, return_sequences=False, activation='tanh')(volume)
        # ks.layers.LeakyReLU(alpha=0.3)
        output = ks.layers.Concatenate(axis=-1)([priceLayers, volumeLayers, ])
        # output = ks.layers.Concatenate([priceLayers, volumeLayers, ])
        # Dense is just a nn layer (usually and here for output)
        # Runs but may need Dense(1 to be Dense(look_back
        # Start here 8/11/2019
        output = Dense(1, name='weightedAverage_output', activation='relu')(output)
        model1 = Model(inputs=[price, volume], outputs=[output])
        # model1.compile(optimizer='rmsprop', loss ='mse')
        model1.compile(loss='mean_squared_error', optimizer='adam')
        # Beginning of my mod
        # Load Model with Data
        # https://machinelearningmastery.com/keras-functional-api-deep-learning/
        print("trainx=" + str(trainX.shape))
        print("trainy=" + str(trainY.shape))
        # Start Here change [trainX,trainX] to [trainX,trainXVolume] 8/26/2019

        model1.fit([trainX, trainXVolume], trainY, epochs=numEpochs, batch_size=1, verbose=2)
        score = model1.evaluate([trainX, trainXVolume], trainY, verbose=0)
        # exit(0)
        # End of Will Teslers Code
    else:
        # **************************************************************
        # Old Sequential model Begin
        # 1.) Create Model
        # https://machinelearningmastery.com/stacked-long-short-term-memory-networks/
        # https: // www.researchgate.net / publication / 236164660_Optimal_Neural_Network_Architecture_for_Stock_Market_Forecasting
        model = tf.keras.Sequential()
        HiddenSize = (look_back + 1) * 2
        # Original model.add(LSTM(HiddenSize, activation='tanh', input_shape=(look_back, 1)))
        # 11/3/2019 This in put shape should be (look_back,2)
        model.add(
            tf.keras.layers.LSTM(HiddenSize, activation='tanh', input_shape=(look_back, 2), return_sequences=True))
        model.add(tf.keras.layers.LSTM(math.ceil(HiddenSize * .5), activation='tanh',
                                       return_sequences=True))  # Last layer does not need return_sequences
        model.add(tf.keras.layers.LSTM(math.ceil(HiddenSize * .1), activation='tanh', return_sequences=False))
        model.add(tf.keras.layers.Dense(1))
        model.add(tf.keras.layers.LeakyReLU(alpha=.001))
        # model.add(Dense(1, input_dim=1, activation='relu'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        # https://stackoverflow.com/questions/22981845/3-dimensional-array-in-numpy
        print(trainX.shape)
        print("***************->trainX=", trainX)
        # Pretty sure I set this up as Trainx as t and TrainY as (t+1) all the way across the datasetPrice
        # Load Model with Data
        print(trainX.shape)
        print(trainY.shape)
        # "tail"
        # Find out what's in this trainY whether or not it's price, volume or both here!!
        print("tail")
        print(scaler.inverse_transform(numpy.reshape(trainY, (1, -1))))
        # Yes these are volumes
        # exit(0)!
        print(numpy.array(trainY)[:10])
        # Fit on trainX, trainY
        model.fit(testX, testY, epochs=numEpochs, batch_size=1, verbose=2)
        score = model.evaluate(trainX, trainY, verbose=0)
        # Evaluate Description: https://stackoverflow.com/questions/51299836/what-values-are-returned-from-model-evaluate-in-keras
        print("2.) Old Sequential model end")
        print("# **********************************************************************")

    print("Evaluation score=?")
    print(model.metrics_names)
    print(score)
    if FunctionalAPI == True:
        trainPredict = model1.predict([trainX, trainXVolume])
    else:
        trainPredict = model.predict(trainX)
    # TrainPredict(although just price) is created from the model in both price and volume (from trainX) above

    # Scale this!!!!!!!!!!!!!!! it isn't scaled!!!!!!!!
    # exit(0)

    print("trainPredict prior=", trainPredict)
    if FunctionalAPI == True:
        testPredict = model1.predict([testX, testXVolume])
    else:
        testPredict = model.predict(testX)  # testX #n,6,2 testPredict n,1
    # Transform back to ORIGINAL UNSCALED data
    # You Are here 8/26/2019
    Debug = True
    if Debug == True:
        print("----testX------151--")
        # see if this shape is n,6,2? for testX
        print("testX Shape={}".format(testX.shape))
        print("Shows final 6,2 vector of testX and reshapes")
        print(scaler.inverse_transform(testX[-1:].reshape(-1, 1)))
        print("-----testPredict--129-----")
        print("Shows final price of testPredict")
        print(scaler.inverse_transform(testPredict[-1:].reshape(-1, 1)))
        print(numpy.array(testPredict).shape)
        print("Shows all of testPredict")
        print(scaler.inverse_transform(testPredict[:].reshape(-1, 1)))
        print("IN conclusion price and volume went in and Price only came out")
        # 12/16/2019
        # exit(0)
        print("----trainX--------")
        print(scaler.inverse_transform(trainX[:-1].reshape(-1, 1)))
        print("----trainPredict--------")
        print(scaler.inverse_transform(trainPredict[:-1].reshape(-1, 1)))
        print(numpy.array(trainPredict).shape)
        # exit(0)
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY1 = scaler.inverse_transform([trainY])
    testY1 = scaler.inverse_transform([testY])
    if Debug == True:
        print("trainPredict={}".format(numpy.array(trainPredict.shape)))
        print("trainPredict after=", trainPredict)
        print(numpy.array(testY1).shape)
        print(testY1)
        print("----------")
        print("testX")
        print(oldtestX[:-2])
        print("-----------")
        print(oldtestX[:2])
    testPredict = scaler.inverse_transform(testPredict)
    print("testPredict after=", testPredict)
    print("shape testPredict", testPredict.shape)
    # 11/23/2019 This is fine it outputs a nX1 array unlike what's in load_5A
    # 11/17/2019 You are here testY1 is good makeTimeSeries is good
    # but testPredict is bad and so is your plotting so fix them now
    # exit(0)
    # calculate root mean squared error of trainPredict and testPredict Models
    # Shape of trainPredict [2355,1] and testPredict [1159,1]  2D Arrays at this point
    Debug = True
    if Debug == True:
        print("data train Predict", trainPredict)
        print("shape trainPredict", trainPredict.shape)
        print("data test Predict", testPredict)
        print("shape testPredict", testPredict.shape)
        # print("shape trainY1", trainY1.shape)
        # print("data trainY1", trainY1)
    trainScore = math.sqrt(mean_squared_error(trainY1[0], trainPredict[:, 0]))
    print("Point #1")
    print("trainY1[0]={}".format(trainY1[0]))
    print("trainPredict={}".format(trainPredict[:, 0]))
    # exit(0)
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY1[0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))
    # from sklearn.metrics import accuracy_scoreaccuracy_score(df.actual_label.values, df.predicted_RF.values)
    # https://towardsdatascience.com/understanding-data-science-classification-metrics-in-scikit-learn-in-python-3bc336865019
    # shift train predictions for plotting
    # Returns a same Shape array as given array
    ##You are here 8/26/2019-----------------------------------------------
    trainPredictPlot = numpy.empty_like(datasetPrice)
    trainPredictPlot[:, :] = numpy.nan
    testPredictPlot = numpy.empty_like(datasetPrice)
    testPredictPlot[:, :] = numpy.nan
    #  model 2D = 2D arrays
    print("trainPredict shape=", numpy.array(trainPredict).shape)
    print("datasetPrice shape=", numpy.array(datasetPrice).shape)
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
    print("Completed trainPredictPlot")
    # testPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
    print("look_back:len(trainPredict) + look_back= {}:{},: ".format(look_back, len(trainPredict) + look_back))
    print("len(trainPredict) + (look_back * 2) + 1 ={} : len(datasetPrice) - 1={},:".format(
        len(trainPredict) + (look_back * 2) + 1, len(datasetPrice) - 1))
    print(numpy.array(trainPredict).shape)
    print(numpy.array(testPredict).shape)
    print(testPredictPlot.shape)
    # The rows are not big enough on the trainPredictPlot to accept the testPredict rows. The slice on trainPredict is too small-simply..
    print(len(trainPredict) + (look_back * 2) + 1, len(datasetPrice) - 1)
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(datasetPrice) - 1, :] = testPredict[
                                                                                        0:len(trainPredict) + (
                                                                                                look_back * 2) + 1:len(
                                                                                            datasetPrice) - 1]  # testPredict[0:2798]
    print("Completed testPredictPlot")
    testPredictPlot = numpy.empty_like(datasetPrice)
    testPredictPlot[:, :] = numpy.nan
    #  model 2D = 2D arrays
    print("testPredict shape=", numpy.array(testPredict).shape)
    print("datasetPrice shape=", numpy.array(datasetPrice).shape)
    print(len(trainPredict) + (look_back * 2) + 1, len(datasetPrice) - 1)
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(datasetPrice) - 1, :] = testPredict[
                                                                                        0:len(trainPredict) + (
                                                                                                look_back * 2) + 1:len(
                                                                                            datasetPrice) - 1]  # testPredict[0:2798]
    print("Completed testPredictPlot")
    # plot baseline and predictions
    # Add Plot for the datasetPrice
    plt.xlabel('Days from beginning')
    plt.ylabel('Price')
    plt.title('Stock Prediction for ' + stockName)
    plt.grid(True)
    plt.plot(scaler.inverse_transform(datasetPrice), color='b', lw=0.5, label='datasetPrice')
    # Add Plot for the training model
    plt.plot(trainPredictPlot, color='r', lw=0.5, label='Training')
    # Add Plot for the test model
    plt.plot(testPredictPlot, color='g', lw=0.5, label='Test')
    # Do the Plot onscreen
    plt.legend(loc=1, fontsize="x-large")
    leg = plt.legend()
    # get the lines and texts inside legend box
    leg_lines = leg.get_lines()
    leg_texts = leg.get_texts()
    # bulk-set the properties of all lines and texts
    plt.setp(leg_lines, linewidth=4)
    plt.setp(leg_texts, fontsize='x-large')
    # plt.show()
    # https://stackoverflow.com/questions/741877/how-do-i-tell-matplotlib-that-i-am-done-with-a-plot
    plt.clf()
    plt.cla()
    plt.close()
    # https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
    # *********************************
    # Part 2 Write the model to disk *
    # serialize model to JSON        *
    # *********************************

    # Changed 9/3/2019
    # if FunctionalAPI == True:
    #    model_json = model1.to_json()
    # else:
    #    testPredict = model.predict(testX)

    # PROBLEM HERE 9/3/2019!!!!!!!!!!!!!!
    if FunctionalAPI == True:
        model_json = model1.to_json()
    else:
        model_json = model.to_json()

    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    if FunctionalAPI == True:
        model1.save_weights("model.h5")
    else:
        model.save_weights("model.h5")
    print("Saved model to disk")
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    # evaluate loaded model on test data
    # loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    loaded_model.compile(loss='mean_squared_error', optimizer='adam')
    #    score = loaded_model.predict(trainX, trainY, verbose=0)
    # print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
    return (0)


"""DEPRECATED"""
def LSTM_model_11A(numEpochs, stockName, look_back, FunctionalAPI):
    """
    Creates multivalue time series with price and volume
    Parameters:
        numEpochs;integer
        stockName:string
        look_back:integer
        FuncitonalAPI:boolean
    """
    # 12/8/19 Use this instead for time series: BRZY84UZ9H66SK72
    # https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol=TCNB&apikey=BRZY84UZ9H66SK72&datatype=csv
    # https://www.alphavantage.co/documentation/#time-series-data
    # NEW alphaVantage Python API : https://alpha-vantage.readthedocs.io/en/latest/
    # https://rapidapi.com/patrick.collins/api/alpha-vantage

    numpy.random.seed(7)
    import quandl
    quandl.ApiConfig.api_key = "6X-hxfxiPcSb3uTS2UyB"
    # df = quandl.get("WIKI/KR")
    # df = quandl.get("EOD/MSFT")
    # df = quandl.get("SHARADAR/SEP",ticker="DDS")
    df = quandl.get(stockName)
    # START HERE and include Volume! 8/24/19
    # ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividend', 'Split', 'Adj_Open', 'Adj_High', 'Adj_Low', 'Adj_Close', 'Adj_Volume']
    print(df.tail(4))

    for index, row in df.head().iterrows():
        print(index, row['Open'], row['High'], row['Adj_Close'])  # HERE
    import pandas as pd
    # New code 9/21/2019
    dataframeAdj_Close = df[['Adj_Close', 'Volume']]
    dataframePriceOnly = df[['Adj_Close']]
    Debug = False

    # This actually splits upa nd creates your
    # Independent and Dependent Variables
    # But it's complicated and you need to get back to it.
    # look_back = 3
    def makeTimeSeries(dataset, look_back, factor1, factor2):
        print("begin makeTimeSeries")
        # IN: dataset=  [[Price1,Volume1]..[PriceN,VolumeN]]
        # Checked comes in price and volume viz:
        # [[3.1752134e-05 6.0479403e-03]
        idx = 0
        l = 0
        print("***********")
        print("* Price,Volume *")
        print("***********")
        # dataX, dataY = [[],[]], [[],[]]
        dataX, dataXInter, dataY = [], [], []
        aSuperList = []
        aIntermediateList = []
        for i in range(len(dataset) - look_back):  # -1
            a1 = []
            a2 = []
            for timeSteps in range(look_back):
                a1 = dataset[
                    i + timeSteps, 0]  # 5 or 6? element slice... Stops at (len-1)th element e.g. if i=0 and look_back=6 so 5th element    goes from (len-k-1 to len -1)_ and e.g. a[1:2],a[2:3],a[3:4]  goes from
                a2 = dataset[
                    i + timeSteps, 1]  # 5 or 6? element slice... Stops at (len-1) e.g. if i=0 and look_back=6 so 5th element   goes from (len-k-1 to len -1)_ and e.g. a[1:2],a[2:3],a[3:4]  goes from
                #            a1 = dataset[i:(i + look_back),0]  # 5 or 6? element slice... Stops at (len-1)th element e.g. if i=0 and look_back=6 so 5th element    goes from (len-k-1 to len -1)_ and e.g. a[1:2],a[2:3],a[3:4]  goes from
                #            a2 = dataset[i:(i + look_back),1] # 5 or 6? element slice... Stops at (len-1) e.g. if i=0 and look_back=6 so 5th element   goes from (len-k-1 to len -1)_ and e.g. a[1:2],a[2:3],a[3:4]  goes from
                dataX.clear()
                dataX.append(a1 * factor1)  # Price 1 elements
                dataX.append(a2 * factor2)  # Volume 1 elements
                dataXInter.append(list(dataX))
            b1 = []
            b2 = []
            b1 = numpy.float32(
                dataset[i + look_back, 0]) * factor1  # Y=(i+look_back)th element e.g. i=0, look_back=6 so 6th element
            b2 = numpy.float32(
                dataset[i + look_back, 1]) * factor2
            dataY.append(b1)  # Y=Price
            aIntermediateList.append(dataXInter.copy())
            dataXInter.clear()
        print("Superlist={}".format(numpy.array(aIntermediateList).shape))
        print("Datay={}".format(numpy.array(dataY).shape))
        print(aIntermediateList[0:6])
        # OUT:  dataX: [[[Price1,Volume1]...[Price6],[Volume6]]...[[Price,Volume1]...[Price6,Volume6]]]
        # dataX Shape:(5,000,6,2)
        # in Fact if we follow Jason Brownlee at this url:
        # https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
        return numpy.array(aIntermediateList), numpy.array(dataY)

    def makeTimeSeries2(dataset, look_back, factor1, factor2):
        print("begin makeTimeSeries")
        # IN: dataset=  [[Price1,Volume1]..[PriceN,VolumeN]]
        # Checked comes in price and volume viz:
        # [[3.1752134e-05 6.0479403e-03]
        idx = 0
        l = 0
        print(numpy.array(dataset).shape)

        print("***********")
        print("* Price,Volume *")
        print("***********")
        # dataX, dataY = [[],[]], [[],[]]
        dataX, dataXInter, dataY, dataYInter = [], [], [], []
        aSuperList = []
        aIntermediateList = []
        dataY.clear()
        counter1 = 0
        for i in range(len(dataset) - look_back):  # -1
            # 4081 itrerations of i
            counter1 = counter1 + 1
            a1 = []
            a2 = []
            iter = 0
            for timeSteps in range(look_back):
                iter = iter + 1
                a1 = dataset[
                    i + timeSteps, 0]
                a2 = dataset[
                    i + timeSteps, 1]
                #            a1 = dataset[i:(i + look_back),0]  # 5 or 6? element slice... Stops at (len-1)th element e.g. if i=0 and look_back=6 so 5th element    goes from (len-k-1 to len -1)_ and e.g. a[1:2],a[2:3],a[3:4]  goes from
                #            a2 = dataset[i:(i + look_back),1] # 5 or 6? element slice... Stops at (len-1) e.g. if i=0 and look_back=6 so 5th element   goes from (len-k-1 to len -1)_ and e.g. a[1:2],a[2:3],a[3:4]  goes from
                dataX.clear()
                dataX.append(a1 * factor1)  # Price 1 elements
                dataX.append(a2 * factor2)  # Volume 1 elements
                dataXInter.append(list(dataX.copy()))
                # Trying to find why the interval does not come out correctly here
            aIntermediateList.append(dataXInter.copy())
            dataXInter.clear()
            dataYInter.clear()
            # Last iteration just append the last Y to the end
            if (i == len(dataset) - look_back - 1):  # have to subtract 1 to convert array length to index
                dataYInter.append(dataset[i])
            else:
                if (i + look_back + 1) < len(dataset):
                    dataYInter.append(numpy.float32(dataset[i + look_back + 1, 0]) * factor1)  # Y=Price
                if (i + look_back + 2) < len(dataset):
                    dataYInter.append(numpy.float32(dataset[i + look_back + 2, 0]) * factor1)
                else:
                    if (i + look_back + 2 < len(dataset) + 1):
                        dataYInter.append(0)
            if len(dataYInter) > 0:
                dataY.append(numpy.array(dataYInter.copy()))
        print("aIntermediateLIst")
        print(numpy.array(aIntermediateList).shape)
        print(aIntermediateList[-1:])
        print(aIntermediateList)
        print("Dataset")
        print(numpy.array(dataset).shape)
        print(dataset[-1:])
        print(dataset)
        print("Last elements)")
        print(dataset[-1:])
        print("................")
        print(aIntermediateList[-1:])
        print("shapes")
        print(len(dataY))
        print("dataY={}".format(numpy.array(dataY).shape))
        print("dataset={}".format(numpy.array(dataset).shape))
        print("aIntermediateList={}".format(numpy.array(aIntermediateList).shape))
        # exit(0)
        # OUT:  dataX: [[[Price1,Volume1]...[Price6],[Volume6]]...[[Price,Volume1]...[Price6,Volume6]]]
        # dataX Shape:(5,000,6,2)
        # in Fact if we follow Jason Brownlee at this url:
        # https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
        print(numpy.array(dataY))

        return numpy.array(aIntermediateList), numpy.array(dataY)

    numpy.random.seed(7)
    # dataframeAdj_Close = dffinalAdj_Close
    # dataframeVolume=dffinalVolume
    print("*********************************\n")
    print("Tail Values: \n", dataframeAdj_Close.tail(4))
    print("*********************************\n")
    datasetPrice = dataframeAdj_Close.values
    datasetPrice = datasetPrice.astype('float32')
    datasetPriceOnly = dataframePriceOnly.values
    datasetPriceOnly = dataframePriceOnly.astype('float32')
    if FunctionalAPI == True:
        datasetVolume = datasetPrice.values
        datasetVolume = datasetVolume.astype('float32')
    # normalize the datasetPrice
    scaler = MinMaxScaler(feature_range=(0, 1))
    # FIT_TRANSFORM CLEANS UP THE NULLS!!!
    # AND NORMALIZES!!!!
    # Error expected 2D error got 1D array
    Debug = True
    if Debug == True:
        print("**************************" + "\n")
        print("* Print the datasetPrice      *" + "\n")
        print("**************************" + "\n")
        print(datasetPrice)
        print(numpy.expand_dims(datasetPrice, axis=2).shape)
        print(numpy.array(datasetPrice).shape)
    datasetPrice = scaler.fit_transform(datasetPrice)

    datasetPriceOnly = scaler.fit_transform(datasetPriceOnly)
    if FunctionalAPI == True:
        datasetVolume = scaler.fit_transform(datasetVolume)
    # split into train and test sets
    train_size = int(len(datasetPrice) * 0.67)
    test_size = len(datasetPrice) - train_size
    if FunctionalAPI == True:
        train_sizeVolume = int(len(datasetVolume) * 0.67)
        test_sizeVolume = len(datasetVolume) - train_sizeVolume  # train and test are both 2 Dimensional Arrays
    # Break 67% into train and the rest of it into test
    # train gets both price and volume here!!
    train, test = datasetPrice[0:train_size, :], datasetPrice[train_size:len(datasetPrice),
                                                 :]  # 2 - 2 Dimensional Arrays
    if FunctionalAPI == True:
        trainVolume, testVolume = datasetVolume[0:train_sizeVolume, :], datasetVolume[
                                                                        train_sizeVolume:len(datasetVolume),
                                                                        :]  # 2 - 2 Dimensional Arrays
    if Debug == True:
        print("**************************")
        print("* Print out train & test *")
        print("**************************")
        for i in range(0, train_size):
            for j in range(0, 1):
                print(train[i, j])
        print("\r\n")
        print("**************************")
        print("Train={}".format(train))
        print("Train shape={}".format(train.shape))
    factor1 = 1;
    factor2 = .34
    look_forward = 0  # initially 0 but will be used later
    # trainX, trainY = makeTimeSeries(train, look_back, factor1, factor2)
    trainX, trainY = makeTimeSeries2(train, look_back, factor1, factor2)
    print("Train=================")
    print(train.shape)
    print("TrainX============================")
    print(trainX.shape)
    print("TrainY============")
    print(trainY.shape)
    # print(scaler.inverse_transform(numpy.array(trainY).reshape(-1, 1)[-10:]))
    if FunctionalAPI == True:
        trainXVolume, trainYVolume = makeTimeSeries2(trainVolume, look_back, factor1, factor2)
    trainX, trainY = makeTimeSeries2(train, look_back, factor1, factor2)

    testX, testY = makeTimeSeries2(test, look_back, factor1, factor2)
    if FunctionalAPI == True:
        testXVolume, testYVolume = makeTimeSeries2(testVolume, look_back, factor1, factor2)
    Debug = True
    if Debug == True:
        print("trainX", trainX)
        print("trainY", trainY)
        print("Shape trainX", trainX.shape)
        print("Shape testX", testX.shape)
        print("Shape trainY", trainY.shape)
        print("Shape testY", testY.shape)

    oldtestX = testX
    trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], trainX.shape[2]))
    testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], testX.shape[2]))

    if FunctionalAPI == True:
        trainXVolume = numpy.reshape(trainXVolume,
                                     (trainXVolume.shape[0], trainXVolume.shape[1], trainXVolume.shape[2]))
        testXVolume = numpy.reshape(testXVolume, (testXVolume.shape[0], testXVolume.shape[1], testXVolume.shape[2]))
    Debug = False
    if Debug == True:
        print("**********CHECKING LAST 2 VALUES IN DATASET****************")
        # YES WE DO GET TO THE LAST 2 VALUES IN THE DATASET!!!!!!!!!!!
        print("New TrainX=", scaler.inverse_transform(numpy.reshape(trainX[-1:], (1, -1))))
        print("New TrainX Shape=", trainX.shape)
        print("New TestX=", scaler.inverse_transform(numpy.reshape(testX[-1:], (1, -1))))
        print("New TestX Shape=", testX.shape)
        # START HERE 2/29/2020
        print(testY)
        Ybar = numpy.array(testY)
        print("***********************")
        print(Ybar)
        print("***********************")
        print("New TestY=", numpy.reshape(testY[:], (-1, 1)))
        print("***********************")
        print("TestY=", testY)
        # only for testing

        print("New TestY=", testY)

        print("New TestY=", scaler.inverse_transform(numpy.reshape(testY[-1:], (1, -1))))

        print("New TestY Shape=", testY.shape)
        print("***********************************************************")
        print("New Test last value=", scaler.inverse_transform(numpy.reshape(test[-1:], (1, -1))))
        print("dataset last value={}".format(
            scaler.inverse_transform(numpy.reshape(datasetPrice[len(datasetPrice) - 1], (1, -1)))))

    score = 0
    # http://www.itdaan.com/blog/2017/11/09/c2525310531416583200a3e14fbb8965.html
    if FunctionalAPI == True:
        # ***************************************************************
        # Beginning of Will Teslers Code

        import keras as ks
        # No activation functions for Input Layer
        price = Input(shape=(look_back, 1), name='price')
        volume = Input(shape=(look_back, 1), name='volume')
        HiddenSize = (look_back + 1) * 2
        priceLayers = [[0]]
        volumeLayers = [[0]]
        # priceLayers = LSTM(64, return_sequences=False)(price)
        # volumeLayers = LSTM(64, return_sequences=False)(volume)
        # tanh for hidden layer
        priceLayers = LSTM(HiddenSize, return_sequences=False, activation='tanh')(price)
        volumeLayers = LSTM(HiddenSize, return_sequences=False, activation='tanh')(volume)
        # ks.layers.LeakyReLU(alpha=0.3)
        output = ks.layers.Concatenate(axis=-1)([priceLayers, volumeLayers, ])
        # output = ks.layers.Concatenate([priceLayers, volumeLayers, ])
        # Dense is just a nn layer (usually and here for output)
        # Runs but may need Dense(1 to be Dense(look_back
        # Start here 8/11/2019
        output = Dense(1, name='weightedAverage_output', activation='relu')(output)
        model1 = Model(inputs=[price, volume], outputs=[output])
        # model1.compile(optimizer='rmsprop', loss ='mse')
        model1.compile(loss='mean_squared_error', optimizer='adam')
        # Beginning of my mod
        # Load Model with Data
        # https://machinelearningmastery.com/keras-functional-api-deep-learning/
        print("trainx=" + str(trainX.shape))
        print("trainy=" + str(trainY.shape))

        # Start Here change [trainX,trainX] to [trainX,trainXVolume] 8/26/2019

        model1.fit([trainX, trainXVolume], trainY, epochs=numEpochs, batch_size=1, verbose=2)
        score = model1.evaluate([trainX, trainXVolume], trainY, verbose=0)
        # exit(0)
        # End of Will Teslers Code
    else:
        # **************************************************************
        # Old Sequential model Begin
        # 1.) Create Model
        # https://machinelearningmastery.com/stacked-long-short-term-memory-networks/
        # https: // www.researchgate.net / publication / 236164660_Optimal_Neural_Network_Architecture_for_Stock_Market_Forecasting
        model = Sequential()
        HiddenSize = (look_back + 1) * 2
        # Original model.add(LSTM(HiddenSize, activation='tanh', input_shape=(look_back, 1)))
        # 11/3/2019 This in put shape should be (look_back,2)
        model.add(LSTM(HiddenSize, activation='tanh', input_shape=(look_back, 2), return_sequences=True))
        model.add(LSTM(math.ceil(HiddenSize * .5), activation='tanh',
                       return_sequences=True))  # Last layer does not need return_sequences
        model.add(LSTM(math.ceil(HiddenSize * .1), activation='tanh', return_sequences=False))
        model.add(Dense(2))
        model.add(LeakyReLU(alpha=.001))
        # model.add(Dense(1, input_dim=1, activation='relu'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        # https://stackoverflow.com/questions/22981845/3-dimensional-array-in-numpy
        print(trainX.shape)
        print("***************->trainX=", trainX)
        # Pretty sure I set this up as Trainx as t and TrainY as (t+1) all the way across the datasetPrice
        # Load Model with Data
        print(trainX.shape)
        print(trainY.shape)

        # "tail"
        # Find out what's in this trainY whether or not it's price, volume or both here!!
        print("tail----------------------------")
        # You are here must correctly reshape this NOT (1,-1)
        print("Shape={}".format(trainY.shape))
        print(trainY)
        print("--------------------")
        print("Shape={}".format(numpy.reshape(trainY, (1, -1)).shape))
        print(scaler.inverse_transform(numpy.reshape(trainY, (1, -1))))
        # exit(0)
        # Yes these are volumes
        # exit(0)!
        print("See array--------------")
        print(numpy.array(trainY)[:24])
        # Fit on trainX, trainY
        # Breaks here 2/5/2020
        print("shapes")
        print(testX.shape)
        print(testY.shape)
        print(testY)
        exit(0)
        # exit(0)
        model.fit(testX, testY, epochs=numEpochs, batch_size=1, verbose=2)
        score = model.evaluate(trainX, trainY, verbose=0)
        # exit(0)
        # Evaluate Description: https://stackoverflow.com/questions/51299836/what-values-are-returned-from-model-evaluate-in-keras
        print("2.) Old Sequential model end")
        print("# **********************************************************************")

    print("Evaluation score=?")
    print(model.metrics_names)
    print(score)
    if FunctionalAPI == True:
        trainPredict = model1.predict([trainX, trainXVolume])
    else:
        trainPredict = model.predict(trainX)
    # TrainPredict(although just price) is created from the model in both price and volume (from trainX) above

    # Scale this!!!!!!!!!!!!!!! it isn't scaled!!!!!!!!
    # exit(0)

    print("trainPredict prior=", trainPredict)
    if FunctionalAPI == True:
        testPredict = model1.predict([testX, testXVolume])
    else:
        testPredict = model.predict(testX)  # testX #n,6,2 testPredict n,1
    # Transform back to ORIGINAL UNSCALED data
    # You Are here 8/26/2019
    Debug = True
    if Debug == True:
        print("----testX------151--")
        # see if this shape is n,6,2? for testX
        print("testX Shape={}".format(testX.shape))
        print("Shows final 6,2 vector of testX and reshapes")
        print(scaler.inverse_transform(testX[-1:].reshape(-1, 1)))
        print("-----testPredict--129-----")
        print("Shows final price of testPredict")
        print(scaler.inverse_transform(testPredict[-1:].reshape(-1, 1)))
        print(numpy.array(testPredict).shape)
        print("Shows all of testPredict")
        print(scaler.inverse_transform(testPredict[:].reshape(-1, 1)))
        print("IN conclusion price and volume went in and Price only came out")
        # 12/16/2019
        # exit(0)
        print("----trainX--------")
        print(scaler.inverse_transform(trainX[:-1].reshape(-1, 1)))
        print("----trainPredict--------")
        print(scaler.inverse_transform(trainPredict[:-1].reshape(-1, 1)))
        print(numpy.array(trainPredict).shape)
        # exit(0)
    trainPredict = scaler.inverse_transform(trainPredict)
    print("trainY shape={}".format(trainY.shape))
    print("testY shape={}".format(testY.shape))
    print("train Y={}".format(trainY))

    # trainY1 = scaler.inverse_transform([trainY])
    trainY1 = scaler.inverse_transform(trainY)
    # exit(0)

    # trainY shape=(4064, 2)
    # testY shape=(1999, 2)
    # ValueError: Found array with dim 3. Estimator expected <= 2.
    # testY1 = scaler.inverse_transform([testY])
    testY1 = scaler.inverse_transform(testY)

    if Debug == True:
        print("trainPredict={}".format(numpy.array(trainPredict.shape)))
        print("trainPredict after=", trainPredict)
        print(numpy.array(testY1).shape)
        print(testY1)
        print("----------")
        print("testX")
        print(oldtestX[:-2])
        print("-----------")
        print(oldtestX[:2])
    testPredict = scaler.inverse_transform(testPredict)
    print("testPredict after=", testPredict)
    print("shape testPredict", testPredict.shape)
    # 11/23/2019 This is fine it outputs a nX1 array unlike what's in load_5A
    # 11/17/2019 You are here testY1 is good makeTimeSeries is good
    # but testPredict is bad and so is your plotting so fix them now
    # exit(0)
    # calculate root mean squared error of trainPredict and testPredict Models
    # Shape of trainPredict [2355,1] and testPredict [1159,1]  2D Arrays at this point
    Debug = True
    if Debug == True:
        print("data train Predict", trainPredict)
        print("shape trainPredict", trainPredict.shape)
        print("data test Predict", testPredict)
        print("shape testPredict", testPredict.shape)
        # print("shape trainY1", trainY1.shape)
        # print("data trainY1", trainY1)
    print("Point #1")
    print("trainY1[0]={}".format(trainY1[0]))
    print("trainPredict={}".format(trainPredict[:, 0]))

    # trainScore = math.sqrt(mean_squared_error(trainY1[0], trainPredict[:, 0]))
    trainScore = math.sqrt(mean_squared_error(trainY1[:, 0], trainPredict[:, 0]))

    print("Point #2")
    print('Train Score: %.2f RMSE' % (trainScore))
    # testScore = math.sqrt(mean_squared_error(testY1[0], testPredict[:, 0]))
    testScore = math.sqrt(mean_squared_error(testY1[:, 0], testPredict[:, 0]))
    print("Point #3")

    print('Test Score: %.2f RMSE' % (testScore))
    # from sklearn.metrics import accuracy_scoreaccuracy_score(df.actual_label.values, df.predicted_RF.values)
    # https://towardsdatascience.com/understanding-data-science-classification-metrics-in-scikit-learn-in-python-3bc336865019
    # shift train predictions for plotting
    # Returns a same Shape array as given array
    ##You are here 8/26/2019-----------------------------------------------
    print("Point #4")
    trainPredictPlot = numpy.empty_like(datasetPrice)
    trainPredictPlot[:, :] = numpy.nan
    testPredictPlot = numpy.empty_like(datasetPrice)
    testPredictPlot[:, :] = numpy.nan
    print("Point #5")
    #  model 2D = 2D arrays
    print("trainPredict shape=", numpy.array(trainPredict).shape)
    print("datasetPrice shape=", numpy.array(datasetPrice).shape)
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
    print("Completed trainPredictPlot")
    # testPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
    print("look_back:len(trainPredict) + look_back= {}:{},: ".format(look_back, len(trainPredict) + look_back))
    print("len(trainPredict) + (look_back * 2) + 1 ={} : len(datasetPrice) - 1={},:".format(
        len(trainPredict) + (look_back * 2) + 1, len(datasetPrice) - 1))
    print(numpy.array(trainPredict).shape)
    print(numpy.array(testPredict).shape)
    print(testPredictPlot.shape)
    # The rows are not big enough on the trainPredictPlot to accept the testPredict rows. The slice on trainPredict is too small-simply..
    print(len(trainPredict) + (look_back * 2) + 1, len(datasetPrice) - 1)
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(datasetPrice) - 1, :] = testPredict[
                                                                                        0:len(trainPredict) + (
                                                                                                look_back * 2) + 1:len(
                                                                                            datasetPrice) - 1]  # testPredict[0:2798]
    print("Completed testPredictPlot")
    testPredictPlot = numpy.empty_like(datasetPrice)
    testPredictPlot[:, :] = numpy.nan
    #  model 2D = 2D arrays
    print("testPredict shape=", numpy.array(testPredict).shape)
    print("datasetPrice shape=", numpy.array(datasetPrice).shape)
    print(len(trainPredict) + (look_back * 2) + 1, len(datasetPrice) - 1)
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(datasetPrice) - 1, :] = testPredict[
                                                                                        0:len(trainPredict) + (
                                                                                                look_back * 2) + 1:len(
                                                                                            datasetPrice) - 1]  # testPredict[0:2798]
    print("Completed testPredictPlot")
    # plot baseline and predictions
    # Add Plot for the datasetPrice
    plt.xlabel('Days from beginning')
    plt.ylabel('Price')
    plt.title('Stock Prediction for ' + stockName)
    plt.grid(True)
    plt.plot(scaler.inverse_transform(datasetPrice), color='b', lw=0.5, label='datasetPrice')
    # Add Plot for the training model
    plt.plot(trainPredictPlot, color='r', lw=0.5, label='Training')
    # Add Plot for the test model
    plt.plot(testPredictPlot, color='g', lw=0.5, label='Test')
    # Do the Plot onscreen
    plt.legend(loc=1, fontsize="x-large")
    leg = plt.legend()
    # get the lines and texts inside legend box
    leg_lines = leg.get_lines()
    leg_texts = leg.get_texts()
    # bulk-set the properties of all lines and texts
    plt.setp(leg_lines, linewidth=4)
    plt.setp(leg_texts, fontsize='x-large')
    # plt.show()
    # https://stackoverflow.com/questions/741877/how-do-i-tell-matplotlib-that-i-am-done-with-a-plot
    plt.clf()
    plt.cla()
    plt.close()
    # https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
    # *********************************
    # Part 2 Write the model to disk *
    # serialize model to JSON        *
    # *********************************

    # Changed 9/3/2019
    # if FunctionalAPI == True:
    #    model_json = model1.to_json()
    # else:
    #    testPredict = model.predict(testX)

    # PROBLEM HERE 9/3/2019!!!!!!!!!!!!!!
    if FunctionalAPI == True:
        model_json = model1.to_json()
    else:
        model_json = model.to_json()

    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    if FunctionalAPI == True:
        model1.save_weights("model.h5")
    else:
        model.save_weights("model.h5")
    print("Saved model to disk")
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    # evaluate loaded model on test data
    # loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    loaded_model.compile(loss='mean_squared_error', optimizer='adam')
    #    score = loaded_model.predict(trainX, trainY, verbose=0)
    # print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
    return (0)

"""
Loads the model
dateListClose:
dateListVolume:
stockName: Name of equity
lookback: Domain days of function.
lookForward: Range days forward.
blocks:
showOnlyGainers: Shows only predicted moves upwards
"""
def Load_model_7A(dateListClose, dateListVolume, stockName, lookback, lookForward, blocks, showOnlyGainers,
                  FunctionalAPI,
                  lstSelection, showGraphs, blnChangeUp):
    """
    Loads multivalue model
    Parameters:
        dateLIstClose:Dictionary/Json
        dateListVolume:Dictioary/Json
        stockName:String
        lookback:integer
        blocks:integer
        showOnlyGainers:boolean
        FunctionslAPI:boolean
    """
    # Additional Comments
    # Stop here and put in dateListVolume 9/1/20109
    # convert to n,6,2 or n samples,6 timesteps and 2 feature
    # 10/24/19
    # Desired shape for combination of dateListClose and dateListVolume is n,6,2
    # How to combine these most efficiently?
    # Price (36,) Volume (36,)
    # The short answer is
    # n,6,2 or 6,6,2 for a total of 72(36+36)
    # n,6,(Price,Volume)
    # https://www.geeksforgeeks.org/python-merge-two-lists-into-list-of-tuples/
    # So to reshape this do tomorrow
    lenClose = len(dateListClose)
    print(numpy.array(dateListClose).shape)
    print(numpy.array(dateListVolume).shape)
    print(dateListClose)
    # 1.) Pair them into a new array of [(P,V),(P,V)....(Pn,Vn)]
    # 2.) Reshape the new array into 36,lookback*blocks,2
    # Goal in this sample is 6,6,2
    # 3.) NewArray=numpy.reshape(newarray, dateListClose.size/lookback,lookback,2)
    dateListCloseArr = []
    dateListVolumeArr = []
    dualArray = numpy.array([[]])  # 2D Array
    # https://www.pluralsight.com/guides/different-ways-create-numpy-arrays
    dualArray = numpy.zeros((lenClose, 2), dtype=float)
    print(lenClose)
    for i in range(lenClose):
        dualArray[i][0] = numpy.array(dateListClose)[i]
        dualArray[i][1] = numpy.array(dateListVolume)[i]
    print(dualArray)
    print(numpy.array(dualArray).shape)
    print(int(lenClose / lookback))
    # Temporarily need this shape
    # (5675, 2, 6)
    # finalArray=numpy.reshape(dualArray,(int(lenClose/lookback),lookback,2))
    #   finalArray = dualArray
    print("dualArray={}".format(dualArray))
    import matplotlib.pyplot as plt
    scaler = MinMaxScaler(feature_range=(0, 1))

    dualArrayfitted = scaler.fit_transform(dualArray)
    # refit the scaler for a single price vector
    # scaler.inverse_transform(numpy.reshape(score, (-1, 2))))
    print("Final re-fitting of scaler to one vector price")
    dataListArrfittedPriceOnly = scaler.fit_transform(numpy.reshape(numpy.array(dateListClose), (-1, 1)))

    blocks = len(dateListClose) / (lookback)
    # LOAD MODEL
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    loaded_model.compile(loss='mean_squared_error', optimizer='adam')
    # ERROR is HERE!!!
    print(dualArrayfitted.shape)
    # Shapes this into original LSTM input shape for inverse transform e.g. shape = (n,6,2)

    #    diagnosticArray = numpy.reshape(scaler.fit_transform(dualArray), (int(lenClose/ lookback), lookback,2 ))
    #    print("diagnosticArray shape={}".format(diagnosticArray.shape))

    # Shapes this into original LSTM input shape for inverse transform e.g. shape = (n,6,2)
    # score = loaded_model.predict(numpy.reshape(scaler.fit_transform(dualArray), (int(lenClose/ lookback), lookback,2 )), verbose=0)
    score = loaded_model.predict(numpy.reshape(dualArrayfitted, (int(lenClose / lookback), lookback, 2)), verbose=0)
    print("Actual shape of input array={}".format(
        numpy.reshape(dualArrayfitted, (int(lenClose / lookback), lookback, 2))))
    print("last price/volume vals of dualArray={}".format(dualArray[-1:]))
    print(numpy.array(score).shape)
    print("Score={}".format(score))
    # Your are here 12/16/19
    # (output is 6,1)  but gets transfered back to (3,2) for the inverse transform
    print('***********************************************************')
    print('*                 Scores comparison                       *')
    print('***********************************************************')
    print(
        "This should have produced only price vector but produces a prediction that is 3 rows of price/volume pairs(weird) or 3 days in this case")
    print("#2.) Final Predicted Score:\n", scaler.inverse_transform(numpy.reshape(score, (-1, 2))))
    print('')
    odd = False
    if len(score) % 2 == 0:
        print("#3.) Transformed Predicted Score:\n", scaler.inverse_transform(numpy.reshape(score, (-1, 2))))
    else:
        odd = True
        print("odd")
        # The append doesn't work yet
        print("**********************************")
        print("The Score =", score)
        score = numpy.append(score, [[0.0]], axis=0)
        print("The new score=", score)
        print("#3.) Transformed Predicted Score:\n", scaler.inverse_transform(numpy.reshape(score, (-1, 2))))
    # TRANSFORM HAS TO BE 2 COLUMNS (WHY? DUNNO)
    trainPredictPlot = scaler.inverse_transform(numpy.reshape(score, (-1, 2)))
    print("Train Predict Plot:\n", trainPredictPlot)
    # *****************
    # Add Plot for the dataset
    # *****************
    plt.xlabel('Days from beginning')
    plt.ylabel('Price')
    if showOnlyGainers == True:
        plt.title('Gainers Stock Prediction from Saved model for ' + stockName)
    else:
        plt.title('Losers Stock Prediction from Saved model for ' + stockName)

    plt.grid(True)
    # plt.plot(scaler.inverse_transform(dataset), color='b', lw=0.5, label='Dataset')
    # Add Plot for the training model
    # (-1,1) should turn it into a flat list
    print("RESHAPED PREDICT PLOT", numpy.reshape(trainPredictPlot, (-1, 1)))
    # You are here 11/20/19
    # if odd subtract dummy element from 1 column array

    # Put this section back in 12/15/19
    # if odd == True:
    # numpy.tail DELETE LAST ROW OF 2D ARRAY
    #    trainPredictPlot = numpy.reshape(trainPredictPlot, (-1, 1))[:-1, :]
    # else:
    #    trainTemp = trainPredictPlot[:,0] #Turn from n,2 to n,1 array(selected 0th dimension)
    # trainPredictPlot = numpy.reshape(trainPredictPlot, (-1, 1)) # Remove rows and turn into columns
    #    trainPredictPlot =trainTemp

    trainPredictPlot = numpy.reshape(trainPredictPlot, (-1, 1))

    print("trainPredictPlot=")
    print(trainPredictPlot)
    print(trainPredictPlot.shape)
    days = [];
    q = 0
    # correct this to full range!!!!!!!!!!!!!!ur here 12/16/2019
    print("---------------------------")

    for idx in range(len(trainPredictPlot)):
        q = q + 1
        days.append(q)
    xi = list(range(len(days)))
    plt.xticks(xi, days)
    plt.plot(xi, trainPredictPlot, color='r', lw=1.0, label='window of ' + str(len(trainPredictPlot)) + ' days')
    # plt.plot(trainPredictPlot, color='r', lw=1.0, label='window of ' + str(len(trainPredictPlot)) + ' days')
    # plt.plot(numpy.reshape(trainPredictPlot,(-1,1)), color='r', lw=0.5, label='Window of 8 days')
    # plt.plot(numpy.reshape(trainPredictPlot,(-1,1)), color='r', lw=0.5, label='Window of 8 days')
    # Add Plot for the test model
    # plt.plot(testPredictPlot, color='g', lw=0.5, label='Test')
    # Do the Plot onscreen
    # plt.legend(loc=1, fontsize="x-large")
    leg = plt.legend()
    # get the lines and texts inside legend box
    leg_lines = leg.get_lines()
    leg_texts = leg.get_texts()
    # bulk-set the properties of all lines and texts
    plt.setp(leg_lines, linewidth=4)
    plt.setp(leg_texts, fontsize='x-large')
    print("Lastelement1=")
    print(trainPredictPlot)
    print(trainPredictPlot[trainPredictPlot.size - 1])
    delta = trainPredictPlot[trainPredictPlot.size - 1] - trainPredictPlot[trainPredictPlot.size - 2]
    if showOnlyGainers == True:
        if ((trainPredictPlot[trainPredictPlot.size - 1] - trainPredictPlot[trainPredictPlot.size - 2]) > 0):
            lstSelection.append("^Stock {} will go up by {}".format(stockName, str(delta)))
            if showGraphs == True:
                plt.show()
        else:
            lstSelection.append("Stock {} will go down by {}".format(stockName, str(delta)))
    else:
        if ((trainPredictPlot[trainPredictPlot.size - 1] - trainPredictPlot[trainPredictPlot.size - 2]) <= 0):
            lstSelection.append("Stock {} will go down by {}".format(stockName, str(delta)))
            if showGraphs == True:
                plt.show()
        else:
            lstSelection.append("^Stock {} will go up  by {}".format(stockName, str(delta)))

    plt.clf()
    plt.cla()
    plt.close()
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    print('***********************************************************')
    print('*                 FINAL PREDICTION VECTOR                 *')
    print('***********************************************************')
    print(
        "Score Direct from prediction: There are 8 3 value window inputs TO THE MODEL and one (8X1) output coming out here...\n",
        score)
    print(
        "...only last value of the (8X1) score vector is an actual prediction. The rest are predicted from prior 3 value windows")
    print("Transformed Score:\n", scaler.inverse_transform(numpy.reshape(score, (-1, 2))))  # - row with 2 columns
    varTransform = scaler.inverse_transform(numpy.reshape(score, (-1, 2)))
    # Point #5

    # solutionVector = numpy.reshape(varTransform, len(score), -1)
    solutionVector = varTransform

    print("solutionVector--feed this to your json function")
    print(solutionVector)  # 8X1 of solutions(?)
    print("Transformed 8 windows of 3 output vector:", solutionVector)

    print('***********************************************************')
    print('*                 FINAL PREDICTION                         *')
    print('***********************************************************')
    # Working here 6/21/2019
    # print("Price PREDICTION={}".format(solutionVector[int(solutionVector.size/2)-1]))
    # print("Price PREDICTION={}".format(solutionVector[solutionVector.size - 1]))
    # if odd==True:
    #    print("PREDICTION=",solutionVector[len(score)-1])
    # else:
    #    print("PREDICTION=", solutionVector[len(score)])
    # Probably should delete the below code
    # Leave the rest of this alonetps://stackoverflow.com/questions/
    # final = scaler.inverse_transform(numpy.reshape(score, (3, 1)))
    # print("#4.) Transformed score with artifacts=", final)
    # print("#4.) Transformed score without artifacts=",final[:,[0]] ) #ht49330195/how-to-use-inverse-transform-in-minmaxscaler-for-a-column-in-a-matrix
    # print(score[1] * 100)
    # print(loaded_model.metrics_names[0])
    # exit(0)

    # if blnChangeUp == True:
    #    print("%s: %.2f%% %.2f%%" % (loaded_model.metrics_names[0], score[1, 0] * 100, score[1, 1] * 100))
    # else:
    #    print("%s: %.2f%%" % (loaded_model.metrics_names[0], score[1] * 100))
    print(" %s%%" % (loaded_model.metrics_names))
    print("score = %s" % (score))
    print("lstSelection={}".format(lstSelection))

    # return (listToJsonConverter(solutionVector.tolist()))
    return (lstSelection)


# MAIN()
# retCode = NumpyReshapeExamples()
# retCode = PimaIndians()
# retCode = LSTM_Forecast()
# retCode = majorLocator()

# Quandl lags by a month
def getValuesFromNet(stockName, numVals, choice):
    # Gets last numVals values for Prediction(later)
    import quandl
    import pandas as pd
    quandl.ApiConfig.api_key = "6X-hxfxiPcSb3uTS2UyB"
    # df = quandl.get("WIKI/KR")
    # df = quandl.get("EOD/MSFT")
    # df = quandl.get("SHARADAR/SEP",ticker='AAPL')
    # df = quandl.get_table('SHARADAR/SEP', ticker=stockName)
    df = quandl.get(stockName)
    print(df.tail(numVals))
    for index, row in df.head().iterrows():
        print(index, row['Open'], row['High'], row['Adj_Close'])
    if choice == 'Adj_Close':
        dilbert = df['Adj_Close'].to_dict()
    if choice == 'Volume':
        dilbert = df['Volume'].to_dict()

    if choice == 'Adj_Close':
        dffinal = pd.DataFrame.from_dict(dilbert, orient='index', columns=["Adj Close"])
    if choice == 'Volume':
        dffinal = pd.DataFrame.from_dict(dilbert, orient='index', columns=["Volume"])
    Debug = False
    # This actually splits upa nd creates your
    # Independent and Dependent Variables
    # But it's complicated and you need to get back to it.
    # look_back = 3
    numpy.random.seed(7)
    dataframe = dffinal
    if choice == 'Adj_Close':
        lastValsList = dataframe['Adj Close'].tail(numVals).tolist()
    if choice == 'Volume':
        lastValsList = dataframe['Volume'].tail(numVals).tolist()
    return (lastValsList)


def getValuesFromNet2(stockName, numVals):
    """parameters:
        stockname:string
        numVales: integer
    """
    # Gets last numVAls Values for prediction(later)
    # EOD df = quandl.get(stockName)
    # EOD for index, row in df.head().iterrows():
    # EOD    print(index, row['Open'], row['High'], row['Adj_Close'])
    # EOD dilbert = df['Adj_Close'].to_dict()
    import quandl
    quandl.ApiConfig.api_key = "6X-hxfxiPcSb3uTS2UyB"
    df = quandl.get_table('SHARADAR/SEP', ticker=stockName)
    print(df.tail(numVals))
    dilbert = df['closeunadj'].to_dict()
    dizenario = {};
    i = 0;
    for keyval in dilbert:
        dizenario[i] = dilbert[keyval]
        i = i + 1

    import pandas as pd
    dffinal = pd.DataFrame.from_dict(dizenario, orient='index', columns=["Adj Close"])
    Debug = False
    # This actually splits upa nd creates your
    # Independent and Dependent Variables
    # But it's complicated and you need to get back to it.
    look_back = 3

    numpy.random.seed(7)
    dataframe = dffinal
    lastValsList = dataframe['Adj Close'].tail(numVals).tolist()
    return (lastValsList)


def getVolatilesCSV2(listLimit, hyperList):
    """
    A Volatility calculator using a ticker list of most volatilies.
    Calculates volatility. Formerly used in place of FinViz and
    may have to be rerealized for nasdaq penny stocks
    parameters:
    listLimit: integer
    hyperlist: []
    """
    # https://www.fool.com/knowledge-center/how-to-calculate-annualized-volatility.aspx
    import requests
    import csv
    import statistics
    from datetime import datetime, timedelta
    timeB = datetime.today().strftime("%m-%d-%Y")
    timeA = (datetime.today() - timedelta(days=1)).strftime("%m-%d-%Y")
    timeMinus30 = (datetime.today() - timedelta(days=30)).strftime("%m-%d-%Y")
    timeMinus15 = (datetime.today() - timedelta(days=15)).strftime("%m-%d-%Y")
    timePlus15 = (datetime.today() + timedelta(days=15)).strftime("%m-%d-%Y")
    url = 'https://s3.amazonaws.com/quandl-production-static/end_of_day_us_stocks/ticker_list.csv'
    # url = 'OTCBB_20190703.csv'
    response = requests.get(url)

    quandl.ApiConfig.api_key = "6X-hxfxiPcSb3uTS2UyB"
    if response.status_code != 200:
        print('Failed to get data:', response.status_code)
    else:
        if len(hyperList) > 0:
            wrapper = hyperList

        else:
            wrapper = csv.reader(response.text.strip().split('\n'))
            print("getting hyperlist")
        x = 0
        stockList = []
        volatilityList = []
        list1 = []
        dict1 = {}
        print("********************")
        for record in wrapper:
            if len(hyperList) > 0:
                stockName = record
                print("Getting hyperlist2")
            else:
                stockName = record[0]
            if stockName != 'Ticker':
                x = x + 1
                if x <= listLimit or listLimit == -666:
                    # try:
                    print("((((((((((((((()))))))))))))))")
                    # stockNameQuandl="EOD/"+"AAPL"
                    if len(hyperList) > 0:
                        stockNameQuandl = "EOD/" + record
                    else:
                        stockNameQuandl = "EOD/" + record[0]
                    # Put previous business day calculator here!!!!
                    print(timeA)
                    print(timeB)
                    print("Stockname=" + stockNameQuandl)
                    # https://stackoverflow.com/questions/22898824/filtering-pandas-dataframes-on-dates
                    # df = quandl.get(stockNameQuandl, start_date=timeA, end_date=timeB)
                    # [timeA-30] with timeB as RHS
                    try:
                        df = quandl.get(stockNameQuandl)
                        dfSubset = df[timeA:timeB]
                        dfMonthly = df[timeMinus15:timeB]  # timeMinus30
                        idx = 0
                        hval = [len(dfMonthly)]
                        print("Length of Data=" + str(len(dfMonthly)))
                        for idx in range(0, len(dfMonthly) - 1):
                            PPrev = dfMonthly.iloc[idx]['Adj_Close']
                            PNext = dfMonthly.iloc[idx + 1]['Adj_Close']
                            DailyPCTChange = ((PNext - PPrev) / PPrev) * 100
                            # print("PPrev="+str(PPrev)+"\n")
                            hval.append(DailyPCTChange)
                        Volatility = statistics.stdev(hval) * math.sqrt(10)  # 21
                        stockList.append(stockName)
                        volatilityList.append(Volatility)
                        dict1.update({stockName: Volatility})
                    except:
                        print("Quandl Error")

                # except:
                #    x = x - 1
                #    print("Quandl Symbol Error1\n")
    return (dict(sorted(set(zip(stockList, volatilityList)), key=lambda x: (-x[1], x[0]))))


# notes:
# Real estate quandl.get("NAHB/HOUSEACT", authtoken="6X-hxfxiPcSb3uTS2UyB")
# Year	Total Housing Starts	Single Family Housing Starts	Multi-Family Housing Starts	New Single-Family Sales	Existing Single-Family Home Sales
def MacD():
    """
    MacD calculator. Using volatiliy to calculate MacD
    Status: Formerly a module of the system was replaced by FinViz but
    may have to be rerealized for Nasdaq penny stocks
    """
    # https://towardsdatascience.com/implementing-macd-in-python-cc9b2280126a
    import pandas as pd
    import numpy as np
    from datetime import datetime
    import matplotlib.pyplot as plt
    import quandl
    from datetime import datetime, timedelta
    timeB = datetime.today().strftime("%m-%d-%Y")
    timeA = (datetime.today() - timedelta(days=1)).strftime("%m-%d-%Y")
    timeMinus30 = (datetime.today() - timedelta(days=30)).strftime("%m-%d-%Y")
    timeMinus15 = (datetime.today() - timedelta(days=15)).strftime("%m-%d-%Y")
    timePlus15 = (datetime.today() + timedelta(days=15)).strftime("%m-%d-%Y")
    stockList = []
    volatilityList = []
    list1 = []
    dict1 = {}
    quandl.ApiConfig.api_key = "6X-hxfxiPcSb3uTS2UyB"
    ticker = "EOD/AMD"
    df = quandl.get(ticker)
    timeframe = '6m'
    #    df = p.chartDF(ticker, timeframe)
    dfSubset = df[timeA:timeB]
    dfMonthly = df[timeMinus30:timeB]

    df = df['Adj_Close']
    #    df.reset_index(level=0, inplace=True)
    #    df.columns = ['ds', 'y']
    #    plt.plot(df, df.y, label='AMD')
    #    plt.show()

    exp1 = df.ewm(span=12, adjust=False).mean()
    exp2 = df.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    exp3 = macd.ewm(span=9, adjust=False).mean()
    plt.plot(df, macd, label='AMD MACD', color='#EBD2BE')
    plt.plot(df, exp3, label='Signal Line', color='#E5A4CB')
    plt.legend(loc='upper left')
    plt.show()
    exit(0)
    exp1 = df.ewm(span=12, adjust=False).mean()
    exp2 = df.ewm(span=26, adjust=False).mean()
    exp3 = df.ewm(span=9, adjust=False).mean()
    macd = exp1 - exp2.plot(df.ds, df.y, label='AMD')
    plt.plot(df, macd, label='AMD MACD', color='orange')
    plt.plot(df, exp3, label='Signal Line', color='Magenta')
    plt.legend(loc='upper left')
    plt.show()

    return (0)

# https://medium.com/@hr23232323/building-a-stock-screener-for-free-in-python-1d1d334eb76d

def FinVizScreener2(setLimit, metrics):
    """
    Screens stocks from FinViz
    Parameters
    -----------
    setLimit: integer
    metrics: = {'RSI (14)RH':70,'P/ERH':28}
    """

    # Document all filters 7/21/2019
    # Filter Columns= +/- No., Ticker, Company, Sector, Industry, Country, Market Cap, P/E, Price, Change,Volume
    # filters=['n_majornews,'exch_nasd', 'idx_sp500']'])

    # filters = ['exch_nasd', 'idx_sp500','sh_price_u1']  # Shows companies in NASDAQ which are in the S&P500
    # filters = ['exch_nasd', 'idx_sp500']
    # filters = ['sh_price_u1']  # Shows companies in NASDAQ which are in the S&P500
    filters = ['sh_price_u1']
    # filters = ['exch_nasd', 'idx_sp500']
    # filters = ['exch_amex','fa_div_high']
    # Get the first 50 results sorted by price ascending
    stock_list = Screener(filters=filters, order='-Change')
    # Export the screener results to .csv
    # 5/22/2020 removed stock_list.to_csv()
    # Cr    eate a SQLite database
    # stock_list.to_sqlite()
#    stock_list.add(filters=['n_majornews'])
    print(stock_list)
    # for stock in stock_list[9:19]:  # Loop through 10th - 20th stocks
    listTickers = []
    outPut = False
    counter = 0
    decideForFilter = 0
    for stock in stock_list:
        counter = counter + 1
        if counter < setLimit:
            dict1A = finviz.get_stock(stock['Ticker'])
            decideForFilter = 0
            for key in dict1A:
                # Set up "AND" Filter
                # https://stackoverflow.com/questions/6812031/how-to-make-unicode-string-with-python3
                # https://stackoverflow.com/questions/5424716/how-to-check-if-string-input-is-a-number
                silError = False

                # try:
                #    val = float(str(dict1A[key]))
                # except ValueError:
                # print("That's not an float!"+str(dict1A[key]))
                #    silError = True

                # New Gross Margin check if float..strip %
                try:
                    val = float(str(dict1A[key]).strip('%'))
                    silError = False
                except ValueError:
                    silError = True

                if key == 'RSI (14)':
                    if silError == False:
                        # print("RSI="+ str(dict1A[key]))
                        if float(dict1A[key]) <= metrics[key + 'RH']:  # 70:
                            decideForFilter += 1
                if key == 'P/E':
                    if silError == False:
                        # print("P/E:=" + str(dict1A[key]))
                        if float(dict1A[key]) <= metrics[key + 'RH']:  # 28:
                            decideForFilter += 100
                # beginning of new stuff 12/19/2019
                # if key == 'Debt/Eq':
                #    if silError == False:
                #        if float(dict1A[key]) <= metrics[key+'RH']:
                #            decideForFilter += 0
                if key == 'Gross Margin':
                    if silError == False:
                        if (float(str(dict1A[key]).strip('%')) <= metrics[key + 'RH']) and (
                                float(str(dict1A[key]).strip('%')) >= metrics[key + 'LH']):
                            decideForFilter += 3
                # if key == 'Quick Ratio':
                #    if silError == False:
                #        if (float(str(dict1A[key])) <= metrics[key+'RH']) and (float(str(dict1A[key])) >= metrics[key+'LH']):
                #            decideForFilter += 3
                if key == 'Debt/Eq':
                    if silError == False:
                        if (float(str(dict1A[key]).strip('%')) <= metrics[key + 'RH']) and (
                                float(str(dict1A[key]).strip('%')) >= metrics[key + 'LH']):
                            decideForFilter += 3
                # if key == 'LT Debt/Eq':
                # Has to meet all criteria to add up to 107
                try:
                    # if decideForFilter == 101:
                    # if decideForFilter == 104: then 107 THEN 110
                    if decideForFilter == 107:
                        # print("Appending List Ticers")
                        listTickers.append(stock['Ticker'])
                        decideForFilter = -666
                    if outPut == True:
                        print(stock['Ticker'], stock['Price'], stock['Volume'])  # Print symbol and price
                        print(
                            "___________________________________________________________________________________________________________________________________________________________________")
                        print(finviz.get_stock(stock['Ticker']))
                        print("********NEWS NEWS NEWS********")
                        print(finviz.get_news(stock['Ticker']))
                        print(finviz.get_insider(stock['Ticker']))
                except:
                    print("Error:Couldn't Retrieve News about " + stock['Ticker'] + " from FinViz.")
        # stock_list.add(filters=['fa_div_high'])  # Show stocks with high dividend yield or just stock_list(filters=['fa_div_high'])
        else:
            break;
    return (listTickers)

# https://xang1234.github.io/scrapingfinviz/
# https://github.com/xang1234/Finviz-Scraper/blob/master/Finviz-Scraper.p
def scrape_finviz(symbols):
    """
    scrape_finviz takes additional information out of finviz related to volatility
    Parameters:
        symbols:string
    """
    import requests

    from bs4 import BeautifulSoup
    import pandas as pd
    import progressbar
    # Get Column Header
    from urllib.request import Request, urlopen

    req = Request('https://finviz.com/quote.ashx?t=FB', headers={'User-Agent': 'XYZ/3.0'})
    webpage = urlopen(req, timeout=10).read()

    # req = requests.get("https://finviz.com/quote.ashx?t=FB")
    soup = BeautifulSoup(webpage, 'html.parser')
    table = soup.find_all(lambda tag: tag.name == 'table')
    rows = table[8].findAll(lambda tag: tag.name == 'tr')
    out = []
    for i in range(len(rows)):
        td = rows[i].find_all('td')
        out = out + [x.text for x in td]

    ls = ['Ticker', 'Sector', 'Sub-Sector', 'Country'] + out[::2]

    dict_ls = {k: ls[k] for k in range(len(ls))}
    print("dict_ls={}".format(dict_ls))
    print("ls={}".format(ls))
    # exit(0)

    df = pd.DataFrame()
    p = progressbar.ProgressBar()
    p.start()

    for j in range(len(symbols)):

        p.update(j / len(symbols) * 100)
        # req = requests.get("https://finviz.com/quote.ashx?t=" + symbols[j])
        req = Request('https://finviz.com/quote.ashx?t=' + symbols[j], headers={'User-Agent': 'XYZ/3.0'})

        # if req.status_code != 200:
        #    continue
        webpage = urlopen(req, timeout=10).read()
        soup = BeautifulSoup(webpage, 'html.parser')
        table = soup.find_all(lambda tag: tag.name == 'table')

        rows = table[6].findAll(lambda tag: tag.name == 'tr')
        sector = []
        for i in range(len(rows)):
            td = rows[i].find_all('td')
            sector = sector + [x.text for x in td]
        sector = sector[2].split('|')
        rows = table[8].findAll(lambda tag: tag.name == 'tr')
        out = []
        for i in range(len(rows)):
            td = rows[i].find_all('td')
            out = out + [x.text for x in td]
        out = [symbols[j]] + sector + out[1::2]
        out_df = pd.DataFrame(out).transpose()
        df = df.append(out_df, ignore_index=True)

    p.finish()

    # df.sort_values(by='col1', ascending=False)
    df = df.rename(columns=dict_ls)

    if 'Volatility' in df.columns:
        df = df.sort_values(by='Volatility', ascending=False)
    # else:
    #    df = df.sort_values(by='P/E',ascending=True)

    return (df)

# To do 9/6/2020
# For Finviz get on this site for Chquopy
# https://github.com/chaquo/chaquopy/issues
# Ask for a package request Finviz 1.3.4
# Tell him it appears to build from gradle but isn't available in code
# Push this all up to gibhub first!!!!!!!!!!!!!!!!!!!!!
    """
    Main:Processes 1 STOCK AT A TIME
    Symbol: string
    choice: integer
    showOnlyGainers: boolean
    Epochs: integer
    skipLSTM: boolean
    """
def Main(Symbol, choice, showOnlyGainers, Epochs, skipLSTM, lstSelection, blnChangeUp, showGraphs, lookBack,
         lookForward, segments):


    # Symbol = "CDMO"
    stockName = "EOD/" + Symbol  # EOD dataset for getValuesFromNet and LSTM_model_4A

    # Epochs = 1
    # is 78 bad?
    # 6/29/2019 was lastValsList = getValuesFromNet(stockName, (lookBack+1) * segments)  # get 24 or some other multiple of 6
    FunctionalAPI = False
    tempCorrectionFactor = 1  # 2 for load_model_6a
    lastValsListClose = getValuesFromNet(stockName, tempCorrectionFactor * (lookBack) * segments,
                                         'Adj_Close')  # get 24 or some other multiple of 6
    lastValsListVol = getValuesFromNet(stockName, tempCorrectionFactor * (lookBack) * segments, 'Volume')

    # try:
    if choice == 1:
        # Quandl EOD
        print("Start LSTM_model_8A")
        if skipLSTM is not True:
            retCode = LSTM_model_10A(Epochs,
                                     stockName,
                                     lookBack,
                                     lookForward,
                                     FunctionalAPI)
            print("Start Load_model_4A")
        # retJSON = Load_model_7A(lastValsListClose,
        lstSelection = Load_model_7A(lastValsListClose,
                                     lastValsListVol,
                                     stockName,
                                     lookBack,
                                     lookForward,
                                     segments,
                                     showOnlyGainers,
                                     FunctionalAPI,
                                     lstSelection,
                                     showGraphs,
                                     blnChangeUp)
        print("*************************************************************")
        # print("Json output prediction ", retJSON)
        print("The Predictions are: {}".format(lstSelection))
        print("*************************************************************")

    return (lstSelection)

"""
Includes Loop for Days forward
"""
"""
Main entry point
Parameters:
screenerLimit:  integer
hyperlist:      array
screener:       boolean
"""
showGraphs = False
screenerLimit = 200
hyperList = ['SHYF','F','DXYN','DXF','ESCRQ','HPNN','SFMI','VKIN']
hyperList = ['F']
screener = True
lstSelection = []
blnChangeUp = False
getWinner = True
Epochs = 1
skipLSTMModel = False
getVolatiles = True
lookBack = 6  # number of training data points
segments = 6  # Multipler of data (points + 1)
daysforward = 0  # YValue
# for complete dictionary list see file PythonMetrics.odt in this project
# Dictionary  name |LH |RH and value
metrics = {'RSI (14)RH': 70, 'P/ERH': 28, 'Debt/EqRH': 2, 'Debt/EqLH': 0, 'Gross MarginRH': 100, 'Gross MarginLH': 15,
           'Quick RatioLH': 1.0, 'Quick RatioRH': 100}
# metrics = {'RSI (14)RH': 70, 'P/ERH': 28, 'Debt/EqRH': 2, 'Debt/EqLH': 0, 'Gross MarginRH': 100, 'Gross MarginLH': 15}
# metrics = {'RSI (14)RH':70,'P/ERH':28}
# {'Index': '-', 'P/E': '-', 'EPS (ttm)': '-12.74', 'Insider Own': '0.06%', 'Shs Outstand': '100.00M', 'Perf Week': '154.20%', 'Market Cap': '9.99M',
# 'Forward P/E': '-', 'EPS next Y': '-', 'Insider Trans': '0.00%', 'Shs Float': '99.53M', 'Perf Month': '73.49%', 'Income': '-34.30M', 'PEG': '-',
# 'EPS next Q': '-', 'Inst Own': '2.50%', 'Short Float': '1.34%', 'Perf Quarter': '-23.86%', 'Sales': '5.50M', 'P/S': '1.82', 'EPS this Y': '-',
# 'Inst Trans': '109.91%', 'Short Ratio': '0.16', 'Perf Half Y': '-84.03%', 'Book/sh': '0.49', 'P/B': '0.20', 'ROA': '-158.40%', 'Target Price': '-',
# 'Perf Year': '-96.39%', 'Cash/sh': '0.01', 'P/C': '19.98', 'EPS next 5Y': '20.00%', 'ROE': '-336.00%', '52W Range': '0.04 - 5.93', 'Perf YTD': '-96.87%',
# 'Dividend': '-', 'P/FCF': '-', 'EPS past 5Y': '-', 'ROI': '-204.20%', '52W High': '-98.32%', 'Beta': '1.81', 'Dividend %': '-', 'Quick Ratio': '0.20',
# 'Sales past 5Y': '-40.50%', 'Gross Margin': '75.20%', '52W Low': '170.73%', 'ATR': '0.01', 'Employees': '69', 'Current Ratio': '0.30', 'Sales Q/Q': '66.70%',
# 'Oper. Margin': '-', 'RSI (14)': '75.34', 'Volatility': '45.48% 19.64%', 'Optionable': 'No', 'Debt/Eq': '0.88', 'EPS Q/Q': '91.10%', 'Profit Margin': '-',
# 'Rel Volume': '32.84', 'Prev Close': '0.08', 'Shortable': 'Yes', 'LT Debt/Eq': '0.00', 'Earnings': 'Jan 29 AMC', 'Payout': '-', 'Avg Volume': '8.22M',
# 'Price': '0.10', 'Recom': '-', 'SMA20': '118.86%', 'SMA50': '36.29%', 'SMA200': '-78.47%', 'Volume': '269,850,649', 'Change': '29.91%'}

# 'Debt/Eq' <3
# 'LT Debt/Eq' <3
# Gross Margin .05
if screener == True:
    fViz = FinVizScreener2(screenerLimit, metrics)
    print("-------------------------")
    print(fViz)

    data = scrape_finviz(fViz)
    hyperList = []
    print(data)

    if hyperList.count == 0: print("FAIL!");exit(0)
    for idx in range(0, len(data)):
        hyperList.append(data.iloc[idx]['Ticker'])
        if 'Volatility' in data:
            print(data.iloc[idx]['Ticker'], data.iloc[idx]['Volatility'])
        else:
            print(data.iloc[idx]['Ticker'])
    print(hyperList)

print("------------------------")
print(hyperList)

if getVolatiles == True:
    dictList = getVolatilesCSV2(-666, hyperList)
    print(dictList)
    # Remove
    limitElements = len(dictList)
else:
    limitElements = len(hyperList)
lname = ""
z = 0

print(hyperList)
print("limitElements" + str(len(hyperList)))
print("LimitElements={}".format(limitElements))
# ->
if getVolatiles == True:
    # Temporary Main
    for lname in hyperList:
        z = z + 1
        if z <= limitElements:
            print("Entry Point 1.")
            # print("Executing:"+lname+"="+str(value1)+"\n")
            #  try:
            print("******************************************")
            print("*            " + lname + "                   *")
            print("******************************************")
            if daysforward == 0:
                lstSelection = Main(lname, 1, getWinner, Epochs, skipLSTMModel, lstSelection, blnChangeUp, showGraphs,
                                    lookBack, daysforward, segments)
            else:
                """
                Loop for Days forward
                """
#START HERE 10/4/2020
                for iterdaysforward in range(daysforward):
                    lstSelection = Main(lname, 1, getWinner, Epochs, skipLSTMModel, lstSelection, blnChangeUp,
                                        showGraphs,
                                        lookBack, iterdaysforward, segments)
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            # except:
            #    print("Main core Error")
            # else:
            #    break
else:
        for lname in hyperList:
            z = z + 1
            if z <= limitElements:
                print("Entry Point 2.")
                # print("Executing:"+lname+"="+str(value1)+"\n")
                #  try:
                print("******************************************")
                print("*            " + lname + "                   *")
                print("******************************************")
                if daysforward == 0:
                    lstSelection = Main(lname, 1, getWinner, Epochs, skipLSTMModel, lstSelection, blnChangeUp,
                                        showGraphs,
                                        lookBack, daysforward, segments)
                else:
                    for daysforward in range(daysforward):
                        lstSelection = Main(lname, 1, getWinner, Epochs, skipLSTMModel, lstSelection, blnChangeUp,
                                            showGraphs,
                                            lookBack, daysforward, segments)
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

                # except:
                # print("Quandl Error\n")
            else:
                break
print("1.) -----------------------------------------")
print("Final Predictions are:{}".format(lstSelection))
#os.system("beep -f 555 -l 460")