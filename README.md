# Selego: Robust Variate Selection for Accurate Time Series Forecasting

### Dataset

I have provided to publicly available dataset in the ` ./Datasets/ ` folder, those are:

	1. NASDAQ (S&P)
	2. EEG-BCI

### How to use selego

There are two steps to be followed to get the ranked input variates with respect to target variate:

1. First, compute Variate-Variate temporal correlation matrix.
	* Open the SelegoVariatesCorrelation folder in the Matlab available inside the Code folder.
	* Open Selego_StarterScript.m file in the editor and provide the necessary input mentioned in the **Note**.
	* Run the Selego_StarterScript.m script.

2. Second, to get a list of ranked variate names.
	* Open SelegoVariateRanking.py file avaiable in the Code folder and provide the necessary input to run the script.
	* Run the SelegoVariateRanking.py
	
### Building forecasting model

To build and train the forecasting model using Selego based ranked variates, I considered three Neural Network based model i.e., LSTM 
(Long short-term Memory), RNN (Recurrent Neural Network), and CNN1d (1D Convolutional Neural Network).

The scripts to run the above mentioned models are available at ` ./Code/Models/ `

**Note: To run the Models successfully please provide the necessary inputs mentioned in the commented script.**

1. To LSTM model:
```
$ python LSTM_Model.py
```

2. To RNN model:
```
$ python RNN_Model.py
```

3. To CNN1d model:
```
$ python CNN1d_Model.py
```