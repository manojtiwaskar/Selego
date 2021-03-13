# Selego: Robust Variate Selection for Accurate Time Series Forecasting

### Dataset

I have provided two publicly available dataset in the ` ./Datasets/ ` folder, those are:

	1. NASDAQ (S&P)
	2. EEG-BCI

### How to use Selego

There are two steps to be followed to get the ranked input variates with respect to target variate:

To compute the Selego based variate selection
	* Open the SelegoVariatesCorrelation folder in the Matlab available inside the Code folder.
	* Open Selego_StarterScript.m file in the editor and provide the necessary input mentioned in the **Note**.
	* Run the Selego_StarterScript.m script.

	
### Building forecasting model

To build and train the forecasting model using Selego based ranked variates, I considered three Neural Network based model i.e., LSTM 
(Long short-term Memory), RNN (Recurrent Neural Network), and CNN1d (1D Convolutional Neural Network).

The scripts to run the above mentioned models are available in ` ./Code/Models/ `

**Note: To successfully run the models, provide the necessary inputs mentioned in the commented script.**

1. To run LSTM model:
```
$ python LSTM_Model.py
```

2. To run RNN model:
```
$ python RNN_Model.py
```

3. To run CNN1d model:
```
$ python CNN1d_Model.py
```

### Additional variate selection techniques

The scripts to run tsFRESH and PCA based variate selection algorithm are also available in ` ./Code/ ` folder.

**Note: To successfully run the below variate selection algorithm, provide the necessary inputs mentioned in the commented script.**

1. To run tsFRESH based variate selection algorithm:
```
$ python tsfreshVariatesSelection.py
```

2. To run PCA based variate selection algorithm:
```
$ python PCA_VariateSelection.py
```
