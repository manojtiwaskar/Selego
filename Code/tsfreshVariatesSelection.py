from tsfresh import extract_relevant_features, select_features, extract_features
from tsfresh.utilities.dataframe_functions import impute
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

'''To use this tsfresh feature extraction based variate ranking script, user has 
to the absolute input data file path and the directory path to store the variate
 rankings. The code assumes that the target variate id the last column in the dataset.'''

if __name__ == '__main__':

    # specify the input data file path.
    input = "../add/input/file/path/xyz.csv"

    ranking_output = "../output/directory/path"

    df = pd.read_csv(input)
    column_names = list(df)

    new_df = pd.DataFrame()
    for i , columnName in enumerate(column_names):
        data = [[i, x] for x in df[columnName]]
        new_df = new_df.append(data, ignore_index=True)
    new_df.columns = ['id', 'TimeSeries']

    extracted_features = extract_features(new_df, column_id='id', n_jobs=10)
    impute(extracted_features)
    totalVariates = extracted_features.shape[0]

    # we do Minmax normalization of each feature (in the feature vector per variate) across the variates
    # to have values of each feature in the same range.
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(extracted_features)
    normalized_data = scaler.transform(extracted_features)

    normalized_data = pd.DataFrame(normalized_data)
    Y = normalized_data.iloc[-1:].values.tolist()

    EuclideanDistances = []
    for featureRow in range(totalVariates - 1):
        variate_name = column_names[featureRow]
        X = normalized_data.iloc[featureRow:featureRow+1,].values.tolist()
        distance = (sum([(a - b) ** 2 for a, b in zip(X[0], Y[0])])) ** 0.5
        EuclideanDistances.append(distance)

    EuclideanDistances = np.asarray(EuclideanDistances)
    ranking = np.argsort(EuclideanDistances)

    result = []
    for indx in ranking:
        result.append(column_names[indx])

    result_df = pd.DataFrame(result)
    output_path = ranking_output + "/Tsfresh_variate_ranking.csv"
    result_df.to_csv(ranking_output, index=None, header=None)

