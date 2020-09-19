import numpy as np
import pandas as pd

'''Before running this script we assume that you already have selego variate-variate correlation matrix
(if not then run the Selego Matlab code available in Selego_FeatureExtraction folder)'''

inputDatasetFilePath = '../dataset/file/path/xyz.csv'
selegoCorrMatfilePath = "../selego/correlationMat/file/path/abc.csv"
variateRankingResultDirectory = "../directory/to/store/variate/ranking"

dataset_df = pd.read_csv(inputDatasetFilePath)
selego_corr_mat = pd.read_csv(selegoCorrMatfilePath, header=None)
selego = selego_corr_mat.to_numpy()
selego_rank = np.argsort(selego[:, -1])
res = selego_rank[::-1]

column_names = list(dataset_df.columns)
ranked_indx = res.tolist()
ranked_variate_names = []
for ind in ranked_indx:
    ranked_variate_names.append(column_names[int(ind)])
    
df = pd.DataFrame(ranked_variate_names)
output = variateRankingResultDirectory + '/SelegoVariateRanking.csv'
df.to_csv(output, header=None, index=None)