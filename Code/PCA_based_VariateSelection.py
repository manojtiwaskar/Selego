import pandas as pd
from pathlib import Path
import numpy as np
from scipy.linalg import svd
import time


'''The code assumes that the target variate is the last column in the dataset
and no time or date column should be present.'''

base_path = Path(__file__).parent
input_file_path = str(base_path) + '/NasdaqDataset.csv'
variate_ranking_output = str(base_path) + "/Nasdaq_pca_ranking.csv"
ranked_variate_names = str(base_path) + "/Nasdaq_pca_variate_names.csv"

df1 = pd.read_csv(input_file_path)
numpy_df1 = df1.values

df_transposed = numpy_df1.transpose()
symmetric_mat = np.dot(df_transposed, numpy_df1)
U, S, V = svd(symmetric_mat)
U_transpose = U.transpose()
pca_trnsf = np.dot(U, U_transpose)

#variate ranking, higher the correlation better the ranking
pca_rank = np.argsort(pca_trnsf[:, -1])
np.savetxt(variate_ranking_output,pca_rank[::-1], delimiter=",")

column_names = list(df1.columns)
# column_names.remove('Time')
ranked_df = pd.read_csv(variate_ranking_output, header=None)
ranked_indx = ranked_df.iloc[1:, 0]
ranked_variates = []
for ind in ranked_indx:
    ranked_variates.append(column_names[int(ind)])
df = pd.DataFrame(ranked_variates)
df.to_csv(ranked_variate_names, header=None, index=None)