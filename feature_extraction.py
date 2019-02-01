import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
pd.options.display.precision = 15

from scipy.stats import kurtosis
from scipy.stats import skew


N = 629_100_000 # input data; total = 629_100_000
rows = 150_000 # length of a test segment. for training, we will split the continuous data stream into segments of the same length
n_offsets = 10 # instead of just considering disjoint segments, allow them to overlap over a fraction of (n_offsets - 1)/n of their length



def extract_features(x,position):
	'''
	Inputs:
	x, a numpy array of shape (rows, 1) containing the acoustic signal for the current segment
	position, the index of the current segment

	Outputs:
	features, a pandas DataFrame of shape (1, n_features), to be appended to the training data

	Currently, n_features = 216
	'''

	features = pd.DataFrame(index = [position], dtype = np.float64) # initialize the dataframe


	# Basic statistics for the entire segment
	features.loc[position, 'mean'] = x.mean()
	features.loc[position, 'std'] = x.std()
	features.loc[position, 'max'] = x.max()
	features.loc[position, 'min'] = x.min()
	features.loc[position, 'skew'] = skew(x)
	features.loc[position, 'kurt'] = kurtosis(x)
	
	# Statistics for only the first 1/3, last 1/3, first 1/15 or last 1/15 of the segment
	features.loc[position, 'std_first_50000'] = x[:50000].std()
	features.loc[position, 'std_last_50000'] = x[-50000:].std()
	features.loc[position, 'std_first_10000'] = x[:10000].std()
	features.loc[position, 'std_last_10000'] = x[-10000:].std()
	
	features.loc[position, 'avg_first_50000'] = x[:50000].mean()
	features.loc[position, 'avg_last_50000'] = x[-50000:].mean()
	features.loc[position, 'avg_first_10000'] = x[:10000].mean()
	features.loc[position, 'avg_last_10000'] = x[-10000:].mean()
	
	features.loc[position, 'min_first_50000'] = x[:50000].min()
	features.loc[position, 'min_last_50000'] = x[-50000:].min()
	features.loc[position, 'min_first_10000'] = x[:10000].min()
	features.loc[position, 'min_last_10000'] = x[-10000:].min()
	
	features.loc[position, 'max_first_50000'] = x[:50000].max()
	features.loc[position, 'max_last_50000'] = x[-50000:].max()
	features.loc[position, 'max_first_10000'] = x[:10000].max()
	features.loc[position, 'max_last_10000'] = x[-10000:].max()

	# Quantile information
	features.loc[position, "q01"] = np.quantile(x, 0.01)
	features.loc[position, "q05"] = np.quantile(x, 0.05)
	features.loc[position, "q25"] = np.quantile(x, 0.25)
	features.loc[position, "q50"] = np.quantile(x, 0.50)
	features.loc[position, "q75"] = np.quantile(x, 0.75)
	features.loc[position, "q95"] = np.quantile(x, 0.95)
	features.loc[position, "q99"] = np.quantile(x, 0.99)

	# Average change over the segment
	features.loc[position, 'av_change_abs'] = np.mean(np.diff(x))
	features.loc[position, 'av_change_rate'] = np.mean(np.nonzero((np.diff(x) / x[:-1]))[0])

	# Repeat everything, with x replaced by its absolute values
	abs_x = np.abs(x)
	features.loc[position, 'abs_mean'] = abs_x.mean()
	features.loc[position, 'abs_std'] = abs_x.std()
	features.loc[position, 'abs_max'] = abs_x.max()
	features.loc[position, 'abs_min'] = abs_x.min()
	features.loc[position, 'abs_skew'] = skew(abs_x)
	features.loc[position, 'abs_kurt'] = kurtosis(abs_x)
	
	features.loc[position, 'abs_std_first_50000'] = abs_x[:50000].std()
	features.loc[position, 'abs_std_last_50000'] = abs_x[-50000:].std()
	features.loc[position, 'abs_std_first_10000'] = abs_x[:10000].std()
	features.loc[position, 'abs_std_last_10000'] = abs_x[-10000:].std()
	
	features.loc[position, 'abs_avg_first_50000'] = abs_x[:50000].mean()
	features.loc[position, 'abs_avg_last_50000'] = abs_x[-50000:].mean()
	features.loc[position, 'abs_avg_first_10000'] = abs_x[:10000].mean()
	features.loc[position, 'abs_avg_last_10000'] = abs_x[-10000:].mean()
	
	features.loc[position, 'abs_min_first_50000'] = abs_x[:50000].min()
	features.loc[position, 'abs_min_last_50000'] = abs_x[-50000:].min()
	features.loc[position, 'abs_min_first_10000'] = abs_x[:10000].min()
	features.loc[position, 'abs_min_last_10000'] = abs_x[-10000:].min()
	
	features.loc[position, 'abs_max_first_50000'] = abs_x[:50000].max()
	features.loc[position, 'abs_max_last_50000'] = abs_x[-50000:].max()
	features.loc[position, 'abs_max_first_10000'] = abs_x[:10000].max()
	features.loc[position, 'abs_max_last_10000'] = abs_x[-10000:].max()


	features.loc[position, "abs_q01"] = np.quantile(abs_x, 0.01)
	features.loc[position, "abs_q05"] = np.quantile(abs_x, 0.05)
	features.loc[position, "abs_q25"] = np.quantile(abs_x, 0.25)
	features.loc[position, "abs_q50"] = np.quantile(abs_x, 0.50)
	features.loc[position, "abs_q75"] = np.quantile(abs_x, 0.75)
	features.loc[position, "abs_q95"] = np.quantile(abs_x, 0.95)
	features.loc[position, "abs_q99"] = np.quantile(abs_x, 0.99)

	

	# Repeat everything, but with information averaged over windows of various sizes
	window_sizes = [10, 100, 1000]
	for window in window_sizes:
		rolling_mean = x.rolling(window).mean().dropna()
		features.loc[position, "mean_mean_" + str(window)] = rolling_mean.mean()
		features.loc[position, "std_mean_" + str(window)] = rolling_mean.std()
		features.loc[position, "min_mean_" + str(window)] = rolling_mean.min()
		features.loc[position, "max_mean_" + str(window)] = rolling_mean.max()
		features.loc[position, "skew_mean_" + str(window)] = skew(rolling_mean)
		features.loc[position, "kurt_mean_" + str(window)] = kurtosis(rolling_mean)
		features.loc[position, "q01_mean_" + str(window)] = np.quantile(rolling_mean, 0.01)
		features.loc[position, "q05_mean_" + str(window)] = np.quantile(rolling_mean, 0.05)
		features.loc[position, "q25_mean_" + str(window)] = np.quantile(rolling_mean, 0.25)
		features.loc[position, "q50_mean_" + str(window)] = np.quantile(rolling_mean, 0.50)
		features.loc[position, "q75_mean_" + str(window)] = np.quantile(rolling_mean, 0.75)
		features.loc[position, "q95_mean_" + str(window)] = np.quantile(rolling_mean, 0.95) 
		features.loc[position, "q99_mean_" + str(window)] = np.quantile(rolling_mean, 0.99)

		abs_rolling_mean = np.abs(rolling_mean)
		features.loc[position, "abs_mean_mean_" + str(window)] = abs_rolling_mean.mean()
		features.loc[position, "abs_std_mean_" + str(window)] = abs_rolling_mean.std()
		features.loc[position, "abs_min_mean_" + str(window)] = abs_rolling_mean.min()
		features.loc[position, "abs_max_mean_" + str(window)] = abs_rolling_mean.max()
		features.loc[position, "abs_skew_mean_" + str(window)] = skew(abs_rolling_mean)
		features.loc[position, "abs_kurt_mean_" + str(window)] = kurtosis(abs_rolling_mean)
		features.loc[position, "abs_q01_mean_" + str(window)] = np.quantile(abs_rolling_mean, 0.01)
		features.loc[position, "abs_q05_mean_" + str(window)] = np.quantile(abs_rolling_mean, 0.05)
		features.loc[position, "abs_q25_mean_" + str(window)] = np.quantile(abs_rolling_mean, 0.25)
		features.loc[position, "abs_q50_mean_" + str(window)] = np.quantile(abs_rolling_mean, 0.50)
		features.loc[position, "abs_q75_mean_" + str(window)] = np.quantile(abs_rolling_mean, 0.75)
		features.loc[position, "abs_q95_mean_" + str(window)] = np.quantile(abs_rolling_mean, 0.95) 
		features.loc[position, "abs_q99_mean_" + str(window)] = np.quantile(abs_rolling_mean, 0.99)	

		rolling_std = x.rolling(window).std().dropna()
		features.loc[position, "mean_std_" + str(window)] = rolling_std.mean()
		features.loc[position, "std_std_" + str(window)] = rolling_std.std()
		features.loc[position, "min_std_" + str(window)] = rolling_std.min()
		features.loc[position, "max_std_" + str(window)] = rolling_std.max()
		features.loc[position, "skew_std_" + str(window)] = skew(rolling_std)
		features.loc[position, "kurt_std_" + str(window)] = kurtosis(rolling_std)
		features.loc[position, "q01_std_" + str(window)] = np.quantile(rolling_std, 0.01)
		features.loc[position, "q05_std_" + str(window)] = np.quantile(rolling_std, 0.05)
		features.loc[position, "q25_std_" + str(window)] = np.quantile(rolling_std, 0.25)
		features.loc[position, "q50_std_" + str(window)] = np.quantile(rolling_std, 0.50)
		features.loc[position, "q75_std_" + str(window)] = np.quantile(rolling_std, 0.75)
		features.loc[position, "q95_std_" + str(window)] = np.quantile(rolling_std, 0.95) 
		features.loc[position, "q99_std_" + str(window)] = np.quantile(rolling_std, 0.99)	

		abs_rolling_std = np.abs(rolling_std)
		features.loc[position, "abs_mean_std_" + str(window)] = abs_rolling_std.mean()
		features.loc[position, "abs_std_std_" + str(window)] = abs_rolling_std.std()
		features.loc[position, "abs_min_std_" + str(window)] = abs_rolling_std.min()
		features.loc[position, "abs_max_std_" + str(window)] = abs_rolling_std.max()
		features.loc[position, "abs_skew_std_" + str(window)] = skew(abs_rolling_std)
		features.loc[position, "abs_kurt_std_" + str(window)] = kurtosis(abs_rolling_std)
		features.loc[position, "abs_q01_std_" + str(window)] = np.quantile(abs_rolling_std, 0.01)
		features.loc[position, "abs_q05_std_" + str(window)] = np.quantile(abs_rolling_std, 0.05)
		features.loc[position, "abs_q25_std_" + str(window)] = np.quantile(abs_rolling_std, 0.25)
		features.loc[position, "abs_q50_std_" + str(window)] = np.quantile(abs_rolling_std, 0.50)
		features.loc[position, "abs_q75_std_" + str(window)] = np.quantile(abs_rolling_std, 0.75)
		features.loc[position, "abs_q95_std_" + str(window)] = np.quantile(abs_rolling_std, 0.95) 
		features.loc[position, "abs_q99_std_" + str(window)] = np.quantile(abs_rolling_std, 0.99)	
	return features


if __name__ == '__main__':
	'''
	Specify datatype to speed up read_csv
	Only reads the first N rows
	'''
	train = pd.read_csv('train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32}, nrows = N)
	print("Done reading!")

	segments = int(np.floor(train.shape[0] / rows)) # number of disjoint segments in the data stream

	'''
	Initialize dataframes for training segments and fill them with extracted features. After completion:
	X_tr.shape = (segments * n_offsets, n_features)
	y_tr.shape = (segments * n_offsets, 1)
	Currently segments = 4194, n_offsets = 10.
	'''

	X_tr = pd.DataFrame(index = [], dtype=np.float64) 
	y_tr = pd.DataFrame(index=range(segments * n_offsets), dtype=np.float64,
	                       columns=['time_to_failure'])

	for offset in range(n_offsets): 
		for segment in tqdm(range(segments) ):
			jump = int(offset * rows / n_offsets) # offset the start of the segment
			position = segment + offset * segments # the index of the current training segment

			seg = train.iloc[segment*rows + jump:segment*rows+ jump +rows] # data for the current training segment
			x = seg['acoustic_data']
			y_tr.loc[position, 'time_to_failure'] = seg['time_to_failure'].values[-1]

			features = extract_features(x,position) # call a method to extract features for the current training segment
			X_tr = X_tr.append(features) # add a data point corresponding to the current training segment

	scaler = StandardScaler()
	scaler.fit(X_tr) # scale the training data, for faster learning

	X_train_scaled = pd.DataFrame(scaler.transform(X_tr), columns=X_tr.columns)
	X_train_scaled.index.name = "seg_id"
	X_train_scaled.to_csv("X_train.csv") # write the training data to file
	y_tr.index.name = "seg_id"
	y_tr.to_csv("Y_train.csv") # write the training labels to file



	submission = pd.read_csv('sample_submission.csv', index_col='seg_id') # read the sample submission, to find test segment indices

	'''
	Initialize dataframe for test segments, and fill it with extracted features. After completion:
	X_test.shape = (2624, n_features)

	'''
	X_test = pd.DataFrame(index = [], columns=X_tr.columns, dtype=np.float64) 

	for i, seg_id in enumerate(tqdm(submission.index)):
		seg = pd.read_csv('test/' + seg_id + '.csv') # read data stream for current test segment
		x = seg['acoustic_data'] # values for acoustic data

		features = extract_features(x,seg_id) # call a method to extract features for the current test segment
		X_test = X_test.append(features) # add a data point corresponding to the current test segment


	X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns) # scale the test data
	X_test_scaled.index.name = "seg_id"
	X_test_scaled.to_csv("X_test.csv") # write the test data to file