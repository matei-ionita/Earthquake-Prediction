import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

n_features = 216 # total 216
n_rows = 4194 # total 41940

def baseline_model():
	# Construct model
	model = Sequential()
	model.add(Dense( int(0.6 * n_features), input_dim =  n_features, kernel_initializer = "normal", activation = "relu", kernel_regularizer=regularizers.l2(0.009) ))
	model.add(Dense(1, kernel_initializer = "normal", kernel_regularizer=regularizers.l2(0.009))) # regression problem, no activation in output layer

	# Compile model
	adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
	model.compile(loss = "mean_absolute_error", optimizer = adam)
	return model

if __name__ == "__main__":
	X_train = pd.read_csv("X_train.csv", index_col = "seg_id", nrows = n_rows)
	Y_train = pd.read_csv("Y_train.csv", index_col = "seg_id", nrows = n_rows)
	X_test  = pd.read_csv( "X_test.csv", index_col = "seg_id")

	seed = 42

	# Cross-validation
	'''
	np.random.seed(seed)
	estimators = []
	estimators.append(('standardize', StandardScaler()))
	estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=500, batch_size=128, verbose=1)))
	pipeline = Pipeline(estimators)
	kfold = KFold(n_splits=5, random_state=seed)
	score = cross_val_score(pipeline, X_train.values, Y_train.values, cv=kfold, verbose = 1)
	print(score)
	print(np.mean(score))
	'''

	# train-test split
	'''
	# X_tr, X_val, Y_tr, Y_val = train_test_split(X_train,Y_train, test_size = 0.2, random_state = seed)
	# score = model.evaluate(X_val,Y_val, batch_size = 128)
	# print(score)
	'''

	model = baseline_model()
	model.fit(X_tr,Y_tr, epochs = 500, batch_size = 128, verbose = 1) # train model
	results = model.predict(X_test) # predict with model

	submission = pd.read_csv('sample_submission.csv', index_col='seg_id') # read sample submission, to get segment indices
	submission['time_to_failure'] = results
	submission.to_csv('submission.csv')