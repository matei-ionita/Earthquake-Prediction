import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR, SVR
from sklearn.metrics import mean_absolute_error
pd.options.display.precision = 15

import lightgbm as lgb
import xgboost as xgb
import time
import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge, RidgeCV
import gc
from catboost import CatBoostRegressor
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

N = 4194

def train_model(X, X_test, y, params, folds, model_type='lgb', plot_feature_importance=False, model=None):

	oof = np.zeros(len(X)) # validation predictions
	prediction = np.zeros(len(X_test)) # testing predictions
	scores = []
	feature_importance = pd.DataFrame() # store feature importance, returned by LightGBM

	for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
		# split training and validation data
		print('Fold', fold_n, 'started at', time.ctime())
		X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
		y_train, y_valid = y.iloc[train_index], y.iloc[valid_index] 

		if model_type == 'lgb':
			model = lgb.LGBMRegressor(**params, n_estimators = 20000, nthread = 4, n_jobs = -1) # set up LightGBM
			model.fit(X_train, y_train, 
			        eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='mae',
			        verbose=10000, early_stopping_rounds=200)
			
			y_pred_valid = model.predict(X_valid)
			y_pred = model.predict(X_test, num_iteration=model.best_iteration_) # _?
		'''
		if model_type == 'xgb':
			train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X_tr.columns)
			valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X_tr.columns)

			watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
			model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=500, params=params)
			y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X_tr.columns), ntree_limit=model.best_ntree_limit)
			y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X_tr.columns), ntree_limit=model.best_ntree_limit)

		if model_type == 'rcv':
			model = RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0), scoring='neg_mean_absolute_error', cv=3)
			model.fit(X_train, y_train)
			print(model.alpha_)

			y_pred_valid = model.predict(X_valid).reshape(-1,)
			score = mean_absolute_error(y_valid, y_pred_valid)
			print(f'Fold {fold_n}. MAE: {score:.4f}.')
			print('')
			
			y_pred = model.predict(X_test).reshape(-1,)
		
		if model_type == 'sklearn':
			model = model
			model.fit(X_train, y_train)

			y_pred_valid = model.predict(X_valid).reshape(-1,)
			score = mean_absolute_error(y_valid, y_pred_valid)
			print(f'Fold {fold_n}. MAE: {score:.4f}.')
			print('')

			y_pred = model.predict(X_test).reshape(-1,)
        
		if model_type == 'cat':
			model = CatBoostRegressor(iterations=20000,  eval_metric='MAE', **params)
			model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)

			y_pred_valid = model.predict(X_valid)
			y_pred = model.predict(X_test)
		'''
		
		oof[valid_index] = y_pred_valid.reshape(-1,)
		scores.append(mean_absolute_error(y_valid, y_pred_valid))

		prediction += y_pred    

		if model_type == 'lgb':
			# feature importance
			fold_importance = pd.DataFrame()
			fold_importance["feature"] = X.columns
			fold_importance["importance"] = model.feature_importances_
			fold_importance["fold"] = fold_n + 1
			feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

	prediction /= n_fold

	print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

	if model_type == 'lgb':
		feature_importance["importance"] /= n_fold
		if plot_feature_importance:
			cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
			    by="importance", ascending=False)[:50].index

			best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

			plt.figure(figsize=(16, 12));
			sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
			plt.title('LGB Features (avg over folds)');

			return oof, prediction, feature_importance
		return oof, prediction

	else:
		return oof, prediction



if __name__ == "__main__":
	X_train = pd.read_csv("X_train.csv", index_col = "seg_id", nrows = N)
	Y_train = pd.read_csv("Y_train.csv", index_col = "seg_id", nrows = N)
	X_test  = pd.read_csv( "X_test.csv", index_col = "seg_id")


	
	n_fold = 5
	folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)


	params = {'num_leaves': 54, # max leaves in the decision tree used at each step. assuming balanced trees, this means decisions are made based on log_2(num_leaves) features, original = 54
		 'min_data_in_leaf': 79, # don't do splits in the tree, if they result in leaves with fewer than min_data_in_leaf data points, original = 79
		 'objective': 'huber',
		 'max_depth': -1,
		 'learning_rate': 0.01, # shrinkage, regularization parameter. F_m(x) = F_{m-1}(x) + learning_rate * h_m(x)
		 "boosting": "gbdt", # original: gbdt
		 # "feature_fraction": 0.8354507676881442,
		 "bagging_freq": 3,
		 "bagging_fraction": 0.8126672064208567, # fraction of training data used to train each base learner
		 "bagging_seed": 11,
		 "metric": 'mae', # mean absolute error, same as the competition metric
		 "verbosity": -1,
		 'reg_alpha': 1.1302650970728192, # L1 regularization original = 1.1302650970728192
		 'reg_lambda': 0.4603427518866501 # L2 regulatization, original = 0.3603427518866501
		 }

	oof_lgb, prediction_lgb, feature_importance = train_model(X = X_train, X_test=X_test, y=Y_train, params=params, folds = folds, model_type='lgb', plot_feature_importance=True)


	plt.figure(figsize=(16, 8))
	plt.plot(Y_train, color='g', label='y_train')
	plt.plot(oof_lgb, color='b', label='lgb')
	# plt.plot(oof_xgb, color='teal', label='xgb')
	# plt.plot(oof_svr, color='red', label='svr')
	# plt.plot((oof_lgb + oof_xgb + oof_svr) / 3, color='gold', label='blend')
	plt.legend();
	plt.title('Predictions vs actual');
	plt.show()

	submission = pd.read_csv('sample_submission.csv', index_col='seg_id')
	submission['time_to_failure'] = prediction_lgb
	# submission['time_to_failure'] = (prediction_lgb + prediction_xgb + prediction_svr) / 3
	# print(submission.head())
	submission.to_csv('submission.csv')