import numpy as np
import matplotlib.pyplot as plt 

class MyLinearRegession():

	def __init__(self, fit_intercept = True):
		self.coef = None
		self.intercept = None
		self.fit_intercept = fit_intercept

	def __repr__(self):
		return 'This is my linear regression model.'

	def fit(self, X, y):

		if len(X.shape)==1:
			X = X.reshape(-1,1)

		if self.fit_intercept:
			X_biased = np.c_[np.ones(X.shape[0]), X]
		else:
			X_biased = X

		if len(y.shape)==1:
			y = y.reshape(-1,1)

		#normal equation
		xTx = np.dot(X_biased.T, X_biased)
		xTx_inv = np.linalg.inv(xTx)
		xTy = np.dot(X_biased.T, y)
		coef = np.dot(xTx_inv, xTy)

		if self.fit_intercept:
			self.intercept=coef[0]
			self.coef = coef[1]
		else:
			self.coef = coef

	def predict(self, new_X):

		if len(new_X.shape) == 1:
			new_X = new_X.reshape(-1, 1)

		predictions = self.intercept + np.dot(new_X, self.coef)

		return predictions

	def rmse(self, y, y_preds):

		if not(type(y) == np.ndarray):
			y = np.array(y)

		if not(type(y_preds) == np.ndarray):
			y_preds = np.array(y_preds)

		rmse = np.sqrt((sum(y - y_preds) ** 2) / len(y))

		return rmse

def plot_fitted(y_true, y_fitted, reference_line = False):

	plt.figure(figsize=(10, 10))
	plt.title('True vs Fitted', fontsize = 14)
	plt.scatter(y_true, y_fitted, s=50, c='b', alpha = 0.5)
	if reference_line:
		plt.plot(y_true, y_fitted, c='k', linestyle=':')

	plt.xlabel('True Values')
	plt.ylabel('Fitted Values')
	plt.grid(True)
	plt.show()

def plot_erros(y_true, y_fitted, reference_line=False):

	plt.figure(figsize=(10, 10))
	plt.title('Error Plot')
	errors = np.round(y_true - y_fitted, 3)
	plt.scatter(range(len(y_true)), errors, s=20)
	if reference_line:
		#plt.plot(range(len(y_true)), errors, linestyle=':', c='k')
		plt.plot(range(len(y_true)), np.zeros(len(y_true)), linestyle = '--')

	plt.xlabel('Index of X Values')
	plt.ylabel('Errros')
	plt.grid(True)
	plt.show()


def main(visualize = False, erros_plot = False):
	mylm = MyLinearRegession()
	X = np.random.normal(np.random.random(1)*100, 2.3, 5000)
	y = np.round(1.5 + (X*20))

	def manual_calc(x):
		return np.round(1.5+np.sqrt(x*20), 3)

	ind = np.arange(len(X))
	np.random.shuffle(ind)
	train_ind, test_ind = ind[:4000], ind[4000:]

	X_train = X[train_ind]
	y_train = y[train_ind]
	X_test = X[test_ind]
	y_test = y[test_ind]

	mylm.fit(X_train, y_train)

	preds = np.round(mylm.predict(X_test), 3)
	rmse = mylm.rmse(y_test, preds)

	output = (
		f'intercept is: {mylm.intercept}\n'
		f'coefficients are: {mylm.coef}\n')
	print(output)

	test = (
		f'Predictios: {preds[1:10]}\n'
		f'True y: {y_test[1:10]}\n'
		f'RMSE: {rmse}\n'
		)
	print(test)

	if visualize:
		plot_fitted(y_test, preds, True)

	if erros_plot:
		plot_erros(y_test, preds, True)


if __name__ == '__main__':
	main(False, True)

