# poodle
# Sung-Jin Kim, April 8, 2016

from sklearn import linear_model 
import pandas as pd

def read_csv( *args, index_col=0, header=[0,1], **kwargs):
	"""
	Emulation for pandas.DataFrame() 
	Parameters
	----------
	The parameters of Pandas DataFrame are used 
	*args : any type
		 all arguments without a keyword
	**kwargs: any type
		 all arguments without a keyword
	"""
	return pd.read_csv( *args, index_col=index_col, header=header, **kwargs)

class LinearRegression( linear_model.LinearRegression):
	def __init__( self, **kwargs):
		super().__init__(**kwargs)

	def fit( self, xy_file):
		df = read_csv( xy_file)
		X = df['X'].values
		y = df['y'].values
		return super().fit( X, y)
