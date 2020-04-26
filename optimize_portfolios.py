import os
import multiprocessing as mp
import pandas as pd
from pandas.tseries.offsets import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json 
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from statsmodels.tsa.arima_model import ARIMA
from arch import arch_model
import cvxpy as cp
from utils import *
from pull_data import *
from create_feature_df import *
from test_signals import *

colors = ['darkgrey','indianred','cornflowerblue','mediumseagreen','darkorange','goldenrod','mediumpurple']

def get_parametricopt_weights(df_train,model,halflife=60,loss='return_loss',reg=2,penalty=0,gamma=0):
	dates = pd.date_range(start=df_train.date.min(),end=df_train.date.max(),freq='MS')
	x = pd.DataFrame({'date':dates,'idx':range(len(dates))})
	df_train = df_train.merge(x,how='left',on='date')
	mkt = [0]*len(df_train) #could be market weights
	N = df_train.groupby('date')['permno'].transform(lambda x: x.count())
	if loss=='ic_tstat_loss':
		df_train['r'] = df_train.groupby('date')['retadj'].rank()
		r = df_train['r']
	elif loss=='return_loss':
		r = df_train['retadj']
	else:
		raise ValueError('unknown loss '+loss)
	F = df_train[model].fillna(0)
	d = df_train.idx
	alpha = 1-np.exp(np.log(.5)/halflife)
	weights = [(1-alpha)**(len(dates)-i-1) for i in range(len(dates))]
	x0 = np.random.randn(len(model))
	if loss=='ic_tstat_loss':
		opt = minimize(ic_tstat_loss,x0,args=(mkt,N,F,r,d,reg,penalty,weights),constraints=[{'type':'eq', 'fun': con}])
	elif loss=='return_loss':
		opt = opt = minimize(return_loss,x0,args=(mkt,N,F,r,d,reg,penalty,weights,gamma),constraints=[{'type':'eq', 'fun': con}])
	return opt.x

def get_markowitz_weights(df_train,model,mean_model='const',cov_model='expw',halflife=60):
	if mean_model=='const':
		means = df_train.mean()
	elif mean_model=='ar1':
		means = ar1(df_train)
	else:
		raise ValueError('unknown mean_model '+cov_model)
	if cov_model=='expw':
		cov = df_train.ewm(halflife=halflife).cov().values[-len(model):]
	elif cov_model=='ogarch':
		cov = ogarch(df_train,len(model),1)
	else:
		raise ValueError('unknown cov_model '+cov_model)
	return np.matmul(np.linalg.inv(cov),means)

def get_riskparity_weights(df_train,model,cov_model='expw',halflife=60,n_components=0,horizon=1,risk_targets=[]):
	x0 = [1/len(model)]*len(model)
	b = [(0,1)]*len(model)
	if cov_model=='expw':
		cov = df_train.ewm(halflife=halflife).cov().values[-len(model):]
	elif cov_model=='ogarch':
		if n_components==0:
			n_components=len(model)
		cov = ogarch(df_train,n_components,horizon)
	else:
		raise ValueError('unknown cov_model '+cov_model)
	if len(risk_targets)==0:
		risk_targets = [1/len(model)]*len(model)
	elif len(risk_targets)!=len(model):
		raise ValueError('risk targets must be same size as model')
	opt = minimize(sse,x0,args=(cov,risk_targets),constraints=[{'type':'eq', 'fun': con2}],bounds=b)
	return opt.x

def return_loss(x,mkt,N,F,r,d,reg,penalty,weights,gamma):
	w = mkt + F.dot(x)
	rets = w*r
	ts = np.bincount(d,rets)
	mean_return = np.average(ts,weights=weights)
	std_return = np.sqrt(np.average((ts-mean_return)**2,weights=weights))
	if gamma==0:
		objective = mean_return/std_return
	else:
		objective = mean_return-gamma*std_return
	return -objective+penalty*np.sum(np.abs(x)**reg)

def ic_tstat_loss(x,mkt,N,F,r,d,reg,penalty,weights):
	w = mkt + F.dot(x)
	df = pd.DataFrame({'w':w,'r':r,'d':d})
	ts = df.groupby('d')[['w','r']].corr()['w'].unstack()['r']
	num = np.average(ts,weights=weights)
	den = np.sqrt(np.average((ts-num)**2,weights=weights))
	return -num/den+penalty*np.sum(np.abs(x)**reg)

def con(x):
	return np.sum(np.power(x,2))-1

def sse(w,cov,rt):
	rw=np.matmul(w,cov)*w/np.matmul(np.matmul(w,cov),w)
	return np.sum((rt-rw)**2)

def con2(w):
	return np.sum(w)-1

def ogarch(data,n_components,horizon):
	means = data.mean()
	stds = data.std()
	data_norm = (data-means)/stds
	mod = PCA(n_components=n_components)
	mod.fit(data_norm)
	pcs = mod.components_
	a = stds.values*pcs
	pcs_T = np.matmul(pcs,data_norm.values.T)
	cov_pc = np.zeros((n_components,n_components))
	for i in range(n_components):
		am = arch_model(pcs_T[i],p=1,q=1)
		res = am.fit(disp='off')
		forecasts = res.forecast(horizon=horizon)
		cov_pc[i,i] = forecasts.variance['h.'+str(horizon)].values[-1]
	return np.matmul(np.matmul(a.T,cov_pc),a)

def ar1(data):
	out = []
	for col in data.columns:	
		model = ARIMA(data[col], order=(1,0,0),freq='MS')
		model_fit = model.fit(disp=0)
		forecast = model_fit.forecast()
		out.append(forecast[0][0])
	return out


class SecurityRanker(Backtester):
	def __init__(self,model_name,ranking_name=''):
		self.model_name = model_name
		if ranking_name!='':
			print('importing previously saved ranking '+ranking_name+' on model '+model_name)
			try:
				self.df_weight_all,self.df_sig,self.df_ret,self.ranking_specs = import_rankings(model_name,ranking_name)
				self.model = self.df_ret.columns[:(len(self.df_ret.columns)-len(self.ranking_specs))]
			except:
				raise ValueError('make sure '+ranking_name+' is output')
		else:
			Backtester.__init__(self,model_name=model_name)
			self.df_weight_all = pd.DataFrame(index=self.df_ret.index)
			self.ranking_specs = {}		
			self.model = self.df_ret.columns
		self.end_date = self.df_ret.index[-1]
		self.df_weight = pd.DataFrame(index=self.df_ret.index,columns=self.model)

	def step_rank(self,method,reestimate_freq,**kwargs):
		print('getting weights for '+self.date.strftime('%Y-%m'))
		if method == 'riskparity':
			df_train = self.df_ret.loc[self.df_ret.index<self.date][self.model]
			params = get_riskparity_weights(df_train,self.model,**kwargs)
		elif method == 'parametricopt':
			df_train = self.df_sig.reset_index()
			df_train = df_train.loc[df_train['date']<self.date]
			params = get_parametricopt_weights(df_train,self.model,**kwargs)
		elif method == 'markowitz':
			df_train = self.df_ret.loc[self.df_ret.index<self.date][self.model]
			params = get_markowitz_weights(df_train,self.model,**kwargs)
		else:
			raise ValueError('unknown method '+method)
		self.df_weight.loc[self.date] = params
		if reestimate_freq==0:
			self.date = self.end_date+DateOffset(months=1)
		else:
			self.date += DateOffset(months=reestimate_freq)

	def rank_rolling(self,name,method,start_date,reestimate_freq,**kwargs):
		self.date = pd.to_datetime(start_date)
		while self.date<=self.end_date:
			self.step_rank(method,reestimate_freq,**kwargs)
		print('done getting weights; normalizing df_weight and appending to df_sig and df_ret')
		self.df_weight = self.df_weight.fillna(method='ffill')
		reshaped = self.df_weight.loc[self.df_sig.index.get_level_values(1)]
		self.df_sig[name] = np.sum(reshaped.values*self.df_sig[self.model].values,1)
		self.df_sig[name] = self.df_sig.groupby('date')[name].apply(normalize_signal_ranks)
		self.df_sig['rw'] = self.df_sig[name]*self.df_sig['retadj']
		self.df_ret[name] = self.df_sig.groupby('date')['rw'].sum()
		self.df_sig = self.df_sig.drop(columns=['rw'])
		self.df_weight.columns = [name+' '+x for x in self.df_weight.columns]
		self.df_weight_all = pd.concat([self.df_weight_all,self.df_weight],1)
		self.df_weight = pd.DataFrame(index=self.df_ret.index,columns=self.model)
		self.ranking_specs[name] = {'method':method,'start_date':start_date,'reestimate_freq':reestimate_freq,**kwargs}

	def rank_constant(self,name,w=[]):
		if len(w)==0:
			w = [1/len(self.model)]*len(self.model)
		self.df_weight.loc[self.df_weight.index[0]] = w
		self.df_weight = self.df_weight.fillna(method='ffill')
		reshaped = self.df_weight.loc[self.df_sig.index.get_level_values(1)]
		self.df_sig[name] = np.sum(reshaped.values*self.df_sig[self.model].values,1)
		self.df_sig[name] = self.df_sig.groupby('date')[name].apply(normalize_signal_ranks)
		self.df_sig['rw'] = self.df_sig[name]*self.df_sig['retadj']
		self.df_ret[name] = self.df_sig.groupby('date')['rw'].sum()
		self.df_sig = self.df_sig.drop(columns=['rw'])
		self.df_weight.columns = [name+' '+x for x in self.df_weight.columns]
		self.df_weight_all = pd.concat([self.df_weight_all,self.df_weight],1)
		self.df_weight = pd.DataFrame(index=self.df_ret.index,columns=self.model)
		self.ranking_specs[name] = {'method':'const','const_weights':w}

	def output_rankings(self,ranking_name):
		if not os.path.isfile('rankings/'+self.model_name+'_'+ranking_name+'_runs.txt'):
			with open('rankings/'+self.model_name+'_'+ranking_name+'_runs.txt', 'w') as outfile:
				json.dump(self.ranking_specs, outfile)
			self.df_weight_all.to_pickle('rankings/'+self.model_name+'_'+ranking_name+'_weight_df.pkl')
			self.df_sig.to_pickle('rankings/'+self.model_name+'_'+ranking_name+'_signal_df.pkl')
			self.df_ret.to_pickle('rankings/'+self.model_name+'_'+ranking_name+'_ret_df.pkl')
		else:
			print('already output ranking '+self.model_name+'_'+ranking_name)

	def analyze_rankings(self,rankings=[]):
		min_date = self.df_weight_all.index.min().strftime('%Y-%m')
		max_date = self.df_weight_all.index.max().strftime('%Y-%m')
		rec = get_recession_dates()
		rec = rec.loc[(rec.index<=max_date)&(rec.index>=min_date)]	

		if type(rankings)!=list:
			rankings = [rankings]
		if len(rankings)==0:
			rankings = list(self.ranking_specs.keys())
		n_runs = len(rankings)
		fig,ax = plt.subplots(n_runs,1,figsize=(12,n_runs*2.5))
		d = self.df_weight_all.index[1]
		for i,name in enumerate(rankings):
			cols = [name+' '+x for x in self.model]
			to_plot = self.df_weight_all[cols]
			to_plot = to_plot.divide(to_plot.sum(1),0)
			to_plot.plot(ax=ax[i],title='',color=colors)
			ax[i].set_xlabel('')
			ylim = ax[i].get_ylim()
			ax[i].text(d,ylim[0]+.9*(ylim[1]-ylim[0]),name)
			ax[i].fill_between(self.df_weight_all.index.values, ylim[0], ylim[1], rec.values[:,0], facecolor='k', alpha=0.1)
			if i==0:
				ax[i].legend(self.model)
				ax[i].set_title('Signal weights in each ranking')
			else:
				ax[i].legend().set_visible(False)
		plt.show()

		df_runs = pd.DataFrame(self.ranking_specs).T.loc[rankings]
		start_date=pd.to_datetime(df_runs['start_date']).max()
		temp = self.df_ret.loc[self.df_ret.index>=start_date][rankings]
		plot_return(temp,rankings)
		plot_model(temp,rankings,model_name=self.model_name)
	
def import_rankings(model_name,ranking_name):
	df_weight = pd.read_pickle('rankings/'+model_name+'_'+ranking_name+'_weight_df.pkl')
	df_sig = pd.read_pickle('rankings/'+model_name+'_'+ranking_name+'_signal_df.pkl')
	df_ret = pd.read_pickle('rankings/'+model_name+'_'+ranking_name+'_ret_df.pkl')
	with open('rankings/'+model_name+'_'+ranking_name+'_runs.txt', 'r') as infile:
		ranking_specs = json.loads(infile.read())
	return df_weight,df_sig,df_ret,ranking_specs

class PortfolioConstructor(SecurityRanker):
	def __init__(self,model_name,ranking_name,rankings_to_use=[],portfolio_name=''):
		self.ranking_name = ranking_name
		if portfolio_name=='':
			if type(rankings_to_use)!=list:
				rankings_to_use = [rankings_to_use]
			self.rankings_to_use = rankings_to_use
			SecurityRanker.__init__(self,model_name=model_name,ranking_name=ranking_name)
			self.df_sig['overall_ranking'] = self.df_sig[rankings_to_use].mean(1)
			self.df_sig['overall_ranking'] = self.df_sig.groupby('date')['overall_ranking'].apply(normalize_signal_ranks)
			self.df_sig['temp'] = self.df_sig['overall_ranking']*self.df_sig['retadj']
			self.df_ret['overall_ranking'] = self.df_sig.groupby('date')['temp'].sum()
			self.df_sig.drop(columns=['temp'],inplace=True)		
			df_runs = pd.DataFrame(self.ranking_specs).T
			self.start_date=pd.to_datetime(df_runs['start_date']).min()
			self.df_sig = self.df_sig.loc[self.df_sig.index.get_level_values(1)>=self.start_date]
			self.df_ret = self.df_ret.loc[self.df_ret.index>=self.start_date]
			self.portfolio_specs = {}
		else:
			print('importing from saved portfolio '+portfolio_name)
			self.model_name = model_name
			self.df_sig,self.df_ret,self.portfolio_specs = import_portfolios(model_name,ranking_name,portfolio_name)
			self.rankings_to_use = self.portfolio_specs[list(self.portfolio_specs.keys())[0]]['rankings_used']
			self.start_date = pd.to_datetime(self.portfolio_specs[list(self.portfolio_specs.keys())[0]]['start_date'])
			self.end_date = self.df_ret.index[-1]
		self.date = self.start_date
		self.df_security_weights = pd.DataFrame()
		self.w_prev = None

	def step_construct(self,n_securities,turn_penalty,bidask_penalty,rebalance_freq):
		print('getting portfolio weights for '+self.date.strftime('%Y-%m'))
		df = self.df_sig.reset_index()
		ranking = df.loc[df.date==self.date].set_index('permno')['overall_ranking']
		bidask_lag = df.loc[df.date==self.date].set_index('permno')['bidask_lag'].fillna(0)
		bidask_lag = bidask_lag-bidask_lag.median()
		if (self.date==self.start_date)|(turn_penalty==0):
			w = select_securities(ranking,n_securities,bidask_lag=bidask_lag,bidask_penalty=bidask_penalty)
		else:
			w_prev = self.w_prev.reindex(ranking.index).fillna(0)
			w_prev = (w_prev>0)*1-(w_prev<0)*1
			w = select_securities(ranking,n_securities,w_prev,turn_penalty,bidask_lag=bidask_lag,bidask_penalty=bidask_penalty)
		self.w_prev = w
		out = pd.DataFrame({'permno':w.index,'date':[self.date]*len(w),'weight':w.values})
		self.date+=DateOffset(months=rebalance_freq)
		self.df_security_weights = self.df_security_weights.append(out)
	
	def construct_portfolio(self,name,n_securities,turn_penalty,bidask_penalty,rebalance_freq):
		while self.date<=self.end_date:
			self.step_construct(n_securities,turn_penalty,bidask_penalty,rebalance_freq)
		self.df_sig[name] = self.df_security_weights.set_index(['permno','date'])['weight']
		self.df_sig['temp'] = self.df_sig[name]*self.df_sig['retadj']
		self.df_ret[name] = self.df_sig.groupby('date')['temp'].sum()
		self.df_sig.drop(columns=['temp'],inplace=True)
		self.portfolio_specs[name] = {'rankings_used':self.rankings_to_use,'start_date':self.start_date.strftime('%Y-%m-%d'),'n_securities':n_securities,'turn_penalty':turn_penalty,'rebalance_freq':rebalance_freq}
		self.date = self.start_date
		self.w_prev = None
		self.df_security_weights = pd.DataFrame()

	def output_portfolios(self,portfolio_name):
		if not os.path.isfile('portfolios/'+self.model_name+'_'+self.ranking_name+'_'+portfolio_name+'_runs.txt'):
			with open('portfolios/'+self.model_name+'_'+self.ranking_name+'_'+portfolio_name+'_runs.txt', 'w') as outfile:
				json.dump(self.portfolio_specs, outfile)
			self.df_sig.to_pickle('portfolios/'+self.model_name+'_'+self.ranking_name+'_'+portfolio_name+'_signal_df.pkl')
			self.df_ret.to_pickle('portfolios/'+self.model_name+'_'+self.ranking_name+'_'+portfolio_name+'_ret_df.pkl')
		else:
			print('already output portfolios '+self.model_name+'_'+self.ranking_name+'_'+portfolio_name)
	
	def analyze_portfolios(self,portfolios=[]):
		if type(portfolios)!=list:
			portfolios = ['overall_ranking',portfolios]
		if len(portfolios)==0:
			portfolios = ['overall_ranking']+list(self.portfolio_specs.keys())
		start_date = pd.DataFrame(self.portfolio_specs).T['start_date'].min()
		temp_sig = self.df_sig.loc[self.df_sig.index.get_level_values(1)>=start_date].reset_index()
		temp_ret = self.df_ret.loc[self.df_ret.index>=start_date]
		plot_trading_cost(temp_sig,temp_ret,portfolios,model_name='ff')

def import_portfolios(model_name,ranking_name,portfolio_name):
	df_sig = pd.read_pickle('portfolios/'+model_name+'_'+ranking_name+'_'+portfolio_name+'_signal_df.pkl')
	df_ret = pd.read_pickle('portfolios/'+model_name+'_'+ranking_name+'_'+portfolio_name+'_ret_df.pkl')
	with open('portfolios/'+model_name+'_'+ranking_name+'_'+portfolio_name+'_runs.txt', 'r') as infile:
		portfolio_specs = json.loads(infile.read())
	return df_sig,df_ret,portfolio_specs

def select_securities(ranking,n_securities,w_prev=None,turn_penalty=0,bidask_lag=None,bidask_penalty=0):
	x = cp.Variable(len(ranking),boolean=True)
	y = cp.Variable(len(ranking),boolean=True)
	constraints = [cp.sum(x)<=n_securities,cp.sum(y)<=n_securities]
	if (turn_penalty==0)&(bidask_penalty==0):
		obj = cp.Minimize(-(x-y)*ranking)
	elif bidask_penalty==0:
		obj = cp.Minimize(-(x-y)*ranking+turn_penalty*cp.sum(cp.abs((x-y)-w_prev))/n_securities)
	elif turn_penalty==0:
		obj = cp.Minimize(-(x-y)*ranking+bidask_penalty*(x+y)*bidask_lag/n_securities)
	else:
		obj = cp.Minimize(-(x-y)*ranking+turn_penalty*cp.sum(cp.abs((x-y)-w_prev))/n_securities+bidask_penalty*(x+y)*bidask_lag/n_securities)
	prob = cp.Problem(obj, constraints)
	prob.solve()
	x = (x.value>.5)*1
	y = (y.value>.5)*1
	w = x/np.sum(x)-y/np.sum(y)
	return pd.Series(w,index=ranking.index)