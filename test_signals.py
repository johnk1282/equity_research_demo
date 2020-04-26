import os
import multiprocessing as mp
import pandas as pd
from pandas.tseries.offsets import *
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from arch import arch_model
import matplotlib.pyplot as plt
import seaborn as sns
import json 
from utils import *
from pull_data import *
from create_feature_df import *

colors = ['darkgrey','indianred','cornflowerblue','mediumseagreen','darkorange','goldenrod','mediumpurple']

class Backtester(object):
	def __init__(self,
				overwrite_features=False,
				feat_path='',
				feat_file='features.pkl',
				univ_name='',
				univ_size=3000,
				univ_start_date='1980-01-01',
				model_name='',
				signal_specs=[],
				feat_list=[],
				weights=[1],
				industry_adjust=False,
				ffill=3):
		print('getting feature dataframe from '+feat_path+feat_file)
		self.df_feat = get_features(overwrite=overwrite_features,feat_path=feat_path,feat_file=feat_file)
		if model_name!='':
			print('initializing from output model '+model_name)
			self.df_sig,self.df_ret,self.model_spec = import_model(model_name)
		else:
			self.df_univ = get_universe(n_stocks=univ_size,start_date=univ_start_date,univ_name=univ_name+'.pkl')
			if len(signal_specs)>0:
				print('initializing model from signal specifications')
				self.model_spec = signal_specs
			elif len(feat_list)>0:
				print('initializing model from feature names')
				signal_specs = [{'signal_name':x,'features':[x],'weights':weights,'industry_adjust':industry_adjust,'ffill':ffill} for x in feat_list]
				self.model_spec = signal_specs
			else:
				raise ValueError('must provide name of already output model, list of signal specifications, or list of features for model')
			print('creating signal dataframe')
			self.df_sig,self.df_ret = create_signal_df(self.df_univ,self.df_feat,self.model_spec)

	def output_model(self,model_name):
		if not os.path.isfile('models/'+model_name+'_model_spec.txt'):
			with open('models/'+model_name+'_model_spec.txt', 'w') as outfile:
				json.dump(self.model_spec, outfile)
			self.df_sig.to_pickle('models/'+model_name+'_signal_df.pkl')
			self.df_ret.to_pickle('models/'+model_name+'_ret_df.pkl')
		else:
			print('already output model with name '+model_name)
	
	def backtest_signals(self,signals,
					start_date=None,
					end_date=None,
					ic_analysis=False,
					return_analysis=False,
					quantile_analysis=False,
					model_analysis=False,
					average_signal_analysis=False,
					interaction_analysis=False,
					time_series_analysis=False,
					trading_cost_analysis=False,
					quantiles=10,
					signal_to_interact=None,
					model_name='ff'):

		if (start_date==None)&(end_date==None):
			start_date='1900-01-01'
			end_date=pd.Timestamp.today()
		elif (start_date==None):
			start_date='1900-01-01'
		elif (end_date==None):
			end_date=pd.Timestamp.today()
		temp = self.df_sig.reset_index()
		temp  = temp.loc[(temp['date']>=start_date)&(temp['date']<end_date)]
		temp_ret  = self.df_ret.loc[(self.df_ret.index>=start_date)&(self.df_ret.index<end_date)].copy()
		if type(signals)!=list:
			signals = [signals]

		if ic_analysis:
			plot_ic(temp,signals)
		if return_analysis:
			plot_return(temp_ret,signals)
		if quantile_analysis:
			plot_quantile(temp,signals,quantiles)
		if model_analysis:
			plot_model(temp_ret,signals,model_name)
		if average_signal_analysis:
			plot_average_signal(temp,signals,model_name)
		if interaction_analysis:
			plot_interaction(temp,signals,signal_to_interact,model_name)
		if time_series_analysis:
			plot_time_series(temp,temp_ret,self.df_feat,signals)
		if trading_cost_analysis:
			plot_trading_cost(temp,temp_ret,signals,model_name)

def import_model(model_name):
	df_sig = pd.read_pickle('models/'+model_name+'_signal_df.pkl')
	df_ret = pd.read_pickle('models/'+model_name+'_ret_df.pkl')
	with open('models/'+model_name+'_model_spec.txt', 'r') as infile:
		model_spec = json.loads(infile.read())
	return df_sig,df_ret,model_spec

def create_signal_df(df_univ,df_feat,model):
	temp = df_univ.copy()
	sig_list = [x['signal_name'] for x in model]
	feat_nlist = [x['features'] for x in model]
	feat_list = list(set([item for sublist in feat_nlist for item in sublist]))
	w_nlist = [x['weights'] for x in model]
	avg_list = [len(x) for x in feat_nlist]
	average = max(avg_list)>1
	ia_list = [x['industry_adjust'] for x in model]
	industry_adjust = max(ia_list)
	univ_cols = list(temp.columns)
	temp[feat_list] = df_feat[feat_list]
	temp = temp.dropna(subset=['retadj'])
	#ffill signal
	ffill_list = [x['ffill'] for x in model]
	if len(set(ffill_list))==1:
		if ffill_list[0]>0:
			temp[feat_list] = temp.groupby('permno')[feat_list].fillna(method='ffill',limit=ffill_list[0])
	else:
		#create groups of features based on how much to ffill
		feat_groups = []
		for ffill in set(ffill_list):
			to_add = [x['features'] for x in model if x['ffill']==ffill]
			to_add = list(set([item for sublist in to_add for item in sublist]))
			feat_groups.append(to_add)
		#check that each feature only has one ffill value 
		if sum([len(x) for x in feat_groups])!=len(feat_list):
			raise ValueError('cannot have multiple ffill parameters for a single feature')
		for i,ffill in enumerate(set(ffill_list)):
			if ffill>0:
				temp[feat_groups[i]] = temp.groupby('permno')[feat_groups[i]].fillna(method='ffill',limit=ffill)
	#if combining features, need to normalize first
	if average:
		temp[feat_list] = temp.groupby('date')[feat_list].apply(normalize_signal_ranks)
	#compute weighted averages of features
	for i,sig in enumerate(sig_list):
		temp[sig] = temp[feat_nlist[i]].multiply(w_nlist[i]).mean(1)
	#adjust within industries
	if industry_adjust:
		sig_ia = [sig_list[i] for i,x in enumerate(ia_list) if x==True]
		df_ia = multi_groupby(temp[['industry']+sig_ia],['date','industry'],normalize_chars)
		temp[sig_ia] = df_ia[sig_ia]
	#finally normalize into rank weights
	temp[sig_list] = temp.groupby('date')[sig_list].apply(normalize_signal_ranks).fillna(0)
	#compute return time series
	temp[[x+'r' for x in sig_list]] = temp[sig_list].multiply(temp['retadj'],0)
	df_ret = temp.groupby('date')[[x+'r' for x in sig_list]].sum()
	df_ret.columns = sig_list
	df_sig = temp[univ_cols+sig_list]
	return df_sig,df_ret

def normalize_chars(df_sub):
	#helper function that normalizes characteristics in cross-section and within industry; will be fed into the multi-groupby function
	to_norm = [x for x in df_sub.columns if x not in ['permno','date','industry']]
	df_sub[to_norm] = (df_sub[to_norm].rank()-df_sub[to_norm].rank().mean())/df_sub[to_norm].rank().max()
	return df_sub

def plot_ic(df,signals):
	min_date = df['date'].min().strftime('%Y-%m')
	max_date = df['date'].max().strftime('%Y-%m')
	rec = get_recession_dates()
	rec = rec.loc[(rec.index<=max_date)&(rec.index>=min_date)]
	for f in signals:
		title1 = f+' signal rank IC; t-stat: '
		retcols = ['retadj']+['retadj'+str(i) for i in range(1,13)]
		tstats = pd.DataFrame()
		ic = df.groupby('date')[[f]+retcols].corr()
		ic = ic[f].unstack()
		tstats['Ranks'] = ic[retcols].mean()/ic[retcols].std()*np.sqrt(ic[retcols].count())
		fig,ax = plt.subplots(2,1,figsize=(12,10),sharex=False)
		ax[0].bar(ic.index,ic['retadj'],label=f+' IC',width=20,color=colors[0])
		ax[0].plot(ic.index,ic['retadj'].rolling(24).mean(),label='2 Yr. Avg.',color=colors[1])
		ax[0].set_title('IC analysis: '+min_date+' through '+max_date+'\n'+title1+str(round(ic['retadj'].mean()/ic['retadj'].std()*np.sqrt(ic['retadj'].count()),2)))
		ax[0].legend()
		ylim = ax[0].get_ylim()
		ax[0].fill_between(ic.index.values, ylim[0], ylim[1], rec.values[:,0], facecolor='k', alpha=0.1)
		tstats.index = ['t+'+str(i) for i in range(1,len(tstats)+1)]
		tstats.plot(kind='bar',color=colors,edgecolor='k',ax=ax[1],legend=False)
		plt.axhline(2,color='k',linestyle='--')
		plt.axhline(-2,color='k',linestyle='--')
		plt.title('Signal decay: t-stat of IC')
		plt.xlabel('Forward return')
		plt.show()

def plot_quantile(df,signals,quantiles=10):
	min_date = df['date'].min().strftime('%Y-%m')
	max_date = df['date'].max().strftime('%Y-%m')
	rec = get_recession_dates()
	rec = rec.loc[(rec.index<=max_date)&(rec.index>=min_date)]
	for f in signals:
		df[f+'_quantile'] = df.groupby('date')[f].transform(lambda x: ((x.rank(method='first')/(x.count()+.00001))*100)//(100/quantiles))
		qrets = df.groupby(['date',f+'_quantile'])['retadj'].mean().unstack()
		qmax = int(qrets.columns.max())
		qrets.columns = list(range(1,qmax+2))
		qrets[str(qmax+1)+'-'+str(1)] = qrets[qmax+1]-qrets[1]
		df['wr'] = df['me_lag']*df['retadj']
		qrets2 = df.groupby(['date',f+'_quantile'])['wr'].sum().unstack()/df.groupby(['date',f+'_quantile'])['me_lag'].sum().unstack()
		qrets2.columns = list(range(1,qmax+2))
		qrets2[str(qmax+1)+'-'+str(1)] = qrets2[qmax+1]-qrets2[1]
		df_q = pd.DataFrame()
		df_q['EW'] = qrets.mean()*12
		df_q['VW'] = qrets2.mean()*12

		df_ret = pd.DataFrame()
		df_ret[str(qmax+1)+'-'+str(1)+' EW'] = qrets[str(qmax+1)+'-'+str(1)]
		df_ret[str(qmax+1)+'-'+str(1)+' VW'] = qrets2[str(qmax+1)+'-'+str(1)]
		stats = pd.DataFrame()
		stats['Mean'] = df_ret.mean()*12
		stats['Vol'] = df_ret.std()*np.sqrt(12)
		stats['Sharpe'] = stats['Mean']/stats['Vol']
		cr = (1+df_ret).cumprod()

		fig = plt.figure(figsize=(12,15))
		gs = fig.add_gridspec(3,2)
		ax0 = fig.add_subplot(gs[0,:])
		ax10 = fig.add_subplot(gs[1,0])
		ax11 = fig.add_subplot(gs[1,1])
		ax2 = fig.add_subplot(gs[2,:])
		df_q.plot(kind='bar',color=colors,title='Quantiles analysis: '+min_date+' through '+max_date+'\n'+'Annualized mean return by '+f+' quantile',edgecolor='k',ax=ax0)
		cr.plot(logy=True,ax=ax2,color=colors,title='Top-minus-bottom quantile portfolios')
		cmap = sns.diverging_palette(240, 10, as_cmap=True)
		(1+qrets[[x for x in range(1,(qmax+2))]]).cumprod().plot(logy=True,colormap=cmap,ax=ax10,title='EW')
		(1+qrets2[[x for x in range(1,(qmax+2))]]).cumprod().plot(logy=True,colormap=cmap,ax=ax11,title='VW',legend=False)
		for i,col in enumerate(df_ret.columns):
			ax2.annotate('Sharpe: '+str(round(stats['Sharpe'].loc[col],2)),(cr.index.values[-1],cr[col].values[-1]))
		ax0.axvline(9.5,color='k',linestyle='--')
		ax10.set_xlabel('')
		ax11.set_xlabel('')
		ax2.set_xlabel('')
		ylim = ax10.get_ylim()
		ax10.fill_between(cr.index.values, ylim[0], ylim[1], rec.values[:,0], facecolor='k', alpha=0.1)
		ylim = ax11.get_ylim()
		ax11.fill_between(cr.index.values, ylim[0], ylim[1], rec.values[:,0], facecolor='k', alpha=0.1)
		ylim = ax2.get_ylim()
		ax2.fill_between(cr.index.values, ylim[0], ylim[1], rec.values[:,0], facecolor='k', alpha=0.1)
		plt.show()

def plot_return(df_ret,signals):
	min_date = df_ret.index.min().strftime('%Y-%m')
	max_date = df_ret.index.max().strftime('%Y-%m')
	rec = get_recession_dates()
	rec = rec.loc[(rec.index<=max_date)&(rec.index>=min_date)]
	
	rets = df_ret[signals]
	stats=pd.DataFrame()
	stats['Mean'] = rets.mean()*12
	stats['Vol'] = rets.std()*np.sqrt(12)
	stats['Sharpe']=rets.mean()/rets.std()*np.sqrt(12)
	cr = (1+rets).cumprod()

	corr = len(signals)>=3
	if corr:
		fig = plt.figure(figsize=(12,15))
		gs = fig.add_gridspec(3,6)
		ax2 = fig.add_subplot(gs[2,1:6])
	else:
		fig = plt.figure(figsize=(12,10))
		gs = fig.add_gridspec(2,6)		
	ax0 = fig.add_subplot(gs[0,:])
	ax10 = fig.add_subplot(gs[1,:4])
	ax11 = fig.add_subplot(gs[1,4:])
	if len(signals)==1:
		cr.plot(logy=True,ax=ax0,color=colors[1],title='Return analysis: '+min_date+' through '+max_date+'\n'+signals[0]+' Long-short portfolio',legend=False)
	else:
		cr.plot(logy=True,ax=ax0,color=colors,title='Return analysis: '+min_date+' through '+max_date+'\n'+'Long-short portfolios',legend=True)
	for i,col in enumerate(rets.columns):
		ax0.annotate('Sharpe: '+str(round(stats['Sharpe'].loc[col],2)),(cr.index.values[-1],cr[col].values[-1]))
	ax0.set_xlabel('')
	ylim = ax0.get_ylim()
	ax0.fill_between(cr.index.values, ylim[0], ylim[1], rec.values[:,0], facecolor='k', alpha=0.1)
	
	stats[['Mean','Vol']].T.plot(kind='bar',ax=ax10,edgecolor='k',title='Summary Stats',color=colors,width=.7)
	stats['Sharpe'].plot(kind='bar',ax=ax11,edgecolor='k',title='Sharpe',color=colors,width=.7)
	ax11.yaxis.set_label_position("right")
	ax11.yaxis.tick_right()
	ax11.set_xticks([])
	for rect in ax10.patches:
		a = str(round(rect.get_height()*100,1))
		ax10.annotate(a,
			xy=(rect.get_x()+rect.get_width()/2,rect.get_height()),
			xytext=(0, 0),
			textcoords="offset points",
			ha='center', va='bottom')
	for rect in ax11.patches:
		a = str(round(rect.get_height(),2))
		ax11.annotate(a,
			xy=(rect.get_x()+rect.get_width()/2,rect.get_height()),
			xytext=(0, 0),
			textcoords="offset points",
			ha='center', va='bottom')
	ax10.set_xticklabels(ax10.get_xticklabels(),rotation=0)

	if corr:
		corr = rets.corr()
		mask = np.triu(np.ones_like(corr, dtype=np.bool))

		cmap = sns.diverging_palette(240, 10, as_cmap=True)
		g = sns.heatmap(corr,mask=mask,annot=True,vmin=-1,vmax=1,square=False,cmap=cmap,linewidths=.5,ax=ax2)
		g.set_xticklabels(g.get_xticklabels(),rotation=10)
		ax2.set_title('Correlation matrix')

	plt.show()

def plot_model(df_ret,signals,model_name='ff'):
	min_date = df_ret.index.min().strftime('%Y-%m')
	max_date = df_ret.index.max().strftime('%Y-%m')
	rec = get_recession_dates()
	rec = rec.loc[(rec.index<=max_date)&(rec.index>=min_date)]
	rets = df_ret[signals].copy()
	if model_name=='ff':
		ff = get_ff_model()
		model_vars = ['Mkt-RF','HML','SMB','UMD']
		rets[model_vars] = ff[model_vars]
	else:
		model_rets = pd.read_pickle('models/'+model_name+'_ret_df.pkl')
		ff = get_ff_model()
		model_rets['Mkt-RF'] = ff['Mkt-RF']
		model_vars = list(model_rets.columns)
		rets[model_vars] = model_rets[model_vars]

	cr = (1+rets).cumprod()
	rets['Const.'] = 1
	params = pd.DataFrame(index=model_vars+['Const.'])
	tvalues = pd.DataFrame(index=model_vars+['Const.'])
	alpha_ts = pd.DataFrame()
	for f in signals:
		res = sm.OLS(rets[f],rets[model_vars+['Const.']]).fit()
		params[f]=res.params
		tvalues[f]=res.tvalues
		alpha_ts[f]=rets[f]-rets[model_vars].dot(res.params[:-1])
	cr_alpha = (1+alpha_ts).cumprod()

	if len(signals)==1:
		fig = plt.figure(figsize=(12,15))
		gs = fig.add_gridspec(3,6)
		ax0 = fig.add_subplot(gs[0,:])
		ax1 = fig.add_subplot(gs[1,:4])
		ax2 = fig.add_subplot(gs[1,4:])
		ax3 = fig.add_subplot(gs[2,:])
		cr.plot(ax=ax0,color=colors)
		ax0.legend(cr.columns)
		ax0.set_yscale('log')
		ax0.set_title('Model analysis: '+min_date+' through '+max_date+'\n'+'Signal vs. '+model_name+' model factors')
		ylim = ax0.get_ylim()
		ax0.fill_between(cr.index.values, ylim[0], ylim[1], rec.values[:,0], facecolor='k', alpha=0.1)
		ax0.set_xlabel('')
	else:
		fig = plt.figure(figsize=(12,10))
		gs = fig.add_gridspec(2,6)
		ax1 = fig.add_subplot(gs[0,:4])
		ax2 = fig.add_subplot(gs[0,4:])
		ax3 = fig.add_subplot(gs[1,:])

	params.iloc[:-1,:].plot(kind='bar',ax=ax1,edgecolor='k',width=.7,title=model_name+' model betas',color=colors)
	tvalues_shape = tvalues.iloc[:-1,:].T.stack().values
	for i,rect in enumerate(ax1.patches):
		t = tvalues_shape[i]
		if abs(t)>4:
			l = '***'
		elif abs(t)>2.7:
			l = '**'
		elif abs(t)>2:
			l='*'
		else:
			l=''
		if t<0:
			m = -1
		if t>0:
			m = 0
		ax1.annotate(l,
			xy=(rect.get_x()+rect.get_width()/2,rect.get_height()),
			xytext=(0, m*15),
			textcoords="offset points",
			ha='center', va='bottom')
	ax1.set_xticklabels(ax1.get_xticklabels(),rotation=20)

	(params.iloc[-1,:]*12*10000).plot(kind='bar',ax=ax2,edgecolor='k',title=model_name+' model alpha',color=colors)
	tvalues_shape2 = tvalues.iloc[-1,:].T.values
	for i,rect in enumerate(ax2.patches):
		t = tvalues_shape2[i]

		if abs(t)>4:
			l = '***'
		elif abs(t)>2.7:
			l = '**'
		elif abs(t)>2:
			l='*'
		else:
			l=''
		if t<0:
			m = -1
		if t>0:
			m = 0
		ax2.annotate(l,
			xy=(rect.get_x()+rect.get_width()/2,rect.get_height()),
			xytext=(0, m*15),
			textcoords="offset points",
			ha='center', va='bottom')
	ax2.yaxis.set_label_position("right")
	ax2.yaxis.tick_right()
	ax2.set_xticks([])

	cr_alpha = (1+alpha_ts).cumprod()
	cr_alpha.plot(ax=ax3,color=colors,title='Ex-post alpha',logy=True)
	ax3.legend(cr_alpha.columns)
	ax3.set_xlabel('')
	ylim = ax3.get_ylim()
	ax3.fill_between(cr.index.values, ylim[0], ylim[1], rec.values[:,0], facecolor='k', alpha=0.1)
	plt.show()

def plot_average_signal(df,signals,model_name='ff'):
	if len(signals)==1:
		raise ValueError('need multiple signals to average')
	df['average'] = df[signals].mean(1)
	df['average'] = df.groupby('date')['average'].apply(normalize_signal_ranks)
	signals = signals+['average']
	df[signals] = df[signals].multiply(df['retadj'],0)
	rets = df.groupby('date')[signals].sum()
	plot_return(rets,signals)
	plot_model(rets,signals,model_name=model_name)
    
def ols_alpha_t(series,mod):
	#alpha tvalue given model with constant as last column
	return sm.OLS(series.dropna(),mod.loc[series.dropna().index]).fit().tvalues[-1]

def plot_interaction(df,signals,f2,model_name='ff',quantiles=10):
	min_date = df['date'].min().strftime('%Y-%m')
	max_date = df['date'].max().strftime('%Y-%m')
	rec = get_recession_dates()
	rec = rec.loc[(rec.index<=max_date)&(rec.index>=min_date)]
	for f1 in signals:
		df[[x+'_quantile' for x in [f1,f2]]] = df.groupby('date')[[f1,f2]].transform(lambda x: ((x.rank(method='first')/(x.count()+.00001))*100)//(100/quantiles))
		if df[f1+'_quantile'].max()==9:
			df[[f1+'_q',f2+'_q']] = df[[f1+'_quantile',f2+'_quantile']].replace(dict(zip(range(10),[1,1,2,2,3,3,4,4,5,5])))
			df[[f1+'_q2',f2+'_q2']] = df[[f1+'_quantile',f2+'_quantile']].replace(dict(zip(range(10),[1,1,1,1,2,2,3,3,3,3])))
			qmax = 5
			q2max = 3
		else:
			df[[f1+'_q',f2+'_q']] = df[[f1+'_quantile',f2+'_quantile']]+1
			df[[f1+'_q2',f2+'_q2']] = df[[f1+'_quantile',f2+'_quantile']]+1
			qmax = df[f1+'_quantile'].max()
			q2max = df[f1+'_quantile'].max()
		rets = df.groupby(['date',f1+'_q',f2+'_q'])['retadj'].mean().unstack([1,2])
		rets2 = df.groupby(['date',f1+'_q2',f2+'_q2'])['retadj'].mean().unstack([1,2])
		to_plot = pd.DataFrame(index=rec.index)
		to_plot[f2+' Low, '+f1+' Low'] = rets2[1][1]
		to_plot[f2+' Low, '+f1+' High'] = rets2[q2max][1]
		to_plot[f2+' High, '+f1+' Low'] = rets2[1][q2max]
		to_plot[f2+' High, '+f1+' High'] = rets2[q2max][q2max]
		mean_rets = rets.mean().unstack()*12
		if model_name=='ff':
			ff = get_ff_model()
			model_vars = ['Mkt-RF','HML','SMB','UMD']
			model_rets = ff[model_vars]
			model_rets['const']=1
		else:
			model_rets = pd.read_pickle('models/'+model_name+'_ret_df.pkl')
			ff = get_ff_model()
			model_rets['Mkt-RF'] = ff['Mkt-RF']
			model_vars = list(model_rets.columns)
			model_rets['const'] = 1
		alpha_ts = rets.apply(lambda x: ols_alpha_t(x,model_rets)).unstack()

		fig = plt.figure(figsize=(12,10))
		gs = fig.add_gridspec(2,26)
		ax0 = fig.add_subplot(gs[1,1:13])
		ax1 = fig.add_subplot(gs[1,14:26])
		ax2 = fig.add_subplot(gs[0,:])
		cmap = sns.diverging_palette(240, 10, as_cmap=True)
		sns.heatmap(mean_rets,annot=True,cmap=cmap,linewidths=.5,ax=ax0)
		sns.heatmap(alpha_ts,annot=True,cmap=cmap,linewidths=.5,ax=ax1)
		ax0.set_title('Annualized mean returns')
		ax1.set_title('t-stat of model alpha')
		ax0.set_ylabel(f1)
		ax0.set_xlabel(f2)
		ax1.set_ylabel('')
		ax1.set_xlabel(f2)
		(1+to_plot.iloc[:,[0,2]]).cumprod().plot(ax=ax2,logy=True,title='Interaction analysis: '+min_date+' through '+max_date+'\n'+f1+' x '+f2,color=colors,linestyle='--')
		(1+to_plot.iloc[:,[1,3]]).cumprod().plot(ax=ax2,logy=True,title='Interaction analysis: '+min_date+' through '+max_date+'\n'+f1+' x '+f2,color=colors)
		ylim = ax2.get_ylim()
		ax2.fill_between(to_plot.index.values, ylim[0], ylim[1], rec.values[:,0], facecolor='k', alpha=0.1)
		plt.show()

def plot_time_series(df_sig,df_ret,df_feat,signals):
	min_date = df_sig['date'].min().strftime('%Y-%m')
	max_date = df_sig['date'].max().strftime('%Y-%m')
	rec = get_recession_dates()
	rec = rec.loc[(rec.index<=max_date)&(rec.index>=min_date)]

	for f in signals:
		fig = plt.figure(figsize=(12,22))
		gs = fig.add_gridspec(5,2)
		ax0 = fig.add_subplot(gs[0,0])
		ax1 = fig.add_subplot(gs[0,1])
		ax2 = fig.add_subplot(gs[1,0])
		ax3 = fig.add_subplot(gs[1,1])
		ax4 = fig.add_subplot(gs[2,0])
		ax5 = fig.add_subplot(gs[2,1])
		ax6 = fig.add_subplot(gs[3,0])
		ax7 = fig.add_subplot(gs[3,1])
		ax8 = fig.add_subplot(gs[4,0])
		ax9 = fig.add_subplot(gs[4,1])
		
		plot_acf(df_ret[f].values,ax=ax0,title=f+': Autocorrelation Function')
		plot_pacf(df_ret[f].values,ax=ax1,title=f+': Partial-Autocorrelation Function')
		model = ARIMA(df_ret[f], order=(1,0,0),freq='MS')
		model_fit = model.fit(disp=0)
		out = pd.DataFrame()
		out['returns'] = df_ret[f]
		out['instantaneous vol'] = np.sqrt(out['returns']**2)
		out['fitted returns'] = model_fit.fittedvalues
		out[['returns','fitted returns']].plot(ax=ax2,title='AR1 Fitted Values',color=colors)
		ax2.axhline(out['fitted returns'].mean(),color='k',linestyle='--')
		out['high_ret'] = out['fitted returns']>out['fitted returns'].mean()
		means = out.groupby('high_ret')['returns'].mean()
		means.index = ['Low','High']
		means.plot(ax=ax3,kind='bar',color=colors[0],edgecolor='k',title='Return Conditional on Fitted Return')
		
		ar = arch_model(df_ret[f]*100, mean='AR', lags=1, vol='garch', p=1, o=0, q=1,dist='Normal')
		res = ar.fit(disp='off')
		out['fitted conditional vol'] = res.conditional_volatility/100
		out[['instantaneous vol','fitted conditional vol']].plot(ax=ax4,title='GARCH(1,1) Fitted Values',color=colors)
		ax4.axhline(out['fitted conditional vol'].mean(),color='k',linestyle='--')
		out['high_vol'] = out['fitted conditional vol']>out['fitted conditional vol'].mean()
		means = out.groupby('high_vol')['returns'].mean()
		means.index = ['Low','High']
		means.plot(ax=ax5,kind='bar',color=colors[0],edgecolor='k',title='Return Conditional on Fitted Vol')
		
		df_sig = df_sig.merge(df_feat.reset_index()[['be_me','permno','date']],how='left',on=['permno','date'])
		df_sig['posw'] = df_sig[f]>0
		ts = df_sig.groupby(['date','posw'])['be_me'].median().unstack()
		out['vs'] = (ts[True]-ts[False])
		#out['vs'] = ((out['vs']-out['vs'].mean())/out['vs'].std()).shift(1)
		out['vs'] = out['vs'].expanding(min_periods=36).apply(lambda x: (x.iloc[-1]-x.mean())/x.std()).shift(1)
		#out['vs'] = out['vs'].rolling(120,min_periods=36).apply(lambda x: (x.iloc[-1]-x.mean())/x.std()).shift(1)
		out['vs'].plot(title='Value Spread Z-Score (Long-Short Median BE/ME)',color=colors[1],ax=ax6)
		ax6.axhline(1,color='k',linestyle='--')
		ax6.axhline(-1,color='k',linestyle='--')
		out['vs_bucket'] = (out['vs']>1)*1+(out['vs']>0)*1
		means = out.groupby('vs_bucket')['returns'].mean()*12
		means.index = ['Low','Med','High']
		means.plot(ax=ax7,kind='bar',color=colors[0],edgecolor='k',title='Return Conditional on Value Spread')

		ff = get_ff_model()
		ar = arch_model(ff['Mkt-RF']*100, mean='AR', lags=1, vol='garch', p=1, o=0, q=1,dist='Normal')
		res = ar.fit(disp='off')
		out['Mkt-RF'] = ff['Mkt-RF'].rolling(12).mean().shift(-6)
		out['Mkt-RF instantaneous vol'] = np.sqrt(ff['Mkt-RF']**2)
		out['Mkt-RF conditional vol'] = res.conditional_volatility/100
		out[['Mkt-RF instantaneous vol','Mkt-RF conditional vol']].plot(ax=ax8,title='GARCH(1,1) Mkt-RF Fitted Values',color=colors)
		ax8.axhline(out['Mkt-RF conditional vol'].mean(),color='k',linestyle='--')
		#out['high_vol'] = out['Mkt-RF conditional vol']>out['Mkt-RF conditional vol'].mean()
		out['high_vol'] = out['Mkt-RF']>out['Mkt-RF'].mean()
		means = out.groupby('high_vol')['returns'].mean()
		means.index = ['Low','High']
		means.plot(ax=ax9,kind='bar',color=colors[0],edgecolor='k',title='Return Conditional on Mkt-RF Fitted Vol')
		
		ax2.set_xlabel('')
		ax3.set_xlabel('')
		ax4.set_xlabel('')
		ax5.set_xlabel('')
		ax6.set_xlabel('')
		ax7.set_xlabel('')
		ax8.set_xlabel('')
		ax9.set_xlabel('')
		ax3.set_xticklabels(ax3.get_xticklabels(),rotation=0)
		ax5.set_xticklabels(ax5.get_xticklabels(),rotation=0)
		ax7.set_xticklabels(ax7.get_xticklabels(),rotation=0)
		ax9.set_xticklabels(ax9.get_xticklabels(),rotation=0)
		ylim = ax2.get_ylim()
		ax2.fill_between(out.index.values, ylim[0], ylim[1], rec.values[:,0], facecolor='k', alpha=0.1)
		ylim = ax4.get_ylim()
		ax4.fill_between(out.index.values, ylim[0], ylim[1], rec.values[:,0], facecolor='k', alpha=0.1)
		ylim = ax6.get_ylim()
		ax6.fill_between(out.index.values, ylim[0], ylim[1], rec.values[:,0], facecolor='k', alpha=0.1)
		ylim = ax8.get_ylim()
		ax8.fill_between(out.index.values, ylim[0], ylim[1], rec.values[:,0], facecolor='k', alpha=0.1)
		plt.show()

def plot_trading_cost(df_sig,df_ret,signals,model_name='ff'):
	min_date = df_sig['date'].min().strftime('%Y-%m')
	max_date = df_sig['date'].max().strftime('%Y-%m')
	rec = get_recession_dates()
	rec = rec.loc[(rec.index<=max_date)&(rec.index>=min_date)]
	signals = list(signals)

	df_sig[[x+'_prev' for x in signals]] = df_sig.groupby('permno')[signals].shift(1).fillna(0)
	df_sig[[x+'_turn' for x in signals]] = (df_sig[signals]-df_sig[[x+'_prev' for x in signals]].values).abs()
	df_sig[[x+'_trade_cost' for x in signals]] = df_sig[[x+'_turn' for x in signals]].multiply(df_sig['bidask_lag'],0)/2
	turn = df_sig.groupby('date')[[x+'_turn' for x in signals]].sum()/2
	trade_cost = df_sig.groupby('date')[[x+'_trade_cost' for x in signals]].sum()
	df_ret[[x+'_net' for x in signals]] = df_ret[signals]-trade_cost.values

	fig = plt.figure(figsize=(12,20))
	gs = fig.add_gridspec(4,3)
	ax0 = fig.add_subplot(gs[0,:])
	ax1 = fig.add_subplot(gs[1,:])
	ax2 = fig.add_subplot(gs[2,:])
	ax3 = fig.add_subplot(gs[3,0])
	ax4 = fig.add_subplot(gs[3,1])
	ax5 = fig.add_subplot(gs[3,2])
	ax = [ax0,ax1,ax2,ax3,ax4,ax5]
	if len(signals)==1:
		turn.rolling(3).mean().plot(ax=ax[0],color=colors[1],title='Trading cost analysis: '+min_date+' through '+max_date+'\n'+'Monthly portfolio turnover (smoothed)')
		trade_cost.rolling(3).mean().plot(ax=ax[1],color=colors[1],title='Estimated monthly trading costs (smoothed)')
		sig = signals[0]
		ax[0].axhline(turn[sig+'_turn'].mean(),color='k',linestyle='--')
		ax[0].annotate(str(round(turn[sig+'_turn'].mean()*100))+'%',(turn.index.values[-1],turn[sig+'_turn'].mean()))
		mean_tc = trade_cost[sig+'_trade_cost'].loc[trade_cost.index>'2005-01-01'].mean()
		ax[1].axhline(mean_tc,color='k',linestyle='--',xmin=1-sum(trade_cost.index>'2005-01-01')/len(trade_cost))
		ax[1].annotate(str(round(mean_tc*10000))+' bps',(trade_cost.index.values[-1],mean_tc))
	else:
		turn.rolling(3).mean().plot(ax=ax[0],color=colors,title='Trading cost analysis: '+min_date+' through '+max_date+'\n'+'Monthly portfolio turnover (smoothed)')
		trade_cost.rolling(3).mean().plot(ax=ax[1],color=colors,title='Estimated monthly trading costs (smoothed)')
		for i,sig in enumerate(signals):
			ax[0].axhline(turn[sig+'_turn'].mean(),color=colors[i],linestyle='--')
			ax[0].annotate(str(round(turn[sig+'_turn'].mean()*100))+'%',(turn.index.values[-1],turn[sig+'_turn'].mean()))
			mean_tc = trade_cost[sig+'_trade_cost'].loc[trade_cost.index>'2005-01-01'].mean()
			ax[1].axhline(mean_tc,color=colors[i],linestyle='--',xmin=1-sum(trade_cost.index>'2005-01-01')/len(trade_cost))
			ax[1].annotate(str(round(mean_tc*10000))+' bps',(trade_cost.index.values[-1],mean_tc))
	ax[0].set_xlabel('')
	ylim = ax[0].get_ylim()
	ax[0].fill_between(turn.index.values, ylim[0], ylim[1], rec.values[:,0], facecolor='k', alpha=0.1)
	ax[1].set_xlabel('')
	ylim = ax[1].get_ylim()
	ax[1].fill_between(trade_cost.index.values, ylim[0], ylim[1], rec.values[:,0], facecolor='k', alpha=0.1)

	cr = (1+df_ret).cumprod()
	if len(signals)==1:
		cr[signals].plot(logy=True,ax=ax[2],color=colors[1],title='Long-short portfolios')
		cr[[x+'_net' for x in signals]].plot(logy=True,ax=ax[2],color=colors[1],linestyle='--',title='Long-short portfolios')
	else:
		cr[signals].plot(logy=True,ax=ax[2],color=colors,title='Long-short portfolios')
		cr[[x+'_net' for x in signals]].plot(logy=True,ax=ax[2],color=colors,linestyle='--',title='Long-short portfolios')
	ax[2].set_xlabel('')
	ylim = ax[2].get_ylim()
	ax[2].fill_between(cr.index.values, ylim[0], ylim[1], rec.values[:,0], facecolor='k', alpha=0.1)

	df_ret = df_ret[signals+[x+'_net' for x in signals]].copy()
	stats=pd.DataFrame()
	stats['Mean'] = df_ret.mean()*12
	stats['Vol'] = df_ret.std()*np.sqrt(12)
	stats['Sharpe']=df_ret.mean()/df_ret.std()*np.sqrt(12)
	if model_name=='ff':
		ff = get_ff_model()
		model_vars = ['Mkt-RF','HML','SMB','UMD']
		df_ret[model_vars] = ff[model_vars]
	else:
		model_rets = pd.read_pickle('models/'+model_name+'_ret_df.pkl')
		ff = get_ff_model()
		df_ret['Mkt-RF'] = ff['Mkt-RF']
		model_vars = list(model_rets.columns)
		df_ret[model_vars] = model_rets[model_vars]
	df_ret['Const.'] = 1
	params = pd.DataFrame(index=model_vars+['Const.'])
	tvalues = pd.DataFrame(index=model_vars+['Const.'])
	for f in signals+[x+'_net' for x in signals]:
		res = sm.OLS(df_ret[f],df_ret[model_vars+['Const.']]).fit()
		params[f]=res.params
		tvalues[f]=res.tvalues

	stats['Mean'][signals].plot(kind='bar',ax=ax[3],edgecolor='k',title='Mean',color=colors,width=.7,legend=False)
	stats['Mean'][[x+'_net' for x in signals]].plot(kind='bar',ax=ax[3],edgecolor='k',title='Mean',color=colors,width=.7,legend=False)
	stats['Sharpe'][signals].plot(kind='bar',ax=ax[4],edgecolor='k',title='Sharpe',color=colors,width=.7,legend=False)
	stats['Sharpe'][[x+'_net' for x in signals]].plot(kind='bar',ax=ax[4],edgecolor='k',title='Sharpe',color=colors,width=.7,legend=False)
	ax[3].set_xticks([])
	ax[4].set_xticks([])
	for rect in ax[3].patches:
		a = str(round(rect.get_height()*100,1))
		ax[3].annotate(a,
			xy=(rect.get_x()+rect.get_width()/2,rect.get_height()),
			xytext=(0, 0),
			textcoords="offset points",
			ha='center', va='bottom')
	for rect in ax[4].patches:
		a = str(round(rect.get_height(),2))
		ax[4].annotate(a,
			xy=(rect.get_x()+rect.get_width()/2,rect.get_height()),
			xytext=(0, 0),
			textcoords="offset points",
			ha='center', va='bottom')
	ax[4].set_xticklabels(ax[4].get_xticklabels(),rotation=0)

	(params.iloc[-1,:]*12*10000)[signals].plot(kind='bar',ax=ax[5],edgecolor='k',title=model_name+' model alpha',color=colors,legend=False,width=.7)
	(params.iloc[-1,:]*12*10000)[[x+'_net' for x in signals]].plot(kind='bar',ax=ax[5],edgecolor='k',title=model_name+' model alpha',color=colors,legend=False,width=.7)
	tvalues_shape2 = tvalues.iloc[-1,:].T.values
	for i,rect in enumerate(ax[5].patches):
		t = tvalues_shape2[i]
		if abs(t)>4:
			l = '***'
		elif abs(t)>2.7:
			l = '**'
		elif abs(t)>2:
			l='*'
		else:
			l=''
		if t<0:
			m = -1
		if t>0:
			m = 0
		ax[5].annotate(l,
			xy=(rect.get_x()+rect.get_width()/2,rect.get_height()),
			xytext=(0, m*15),
			textcoords="offset points",
			ha='center', va='bottom')
	ax[5].yaxis.set_label_position("right")
	ax[5].yaxis.tick_right()
	ax[5].set_xticks([])

	plt.show()

def get_universe(n_stocks=3000,start_date='1980-01-01',univ_path='universes/',univ_name=''):
	if (univ_name!='')&(os.path.isfile(univ_path+univ_name)):
		print('reading universe from '+univ_path+univ_name)
		univ = pd.read_pickle(univ_path+univ_name)
	else:
		print('creating universe')
		crsp = get_crsp_m()
		crsp = crsp.loc[crsp['shrcd'].isin([10,11])]
		crsp = crsp.loc[crsp['exchcd'].isin([1,2,3,4])]
		crsp['date'] = pd.to_datetime(crsp.date)
		crsp['date'] = crsp['date'] - MonthBegin(1)
		crsp = crsp.sort_values(by=['permno','date'])
		for i in range(1,13):
			crsp['retadj'+str(i)] = crsp.groupby('permno')['retadj'].shift(-i)
		crsp['me'] = crsp['shrout']*crsp['prc'].abs()/1000
		crsp['bidask'] = (crsp['ask']-crsp['bid'])/crsp[['ask','bid']].mean(1)
		crsp[[x+'_lag' for x in ['me','bidask']]] = crsp.groupby('permno')[['me','bidask']].shift(1)
		crsp['me_lag'] = np.where(crsp['me_lag'].isnull(),crsp['me']/(1+crsp['retx']),crsp['me_lag'])
		crsp['date_lag'] = crsp.groupby(['permno'])['date'].shift(1)
		crsp['date_lag'] = np.where(crsp['date_lag'].isnull(),crsp['date']-MonthBegin(1),crsp['date_lag'])
		crsp['date_comp'] = crsp['date']-MonthBegin(1)
		crsp['me_lag'] = np.where(crsp['date_comp']==crsp['date_lag'],crsp['me_lag'],np.nan)
		crsp['bidask_lag'] = np.where(crsp['date_comp']==crsp['date_lag'],crsp['bidask_lag'],np.nan)
		crsp['me_rank'] = crsp.groupby('date')['me_lag'].rank(ascending=False)
		crsp = crsp.loc[crsp.me_rank<=n_stocks]
		crsp = crsp.loc[crsp.date>=start_date]
		crsp['bidask_lag_m'] = crsp.groupby('date')['bidask_lag'].median()
		crsp['bidask_lag'] = crsp['bidask_lag'].combine_first(crsp['bidask_lag_m'])
		industry_map = get_industry_sic_map()
		crsp['industry'] = crsp.hsiccd.replace(industry_map)
		crsp['industry'] = crsp['industry'].where(crsp['industry'].isin(list(range(48))),49)
		univ = crsp[['permno','date','me_lag','bidask_lag','industry','retadj']+['retadj'+str(i) for i in range(1,13)]].set_index(['permno','date'])
		if univ_name != '':
			print('saving unverse to path '+univ_path+univ_name)
			univ.to_pickle(univ_path+univ_name)
	return univ

