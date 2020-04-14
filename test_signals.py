import os
import multiprocessing as mp
import pandas as pd
from pandas.tseries.offsets import *
import numpy as np
from utils import *
from pull_data import *
import pandas_datareader as pdr
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import json 

colors = ['darkgrey','indianred','cornflowerblue','mediumseagreen','darkorange','goldenrod']

def output_model(df_univ,df_feat,model,model_name,overwrite=False):
	if (overwrite)|(not os.path.isfile('models/'+model_name+'_model_spec.txt')):
		with open('models/'+model_name+'_model_spec.txt', 'w') as outfile:
			json.dump(model, outfile)
		df_sig = create_signal_df(df_univ,df_feat,model)
		df_sig.to_pickle('models/'+model_name+'_signal_df.pkl')
		sig_list = [x['signal_name'] for x in model]
		df_sig[sig_list] = df_sig[sig_list].multiply(df_sig['retadj'],0)
		df_ret = df_sig.groupby('date')[sig_list].sum()
		df_ret.to_pickle('models/'+model_name+'_ret_df.pkl')
	else:
		print('model '+model_name+' already output; import with import_model')

def import_model(model_name,ret_df=True,signal_df=True):
	if ret_df:
		df_ret = pd.read_pickle('models/'+model_name+'_ret_df.pkl')
		if signal_df:
			df_sig = pd.read_pickle('models/'+model_name+'_signal_df.pkl')
			return df_ret,df_sig
		else:
			return df_ret
	else:
		df_sig = pd.read_pickle('models/'+model_name+'_signal_df.pkl')
		return df_sig

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
	if average:
		temp[feat_list] = temp.groupby('date')[feat_list].apply(lambda x: 2*(x.rank()-x.rank().mean())/(x.rank()-x.rank().mean()).abs().sum())
	for i,sig in enumerate(sig_list):
		temp[sig] = temp[feat_nlist[i]].dot(w_nlist[i])
	if industry_adjust:
		sig_ia = [sig_list[i] for i,x in enumerate(ia_list) if x==True]
		df_ia = multi_groupby(temp[['industry']+sig_ia],['date','industry'],normalize_chars)
		temp[sig_ia] = df_ia[sig_ia]
	temp[sig_list] = temp.groupby('date')[sig_list].apply(lambda x: 2*(x.rank()-x.rank().mean())/(x.rank()-x.rank().mean()).abs().sum())
	temp = temp[univ_cols+sig_list]
	return temp

def normalize_chars(df_sub):
	#helper function that normalizes characteristics in cross-section and within industry; will be fed into the multi-groupby function
	to_norm = [x for x in df_sub.columns if x not in ['permno','date','industry']]
	df_sub[to_norm] = (df_sub[to_norm].rank()-df_sub[to_norm].rank().mean())/df_sub[to_norm].rank().max()
	return df_sub

def backtest_signal(df,signal,
					start_date=None,
					end_date=None,
					ic_analysis=True,
					return_analysis=True,
					model_analysis=True,
					comparative_analysis=False,
					interaction_analysis=False,
					quantiles=10,
					signals_to_compare=[],
					signal_to_interact=None,
					model_name='ff'):

	if (start_date==None)&(end_date==None):
	    start_date='1900-01-01'
	    end_date=pd.Timestamp.today()
	elif (start_date==None):
	    start_date='1900-01-01'
	elif (end_date==None):
	    end_date=pd.Timestamp.today()
	temp = df.reset_index()
	temp  = temp.loc[(temp['date']>=start_date)&(temp['date']<end_date)]
	if ic_analysis:
		plot_IC(temp,signal)
	if return_analysis:
		plot_returns(temp,signal,quantiles)
	if model_analysis:
		plot_model(temp,signal,model_name)
	if comparative_analysis:
		compare_signals(temp,[signal]+signals_to_compare,model_name)
	if interaction_analysis:
		plot_interaction(temp,signal,signal_to_interact,model_name,quantiles)

def plot_IC(df,f):
	min_date = df['date'].min().strftime('%Y-%m')
	max_date = df['date'].max().strftime('%Y-%m')
	rec = pdr.fred.FredReader('USREC', start=min_date, end=max_date).read()
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

def plot_returns(df,f,quantiles=10):
	min_date = df['date'].min().strftime('%Y-%m')
	max_date = df['date'].max().strftime('%Y-%m')
	rec = pdr.fred.FredReader('USREC', start=min_date, end=max_date).read()
	df[f+'_quantile'] = df.groupby('date')[f].transform(lambda x: pd.qcut(x,quantiles,labels=False,retbins=False,duplicates='drop'))
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

	df['wr_rank'] = df[f]*df['retadj']
	df_ret = pd.DataFrame()
	df_ret[f+' rank-weighted'] = df.groupby('date')['wr_rank'].sum()
	df_ret[str(qmax+1)+'-'+str(1)+' EW'] = qrets[str(qmax+1)+'-'+str(1)]
	df_ret[str(qmax+1)+'-'+str(1)+' VW'] = qrets2[str(qmax+1)+'-'+str(1)]
	stats = pd.DataFrame()
	stats['Mean'] = df_ret.mean()*12
	stats['Vol'] = df_ret.std()*np.sqrt(12)
	stats['Sharpe'] = stats['Mean']/stats['Vol']
	cr = (1+df_ret).cumprod()

	fig = plt.figure(figsize=(12,25))
	gs = fig.add_gridspec(5, 2)
	ax0 = fig.add_subplot(gs[0,:])
	ax1 = fig.add_subplot(gs[1,:])
	ax20 = fig.add_subplot(gs[2,0])
	ax21 = fig.add_subplot(gs[2,1])
	ax3 = fig.add_subplot(gs[3,:])
	cr.iloc[:,:1].plot(logy=True,ax=ax0,color='k',linestyle='--',title='Return analysis: '+min_date+' through '+max_date+'\n'+f+' rank weighted long-short portfolio')
	cr.iloc[:,1:].plot(logy=True,ax=ax3,color=colors,title='Top-minus-bottom quantile portfolios')
	df_q.plot(kind='bar',color=colors,title='Annualized mean return by signal quantile',edgecolor='k',ax=ax1)
	cmap = sns.diverging_palette(240, 10, as_cmap=True)
	(1+qrets[[x for x in range(1,(qmax+2))]]).cumprod().plot(logy=True,colormap=cmap,ax=ax20,title='EW')
	(1+qrets2[[x for x in range(1,(qmax+2))]]).cumprod().plot(logy=True,colormap=cmap,ax=ax21,title='VW',legend=False)
	for i,col in enumerate(df_ret.columns):
		if i<1:
			ax0.annotate('Sharpe: '+str(round(stats['Sharpe'].loc[col],2)),(cr.index.values[-1],cr[col].values[-1]))
		else:
			ax3.annotate('Sharpe: '+str(round(stats['Sharpe'].loc[col],2)),(cr.index.values[-1],cr[col].values[-1]))
	ax0.set_xlabel('')
	ax1.axvline(9.5,color='k',linestyle='--')
	ax20.set_xlabel('')
	ax21.set_xlabel('')
	ax3.set_xlabel('')
	ylim = ax0.get_ylim()
	ax0.fill_between(cr.index.values, ylim[0], ylim[1], rec.values[:,0], facecolor='k', alpha=0.1)
	ylim = ax20.get_ylim()
	ax20.fill_between(cr.index.values, ylim[0], ylim[1], rec.values[:,0], facecolor='k', alpha=0.1)
	ylim = ax21.get_ylim()
	ax21.fill_between(cr.index.values, ylim[0], ylim[1], rec.values[:,0], facecolor='k', alpha=0.1)
	ylim = ax3.get_ylim()
	ax3.fill_between(cr.index.values, ylim[0], ylim[1], rec.values[:,0], facecolor='k', alpha=0.1)
	plt.show()

def plot_model(df,f,model_name='ff'):
	min_date = df['date'].min().strftime('%Y-%m')
	max_date = df['date'].max().strftime('%Y-%m')
	rec = pdr.fred.FredReader('USREC', start=min_date, end=max_date).read()

	rets = pd.DataFrame()
	df[f+' rank-weighted'] = df[f].multiply(df['retadj'],0)
	rets[f] = df.groupby('date')[f+' rank-weighted'].sum()
	if model_name=='ff':
		ff = pdr.famafrench.FamaFrenchReader('F-F_Research_Data_Factors',start='1960-01-01').read()[0]/100
		ff['UMD'] = pdr.famafrench.FamaFrenchReader('F-F_Momentum_Factor',start='1960-01-01').read()[0].iloc[:,0]/100
		ff.index = pd.to_datetime(ff.index.astype(str),format='%Y-%m')
		model_vars = ['Mkt-RF','HML','SMB','UMD']
		rets[model_vars] = ff[model_vars]
	else:
		model_rets = pd.read_pickle('models/'+model_name+'_ret_df.pkl')
		model_vars = list(model_rets.columns)
		rets[model_vars] = model_rets[model_vars]

	cr = (1+rets).cumprod()
	rets['Const.'] = 1
	params = pd.DataFrame(index=model_vars+['Const.'])
	tvalues = pd.DataFrame(index=model_vars+['Const.'])
	alpha_ts = pd.DataFrame()
	res = sm.OLS(rets[f],rets[model_vars+['Const.']]).fit()
	params[f]=res.params
	tvalues[f]=res.tvalues
	alpha_ts[f]=rets[f]-rets[model_vars].dot(res.params[:-1])
	cr_alpha = (1+alpha_ts).cumprod()

	fig = plt.figure(figsize=(12,15))
	gs = fig.add_gridspec(3,6)
	ax0 = fig.add_subplot(gs[0,:])
	ax1 = fig.add_subplot(gs[1,:4])
	ax2 = fig.add_subplot(gs[1,4:])
	ax3 = fig.add_subplot(gs[2,:])

	cr.iloc[:,0].plot(ax=ax0,color='k',linestyle='--')
	cr.iloc[:,1:].plot(ax=ax0,color=colors)
	ax0.legend(cr.columns)
	ax0.set_yscale('log')
	ax0.set_title('Model analysis: '+min_date+' through '+max_date+'\n'+f+' signal vs. '+model_name+' model factors')
	ylim = ax0.get_ylim()
	ax0.fill_between(cr.index.values, ylim[0], ylim[1], rec.values[:,0], facecolor='k', alpha=0.1)
	ax0.set_xlabel('')

	params.iloc[:-1,:].plot(kind='bar',ax=ax1,color=colors[0],legend=False,edgecolor='k')
	for i,rect in enumerate(ax1.patches):
		t = res.tvalues[i]
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
			xytext=(0, m*10),
			textcoords="offset points",
			ha='center', va='bottom')
	ax1.set_title('Model betas')
	ax1.set_xticklabels(ax1.get_xticklabels(),rotation=20)
    
	(params.iloc[-1,:]*10000*12).plot(kind='bar',ax=ax2,color=colors[0],legend=False,edgecolor='k')
	for i,rect in enumerate(ax2.patches):
		t = res.tvalues[-1]
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
			xytext=(0, m*10),
			textcoords="offset points",
			ha='center', va='bottom')
	ax2.yaxis.set_label_position("right")
	ax2.yaxis.tick_right()
	ax2.set_xticks([])
	ax2.set_title('Model alpha (annualized bps)')
	
	cr_alpha.plot(ax=ax3,color='k',linestyle='--',legend=False,logy=True)
	ax3.set_xlabel('')
	ax3.set_title('Ex-post alpha')
	ylim = ax3.get_ylim()
	ax3.fill_between(cr.index.values, ylim[0], ylim[1], rec.values[:,0], facecolor='k', alpha=0.1)

def compare_signals(df,fs,model_name='ff'):
	min_date = df['date'].min().strftime('%Y-%m')
	max_date = df['date'].max().strftime('%Y-%m')
	rec = pdr.fred.FredReader('USREC', start=min_date, end=max_date).read()
	df['average'] = df[fs].mean(1)
	df['average'] = df.groupby('date')['average'].apply(lambda x: 2*(x.rank()-x.rank().mean())/(x.rank()-x.rank().mean()).abs().sum())
	fs2 = fs+['average']
	df[[f+' rank-weighted' for f in fs2]] = df[fs2].multiply(df['retadj'],0)
	rets = df.groupby('date')[[f+' rank-weighted' for f in fs2]].sum()
	rets.columns = [x.split()[0] for x in rets.columns]
	cr = (1+rets).cumprod()
	stats=pd.DataFrame()
	stats['Sharpe']=rets.mean()/rets.std()*np.sqrt(12)

	corr = rets.corr()
	mask = np.triu(np.ones_like(corr, dtype=np.bool))

	if model_name=='ff':
		ff = pdr.famafrench.FamaFrenchReader('F-F_Research_Data_Factors',start='1960-01-01').read()[0]/100
		ff['UMD'] = pdr.famafrench.FamaFrenchReader('F-F_Momentum_Factor',start='1960-01-01').read()[0].iloc[:,0]/100
		ff.index = pd.to_datetime(ff.index.astype(str),format='%Y-%m')
		model_vars = ['Mkt-RF','HML','SMB','UMD']
		rets[model_vars] = ff[model_vars]
	else:
		model_rets = pd.read_pickle('models/'+model_name+'_ret_df.pkl')
		model_vars = list(model_rets.columns)
		rets[model_vars] = model_rets[model_vars]


	rets['Const.'] = 1
	params = pd.DataFrame(index=model_vars+['Const.'])
	tvalues = pd.DataFrame(index=model_vars+['Const.'])
	alpha_ts = pd.DataFrame()
	for f in fs2:
		res = sm.OLS(rets[f],rets[model_vars+['Const.']]).fit()
		params[f]=res.params
		tvalues[f]=res.tvalues
		alpha_ts[f]=rets[f]-rets[model_vars].dot(res.params[:-1])
	cr_alpha = (1+alpha_ts).cumprod()

	fig = plt.figure(figsize=(12,25))
	gs = fig.add_gridspec(4,6)
	ax0 = fig.add_subplot(gs[0,:])
	ax1 = fig.add_subplot(gs[1,1:6])
	ax2 = fig.add_subplot(gs[2,:4])
	ax3 = fig.add_subplot(gs[2,4:])
	ax4 = fig.add_subplot(gs[3,:])  

	cr.plot(ax=ax0,color=colors)
	ax0.legend(cr.columns)
	ax0.set_yscale('log')
	ax0.set_title('Comparative analysis: '+min_date+' through '+max_date+'\n'+'Rank weighted long-short portfolios',)
	ylim = ax0.get_ylim()
	ax0.fill_between(cr.index.values, ylim[0], ylim[1], rec.values[:,0], facecolor='k', alpha=0.1)
	for i,col in enumerate(fs2):
		ax0.annotate('Sharpe: '+str(round(stats['Sharpe'].loc[col],2)),(cr.index.values[-1],cr[col].values[-1]))
	ax0.set_xlabel('')

	cmap = sns.diverging_palette(240, 10, as_cmap=True)
	g = sns.heatmap(corr,mask=mask,annot=True,vmin=-1,vmax=1,square=False,cmap=cmap,linewidths=.5,ax=ax1)
	g.set_xticklabels(g.get_xticklabels(),rotation=10)
	ax1.set_title('Correlation matrix')

	params.iloc[:-1,:].plot(kind='bar',ax=ax2,edgecolor='k',width=.7,title='Model betas',color=colors)
	tvalues_shape = tvalues.iloc[:-1,:].T.stack().values
	for i,rect in enumerate(ax2.patches):
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
		ax2.annotate(l,
			xy=(rect.get_x()+rect.get_width()/2,rect.get_height()),
			xytext=(0, m*15),
			textcoords="offset points",
			ha='center', va='bottom')
	ax2.set_xticklabels(ax2.get_xticklabels(),rotation=20)

	(params.iloc[-1,:]*12*10000).plot(kind='bar',ax=ax3,edgecolor='k',title='Model alpha (annualized bps)',color=colors)
	tvalues_shape2 = tvalues.iloc[-1,:].T.values
	for i,rect in enumerate(ax3.patches):
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
		ax3.annotate(l,
			xy=(rect.get_x()+rect.get_width()/2,rect.get_height()),
			xytext=(0, m*15),
			textcoords="offset points",
			ha='center', va='bottom')
	ax3.yaxis.set_label_position("right")
	ax3.yaxis.tick_right()
	ax3.set_xticks([])

	cr_alpha = (1+alpha_ts).cumprod()
	cr_alpha.plot(ax=ax4,color=colors,title='Ex-post alpha',logy=True)
	#cr_alpha.iloc[:,-1].plot(ax=ax4,color='k',logy=True,linestyle='--')
	ax4.legend(cr_alpha.columns)
	ax4.set_xlabel('')
	ylim = ax4.get_ylim()
	ax4.fill_between(cr.index.values, ylim[0], ylim[1], rec.values[:,0], facecolor='k', alpha=0.1)

	plt.show()
    
def get_universe(n_stocks=3000,start_date='1960-01-01',univ_path='',univ_file='univ_3000.pkl'):
	if os.path.isfile(univ_path+univ_file):
		univ = pd.read_pickle(univ_path+univ_file)
	else:
		crsp = get_crsp_m()
		crsp = crsp.loc[crsp['shrcd'].isin([10,11])]
		crsp = crsp.loc[crsp['exchcd'].isin([1,2,3,4])]
		crsp['date'] = pd.to_datetime(crsp.date)
		crsp['date'] = crsp['date'] - MonthBegin(1)
		crsp = crsp.sort_values(by=['permno','date'])
		for i in range(1,13):
			crsp['retadj'+str(i)] = crsp.groupby('permno')['retadj'].shift(-i)
		crsp['me'] = crsp['shrout']*crsp['prc'].abs()/1000
		crsp['me_lag'] = crsp.groupby('permno')['me'].shift(1)
		crsp['me_lag'] = np.where(crsp['me_lag'].isnull(),crsp['me']/(1+crsp['retx']),crsp['me_lag'])
		crsp['date_lag'] = crsp.groupby(['permno'])['date'].shift(1)
		crsp['date_lag'] = np.where(crsp['date_lag'].isnull(),crsp['date']-MonthBegin(1),crsp['date_lag'])
		crsp['date_comp'] = crsp['date']-MonthBegin(1)
		crsp['me_lag'] = np.where(crsp['date_comp']==crsp['date_lag'],crsp['me_lag'],np.nan)
		crsp['me_rank'] = crsp.groupby('date')['me_lag'].rank(ascending=False)
		crsp = crsp.loc[crsp.me_rank<=n_stocks]
		crsp = crsp.loc[crsp.date>=start_date]
		industry_map = get_industry_sic_map()
		crsp['industry'] = crsp.hsiccd.replace(industry_map)
		crsp['industry'] = crsp['industry'].where(crsp['industry'].isin(list(range(48))),49)
		univ = crsp[['permno','date','me_lag','industry','retadj']+['retadj'+str(i) for i in range(1,13)]].set_index(['permno','date'])
		univ.to_pickle(univ_path+univ_file)
	return univ

def ols_alpha_t(series,mod):
	#alpha tvalue given model with constant as last column
	return sm.OLS(series.dropna(),mod.loc[series.dropna().index]).fit().tvalues[-1]

def plot_interaction(df,f1,f2,model_name='ff',quantiles=10):
	min_date = df['date'].min().strftime('%Y-%m')
	max_date = df['date'].max().strftime('%Y-%m')
	rec = pdr.fred.FredReader('USREC', start=min_date, end=max_date).read()
	df[[x+'_quantile' for x in [f1,f2]]] = df.groupby('date')[[f1,f2]].transform(lambda x: pd.qcut(x,quantiles,labels=False,retbins=False,duplicates='drop'))
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
	to_plot[f2+' High, '+f1+' Low'] = rets2[1][q2max]
	to_plot[f2+' Low, '+f1+' High'] = rets2[q2max][1]
	to_plot[f2+' High, '+f1+' High'] = rets2[q2max][q2max]
	mean_rets = rets.mean().unstack()*12
	if model_name=='ff':
		ff = pdr.famafrench.FamaFrenchReader('F-F_Research_Data_Factors',start='1960-01-01').read()[0]/100
		ff['UMD'] = pdr.famafrench.FamaFrenchReader('F-F_Momentum_Factor',start='1960-01-01').read()[0].iloc[:,0]/100
		ff.index = pd.to_datetime(ff.index.astype(str),format='%Y-%m')
		model_vars = ['Mkt-RF','HML','SMB','UMD']
		model_rets = ff[model_vars]
		model_rets['const']=1
	else:
		model_rets = pd.read_pickle('models/'+model_name+'_ret_df.pkl')
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
	(1+to_plot).cumprod().plot(ax=ax2,logy=True,title='Interaction analysis: '+min_date+' through '+max_date+'\n'+f1+' x '+f2,color=colors)
	ylim = ax2.get_ylim()
	ax2.fill_between(to_plot.index.values, ylim[0], ylim[1], rec.values[:,0], facecolor='k', alpha=0.1)
	plt.show()


def get_industry_sic_map():
	#map from SIC codes to Fama-French industries (from Kenneth French's data website)
	sic_ranges = [list(range(100,200))+list(range(200,300))+list(range(700,800))+list(range(910,919))+list(range(2048,2049)),
					   list(range(2000,2047))+list(range(2050,2064))+list(range(2070,2080))+list(range(2090,2093))+list(range(2095,2096))+list(range(2098,2100)),
					   list(range(2064,2069))+list(range(2086,2088))+list(range(2096,2098)),
					   list(range(2080,2081))+list(range(2082,2086)),
					   list(range(2100,2200)),
					   list(range(920,1000))+list(range(3650,3653))+list(range(3732,3733))+list(range(3930,3932))+list(range(3940,3950)),
					   list(range(7800,7834))+list(range(7840,7842))+list(range(7900,7901))+list(range(7910,7912))+list(range(7920,7934))+list(range(7940,7950))+list(range(7980,7981))+list(range(7990,8000)),
					   list(range(2700,2750))+list(range(2770,2772))+list(range(2780,2800)),
					   list(range(2047,2048))+list(range(2391,2393))+list(range(2510,2520))+list(range(2590,2600))+list(range(2840,2845))+list(range(3160,3162))+list(range(3170,3173))+list(range(3190,3200))+list(range(3229,3230))+list(range(3260,3261))+list(range(3262,3264))+list(range(3269,3270))+list(range(3230,3232))+list(range(3630,3640))+list(range(3750,3752))+list(range(3800,3801))+list(range(3860,3862))+list(range(3870,3874))+list(range(3910,3912))+list(range(3914,3916))+list(range(3960,3963))+list(range(3991,3992))+list(range(3995,3996)),
					   list(range(2300,2391))+list(range(3020,3022))+list(range(3100,3112))+list(range(3130,3132))+list(range(3140,3152))+list(range(3963,3966)),
					   list(range(8000,8100)),
					   list(range(3693,3694))+list(range(3840,3852)),
					   list(range(2830,2832))+list(range(2833,2837)),
					   list(range(2800,2830))+list(range(2850,2880))+list(range(2890,2900)),
					   list(range(3031,3032))+list(range(3041,3042))+list(range(3050,3054))+list(range(3060,3100)),
					   list(range(2200,2285))+list(range(2290,2296))+list(range(2297,2300))+list(range(2393,2396))+list(range(2397,2399)),
					   list(range(800,900))+list(range(2400,2440))+list(range(2450,2460))+list(range(2490,2500))+list(range(2660,2662))+list(range(2950,2953))+list(range(2950,2953))+list(range(3200,3201))+list(range(3210,3212))+list(range(3240,3242))+list(range(3250,3260))+list(range(3261,3262))+list(range(3264,3265))+list(range(3270,3276))+list(range(3280,3282))+list(range(3290,3294))+list(range(3295,3300))+list(range(3420,3443))+list(range(3446,3447))+list(range(3448,3453))+list(range(3490,3500))+list(range(3996,3997)),
					   list(range(1500,1512))+list(range(1520,1550))+list(range(1600,1800)),
					   list(range(3300,3301))+list(range(3310,3318))+list(range(3320,3326))+list(range(3330,3342))+list(range(3350,3358))+list(range(3360,3380))+list(range(3390,3400)),
					   list(range(3400,3401))+list(range(3443,3445))+list(range(3460,3480)),
					   list(range(3510,3537))+list(range(3538,3539))+list(range(3540,3570))+list(range(3580,3583))+list(range(3585,3587))+list(range(3589,3600)),
					   list(range(3600,3601))+list(range(3610,3614))+list(range(3620,3622))+list(range(3623,3630))+list(range(3640,3647))+list(range(3648,3650))+list(range(3660,3661))+list(range(3690,3693))+list(range(3699,3700)),
					   list(range(2296,2297))+list(range(2396,2397))+list(range(3010,3012))+list(range(3537,3538))+list(range(3647,3648))+list(range(3694,3695))+list(range(3700,3701))+list(range(3710,3712))+list(range(3713,3717))+list(range(3790,3793))+list(range(3799,3800)),
					   list(range(3720,3722))+list(range(3723,3726))+list(range(3728,3730)),
					   list(range(3730,3732))+list(range(3740,3744)),
					   list(range(3760,3770))+list(range(3795,3796))+list(range(3480,3490)),
					   list(range(1040,1050)),
					   list(range(1000,1040))+list(range(1050,1120))+list(range(1400,1500)),
					   list(range(1200,1300)),
					   list(range(1300,1301))+list(range(1310,1340))+list(range(1370,1383))+list(range(1389,1390))+list(range(2900,2913))+list(range(2990,3000)),
					   list(range(4900,4901))+list(range(4910,4912))+list(range(4920,4926))+list(range(4930,4933))+list(range(4939,4943)),
					   list(range(4800,4801))+list(range(4810,4814))+list(range(4820,4823))+list(range(4830,4842))+list(range(4880,4893))+list(range(4899,4900)),
					   list(range(7020,7022))+list(range(7030,7034))+list(range(7200,7201))+list(range(7210,7213))+list(range(7214,7215))+list(range(7215,7218))+list(range(7219,7222))+list(range(7230,7232))+list(range(7240,7242))+list(range(7250,7252))+list(range(7260,7300))+list(range(7395,7396))+list(range(7500,7501))+list(range(7520,7550))+list(range(7600,7601))+list(range(7620,7621))+list(range(7620,7621))+list(range(7622,7624))+list(range(7629,7632))+list(range(7640,7641))+list(range(7690,7700))+list(range(8100,8500))+list(range(8600,8700))+list(range(8800,8900))+list(range(7510,7516)),
					   list(range(2750,2760))+list(range(3993,3994))+list(range(7218,7219))+list(range(7300,7301))+list(range(7310,7343))+list(range(7349,7354))+list(range(7359,7373))+list(range(7374,7386))+list(range(7389,7395))+list(range(7396,7398))+list(range(7399,7400))+list(range(7519,7420))+list(range(8700,8701))+list(range(8710,8714))+list(range(8720,8722))+list(range(8730,8735))+list(range(8740,8749))+list(range(8900,8912))+list(range(8920,9000))+list(range(4220,4230)),
					   list(range(3570,3580))+list(range(3680,3690))+list(range(3695,3696))+list(range(7373,7374)),
					   list(range(3622,3623))+list(range(3661,3667))+list(range(3669,3680))+list(range(3810,3811))+list(range(3812,3813)),
					   list(range(3811,3812))+list(range(3820,3828))+list(range(3829,3840)),
					   list(range(2520,2550))+list(range(2600,2640))+list(range(2670,2700))+list(range(2760,2762))+list(range(3950,3956)),
					   list(range(2440,2450))+list(range(2640,2660))+list(range(3220,3222))+list(range(3410,3413)),
					   list(range(4000,4014))+list(range(4040,4050))+list(range(4100,4101))+list(range(4110,4122))+list(range(4130,4132))+list(range(4140,4143))+list(range(4150,4152))+list(range(4170,4174))+list(range(4190,4201))+list(range(4210,4220))+list(range(4230,4232))+list(range(4240,4250))+list(range(4400,4701))+list(range(4710,4713))+list(range(4720,4750))+list(range(4780,4781))+list(range(4782,4786))+list(range(4789,4790)),
					   list(range(5000,5001))+list(range(5010,5016))+list(range(5020,5024))+list(range(5030,5061))+list(range(5063,5066))+list(range(5070,5089))+list(range(5080,5089))+list(range(5090,5095))+list(range(5099,5101))+list(range(5110,5114))+list(range(5120,5123))+list(range(5130,5173))+list(range(5180,5183))+list(range(5190,5200)),
					   list(range(5200,5201))+list(range(5210,5232))+list(range(5250,5252))+list(range(5260,5262))+list(range(5270,5272))+list(range(5300,5301))+list(range(5310,5312))+list(range(5320,5321))+list(range(5330,5332))+list(range(5334,5335))+list(range(5340,5350))+list(range(5390,5401))+list(range(5410,5413))+list(range(5420,5470))+list(range(5490,5501))+list(range(5510,5580))+list(range(5590,5701))+list(range(5710,5723))+list(range(5730,5737))+list(range(5750,5800))+list(range(5900,5901))+list(range(5910,5913))+list(range(5920,5933))+list(range(5940,5991))+list(range(5992,5996))+list(range(5999,6000)),
					   list(range(5800,5830))+list(range(5890,5900))+list(range(7000,7001))+list(range(7010,7020))+list(range(7040,7050))+list(range(7213,7214)),
					   list(range(6000,6001))+list(range(6010,6037))+list(range(6040,6063))+list(range(6080,6083))+list(range(6090,6101))+list(range(6110,6114))+list(range(6120,6180))+list(range(6190,6199)),
					   list(range(6300,6301))+list(range(6310,6332))+list(range(6350,6352))+list(range(6360,6362))+list(range(6370,6380))+list(range(6390,6412)),
					   list(range(6500,6501))+list(range(6510,6511))+list(range(6512,6516))+list(range(6517,6533))+list(range(6540,6542))+list(range(6550,6554))+list(range(6590,6600))+list(range(6610,6612)),
					   list(range(6200,6300))+list(range(6700,6701))+list(range(6710,6727))+list(range(6730,6734))+list(range(6740,6780))+list(range(6790,6796))+list(range(6798,6800)),
					   list(range(4950,4962))+list(range(4970,4972))+list(range(4990,4992))]
	industry_map = {}
	for i,sic_range in enumerate(sic_ranges):
		industry_map.update({x:i for x in sic_range})
	return industry_map