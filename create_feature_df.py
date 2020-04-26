import os
import multiprocessing as mp
import pandas as pd
from pandas.tseries.offsets import *
import numpy as np
from utils import *
from pull_data import *

#----------------------------------------------------------
# function to check whether stock features have been 
# calculated, process, and return completed dataframe  
# see calculate_features for functions that calculate from inputs  
#----------------------------------------------------------

def get_features(overwrite=False,feat_path='',feat_file='features.pkl'):
	#calls functions below to calculate features from input data and append to dataframe
	#by default does not overwrite steps that have already been done and logged
	#after features are calculated, they are subset/normalized
	if overwrite:
		log = []
	elif os.path.isfile(feat_path+feat_file[:-4]+'_output_log.txt'):
		with open(feat_path+feat_file[:-4]+'_output_log.txt','r') as f:
			log = f.read().split('\n')
	else:
		log = []
	if 'technical' not in log:
		print('calculating technical features')
		calc_technical(feat_path=feat_path,feat_file=feat_file)
		with open(feat_path+feat_file[:-4]+'_output_log.txt','a') as f:
			f.write('technical\n')
	if 'fundamental' not in log:
		print('calculating fundamental features')
		calc_fundamental(feat_path=feat_path,feat_file=feat_file)
		with open(feat_path+feat_file[:-4]+'_output_log.txt','a') as f:
			f.write('fundamental\n')
	if 'earnret' not in log:
		print('calculating earnret features')
		calc_earnret(feat_path=feat_path,feat_file=feat_file)
		with open(feat_path+feat_file[:-4]+'_output_log.txt','a') as f:
			f.write('earnret\n')
	if 'estimates' not in log:
		print('calculating estimates features')
		calc_estimates(feat_path=feat_path,feat_file=feat_file)
		with open(feat_path+feat_file[:-4]+'_output_log.txt','a') as f:
			f.write('estimates\n')
	if 'meta' not in log:
		print('calculating meta features')
		calc_meta(feat_path=feat_path,feat_file=feat_file)
		with open(feat_path+feat_file[:-4]+'_output_log.txt','a') as f:
			f.write('meta\n')
	out = pd.read_pickle(feat_path+feat_file)
	return out

#----------------------------------------------------------
# functions to calculate stock features from input 
# data, lag appropriately, and append to output file
# see pull_data for functions that import files from WRDS
#----------------------------------------------------------

def output_features(df,cols,feat_path='',feat_file='features.pkl'):
	#append columns (cols; list of column names) from dataframe (df) to output file 
	#note that output file is indexed by ['permno', 'date'], which must exist in df
	try:
		df = df.set_index(['permno','date'])
	except:
		return ValueError('df must contain [permno,date] as columns for indexing')
	if os.path.isfile(feat_path+feat_file):
		#overwrite output dataframe after appending new features
		features = pd.read_pickle(feat_path+feat_file)
		features[cols] = df[cols]
		features.to_pickle(feat_file)
	else:
		#initialize index with CRSP universe
		features = get_crsp_m()[['permno','date']]
		features['date'] -= MonthBegin(1)
		features = features.set_index(['permno','date'])
		features[cols] = df[cols]
		features.to_pickle(feat_path+feat_file)

def calc_meta(feat_path='',feat_file='features.pkl'):
	features = pd.read_pickle(feat_path+feat_file).reset_index()
	date_comp = features['date'] + MonthEnd(1)
	is_within_5 = features['last_filing_date']+DateOffset(days=5) > date_comp
	is_within_10 = features['last_filing_date']+DateOffset(days=10) > date_comp
	features['ret_unexpl_22'] = features['mean_22']*22-features['momentum_earn_3'].fillna(0)*3
	features['ret_unexpl_10'] = np.where(is_within_10,features['mean_10']*10,features['mean_10']*10-features['momentum_earn_3'].fillna(0)*3)
	features['ret_unexpl_5'] = np.where(is_within_5,features['mean_5']*5,features['mean_5']*5-features['momentum_earn_3'].fillna(0)*3)
	cols = ['ret_unexpl_22','ret_unexpl_10','ret_unexpl_5']
	out = multi_groupby(features[['permno','date']+cols],['permno'],groupby_rolling_z)
	output_features(out,cols,feat_path=feat_path,feat_file=feat_file)

def groupby_rolling_z(df_sub):
    cols = [x for x in df_sub.columns if x not in ['permno','date']]
    df_sub[[x+'_z' for x in cols]] = df_sub.rolling(12)[cols].apply(lambda x: (x[-1]-x.mean())/x.std(),raw=True)
    return df_sub

def calc_technical(feat_path='',feat_file='features.pkl'):
	#calculate technical features with crsp daily data
	crsp_d = get_crsp_d()
	crsp_d['date'] = pd.to_datetime(crsp_d['date'])
	crsp_d = crsp_d.loc[crsp_d.date>pd.to_datetime('1950-01-01')]
	crsp_d['mve'] = crsp_d['prc'].abs()*crsp_d['shrout']
	crsp_d['mve_rank'] = crsp_d.groupby('date')['mve'].rank(ascending=False)
	min_rank = crsp_d.groupby('permno')['mve_rank'].min()
	crsp_d = crsp_d.loc[crsp_d.permno.isin(min_rank.index[min_rank<=3500])]
	
	crsp_d.sort_values(by=['permno','date'],inplace=True)
	ind = get_indices_d()
	ind['date'] = pd.to_datetime(ind['date'])
	ind.sort_values(by=['date'],inplace=True)
	for i in [22,252,756]:
		ind['sp_std_'+str(i)] = ind['sprtrn'].rolling(i,min_periods=i).std()
	crsp_d = crsp_d.merge(ind[['date','sprtrn']+['sp_std_'+str(i) for i in [22,252,756]]],how='left',on=['date'])
	out = multi_groupby(crsp_d,['permno'],calc_technical_sub)
	cols = out.columns
	out.reset_index(inplace=True)
	#append to output file
	output_features(out,cols,feat_path=feat_path,feat_file=feat_file)

def calc_technical_sub(df_sub):
	#calculate mean returns over horizons
	for i in [5,10,22,198,252,756]:
		df_sub['mean_'+str(i)] = df_sub['ret'].rolling(i,min_periods=i).mean()
	#calculate volatility, correlation, and beta over horizons
	for i in [22,252,756]:
		df_sub['std_'+str(i)] = df_sub['ret'].rolling(i,min_periods=i).std()
		df_sub['corr_sp_'+str(i)] = df_sub['ret'].rolling(i,min_periods=i).corr(df_sub['sprtrn'])
		df_sub['beta_sp_'+str(i)] = df_sub['corr_sp_'+str(i)]*df_sub['std_'+str(i)]/df_sub['sp_std_'+str(i)]
		df_sub['beta_sp_'+str(i)+'_sq'] = df_sub['beta_sp_'+str(i)]**2
	#calculate idiosyncratic volatility 
	for i in [22,252]:	
		df_sub['e_'+str(i)] = df_sub['ret']-df_sub['beta_sp_'+str(i)]*df_sub['sprtrn']
		df_sub['e_std_'+str(i)] = df_sub['e_'+str(i)].rolling(i,min_periods=i).std()
	#calculate liquidity metrics
	df_sub['volume_5'] = df_sub['vol'].rolling(5,min_periods=5).sum()
	df_sub['stdvolume_22'] = df_sub['vol'].rolling(22,min_periods=22).std()
	df_sub['bidask'] = (df_sub['ask'].abs()-df_sub['bid'].abs())/df_sub['prc'].abs()
	df_sub['bidask_5'] = df_sub['bidask'].rolling(5,min_periods=5).sum()
	df_sub['volume_5_shrout'] = df_sub['volume_5']/df_sub['shrout']
	#change in shares outstanding
	df_sub['shrout_pct'] = df_sub['shrout']/df_sub.groupby(['permno'])['shrout'].shift(22)-1
	#final features
	cols = ['mean_5','mean_10','mean_22','mean_198',
			'mean_252','mean_756','std_22','std_252','volume_5',
			'volume_5_shrout','stdvolume_22','bidask_5','mve',
			'corr_sp_22','corr_sp_252','corr_sp_756','beta_sp_22','beta_sp_252',
			'beta_sp_756','beta_sp_22_sq','beta_sp_252_sq',
			'beta_sp_756_sq','shrout_pct','e_std_22','e_std_252']
	#lag by 2 business days to ensure availability + 1 month to align with t+1 return
	df_sub['date'] = df_sub['date']+BDay(2)+MonthBegin(1)
	return df_sub.groupby(['permno','date'])[cols].last()

def calc_fundamental(feat_path='',feat_file='features.pkl'):
	#calculate fundamental features with compustat quarterly data
	comp = get_compustat_q()
	#merge filing dates onto compustat data
	aod = get_comp_dates()
	aod.sort_values(by=['rdq'],ascending=False,inplace=True)
	aod.drop_duplicates(subset=['gvkey','datadate'],inplace=True)
	comp = comp.merge(aod,how='left',on=['gvkey','datadate'])
	comp = comp.sort_values(by=['datadate','gvkey'])
	#take differences of some fiscal YTD variables
	y_vars = ['capxy','recchy','invchy','apalchy','aqcy']
	comp[[var+'_lag' for var in y_vars]] = comp.groupby('gvkey')[y_vars].shift(1)
	#comp[[var[:-1]+'q' for var in y_vars]] = np.where(comp.fqtr==1,comp[y_vars],comp[y_vars]-comp[[var+'_lag' for var in y_vars]].values)
	comp[[var[:-1]+'q' for var in y_vars]] = comp[y_vars].where(comp.fqtr==1,comp[y_vars]-comp[[var+'_lag' for var in y_vars]].values)
	#calculate book equity
	comp['ps'] = np.where(comp['pstkq'].isnull(),0,comp['pstkq'])
	comp['txditc'] = comp['txditcq'].fillna(0)
	comp['be'] = (comp['seqq']+comp['txditc']-comp['ps']).combine_first(comp['atq']-comp['ltq'])
	comp['be'] = comp['be'].where(comp['be']>0,np.nan)
	#growth in book equity
	comp['be_growth'] = comp.groupby('gvkey')['be'].pct_change(1)
	cols_out = ['be_growth']
	#make assets non-negative
	comp['atq'] = comp['atq'].where(comp['atq']>0,np.nan)
	#asset growth and investment
	comp['asset_growth'] = comp.groupby('gvkey')['atq'].pct_change(4)
	comp['inv'] = comp['aqcq']/comp['atq']
	comp['inv2'] = comp['aqcq']/comp['be']
	cols_out += ['asset_growth','inv','inv2']
	#total debt
	comp['td'] = comp['dlttq']+comp['dlcq'].fillna(0)
	#calculate accruals, sales, profitability, cash flow
	comp['wcap'] = comp['actq']-comp['lctq']-comp['cheq']+comp['dlcq']+comp['txtq']
	comp['wc_d'] = comp.groupby('gvkey')['wcap'].diff(1)
	comp['acc'] = comp['dpq'] - comp['wc_d']
	comp['acc2'] = comp['recchq'] - comp['invchq'] - comp['apalchq']
	comp['sales'] = comp['saleq'].combine_first(comp['revtq'])
	comp['prof'] = comp['sales'] - comp['cogsq'] - comp['xsgaq'].fillna(0) + comp['xrdq'].fillna(0)
	comp['prof2'] = comp['prof'] - comp['recchq'].fillna(0) - comp['invchq'].fillna(0) - comp['apalchq'].fillna(0)
	comp['cf'] = (comp['ibq'] + comp['dpq'].fillna(0) - comp['wc_d'].fillna(0) - comp['capxq'].fillna(0))
	comp['cf2'] = comp['prof'] - comp['capxq'].fillna(0)
	comp['eps'] = comp['epspxq']
	#all cash flow variables
	cf_vars = ['sales','eps','prof','prof2','cf','cf2']
	#Novy-Marx fundamental momentum applied to flow variables
	comp[[var+'_d' for var in cf_vars]] = comp.groupby('gvkey')[cf_vars].diff(4)
	comp[[var+'_d_std' for var in cf_vars]] = comp.groupby('gvkey')[[var+'_d' for var in cf_vars]].rolling(8,min_periods=6).std().reset_index(0,drop=True)
	comp[[var+'_momentum' for var in cf_vars]] = comp[[var+'_d' for var in cf_vars]]/comp[[var+'_d_std' for var in cf_vars]].values
	cols_out += [var+'_momentum' for var in cf_vars]
	#growth and volatility applied to flow variables
	comp[[var+'_l' for var in cf_vars]] = comp.groupby('gvkey')[cf_vars].shift(4)
	comp[[var+'_growth' for var in cf_vars]] = comp.groupby('gvkey')[cf_vars].pct_change(4)
	comp[[var+'_growth' for var in cf_vars]] = comp[[var+'_growth' for var in cf_vars]].where(comp[[var+'_l' for var in cf_vars]].values>0,np.nan)
	comp[[var+'_std' for var in cf_vars]] = comp.groupby('gvkey')[[var+'_growth' for var in cf_vars]].rolling(10,min_periods=6).std().reset_index(0,drop=True)
	cols_out += [var+'_growth' for var in cf_vars]+[var+'_std' for var in cf_vars]
	#profitability, cash flow metrics to assets/book equity
	ratio_vars = cf_vars+['acc','acc2','td']
	comp[[var+'_asset' for var in ratio_vars]] = comp[ratio_vars].divide(comp['atq'],0)
	comp[[var+'_be' for var in ratio_vars]] = comp[ratio_vars].divide(comp['be'],0)
	cols_out += [var+'_asset' for var in ratio_vars]+[var+'_be' for var in ratio_vars]
	#change in leverage
	comp['td_l'] = comp.groupby('gvkey')['td'].shift(1)
	comp['td_d'] = comp.groupby('gvkey')['td'].diff(1)
	comp['td_d_std'] = comp.groupby('gvkey')['td_d'].rolling(8,min_periods=6).std().reset_index(0,drop=True)
	comp['chdebt_asset'] = comp['td_d']/comp['atq']
	comp['chdebt_z'] = comp['td_d']/comp['td_d_std']
	comp['chdebt_growth'] = comp['td_d']/comp['td_l']
	comp['chdebt_growth'] = comp['chdebt_growth'].where(comp['td_l']>0,np.nan)
	cols_out += ['chdebt_asset','chdebt_z','chdebt_growth']
	comp[cols_out] = comp[cols_out].replace({np.inf:np.nan,-np.inf:np.nan})
	#import crsp
	crsp = get_crsp_m()
	crsp['date'] = pd.to_datetime(crsp.date)
	crsp['date'] = crsp['date']-MonthBegin(1)
	crsp = crsp.sort_values(by=['date','permno'])
	#calculate lagged market equity
	crsp['me'] = crsp['shrout']*crsp['prc'].abs()/1000
	crsp['me_lag'] = crsp.groupby('permno')['me'].shift(1)
	crsp['me_lag'] = np.where(crsp['me_lag'].isnull(),crsp['me']/(1+crsp['retx']),crsp['me_lag'])
	crsp['date_lag'] = crsp.groupby(['permno'])['date'].shift(1)
	crsp['date_lag'] = np.where(crsp['date_lag'].isnull(),crsp['date']-MonthBegin(1),crsp['date_lag'])
	crsp['date_comp'] = crsp['date']-MonthBegin(1)
	crsp['me_lag'] = np.where(crsp['date_comp']==crsp['date_lag'],crsp['me_lag'],np.nan)
	#get linking table to join crsp and compustat
	link = get_crspcomp_link()
	crsp = crsp.merge(link[['permno','gvkey','linkdt','linkenddt']],how='left',on='permno')
	crsp['gvkey'] = crsp.gvkey.where(~((crsp.date<crsp.linkdt)|(crsp.date>crsp.linkenddt)),np.nan)
	crsp = crsp.dropna(subset=['gvkey'])
	crsp = crsp.drop_duplicates(subset=['permno','date'])
	#join crsp and compustat after lagging compustat by 2 business days after filing date + 1 month to align with t+1 return
	#until now compustat was implicitly indexed by the period of filing, now switch to filing date 
	crsp['jdate'] = pd.to_datetime(crsp.date)
	comp['jdate'] = pd.to_datetime(comp.rdq)+BDay(2)+MonthBegin(1)
	comp = comp.sort_values(by=['gvkey','jdate','datadate'],ascending=[True,True,False]).dropna(subset=['jdate']).drop_duplicates(subset=['gvkey','jdate'])
	#merge on exact date
	to_merge = ['be','td','td_d','sales','prof','prof2','cf','cf2','rdq','datadate']
	crsp = crsp.merge(comp[['gvkey','jdate']+cols_out+to_merge],how='left',on=['gvkey','jdate'])
	#forward fill up to 3 months
	crsp[to_merge] = crsp.groupby('permno')[to_merge].fillna(method='ffill',limit=3)
	#calculate enterprise value, value signals, leverage on me/ev
	crsp['ev'] = crsp['me_lag']+crsp['td'].fillna(0)
	ratio_vars2 = ['td_d','be','sales','prof','prof2','cf','cf2']
	crsp[[var+'_me' for var in ratio_vars2]] = crsp[ratio_vars2].divide(crsp['me_lag'],0)
	crsp[[var+'_ev' for var in ratio_vars2]] = crsp[ratio_vars2].divide(crsp['ev'],0)
	cols = cols_out + [var+'_me' for var in ratio_vars2]+[var+'_ev' for var in ratio_vars2]
	crsp = crsp.rename(columns={'rdq':'last_filing_date','datadate':'last_filing_period'})
	cols += ['last_filing_date','last_filing_period']
	output_features(crsp,cols,feat_path=feat_path,feat_file=feat_file)

def calc_earnret(feat_path='',feat_file='features.pkl'):
	#calculate earnings momentum using returns around quarterly filing dates (Novy-Marx)
	#get crsp daily file, merge with compustat linking table and subset to companies in compustat
	crsp = get_crsp_d()
	crsp['date'] = pd.to_datetime(crsp.date)
	crsp = crsp.sort_values(by=['date','permno'])
	link = get_crspcomp_link()
	crsp = crsp.merge(link[['permno','gvkey','linkdt','linkenddt']],how='left',on='permno')
	crsp['gvkey'] = crsp.gvkey.where(~((crsp.date<crsp.linkdt)|(crsp.date>crsp.linkenddt)),np.nan)
	crsp = crsp.dropna(subset=['gvkey'])
	#merge on S&P 500 index returns
	ind = get_indices_d()
	ind['date'] = pd.to_datetime(ind['date'])
	crsp = crsp.merge(ind[['date','sprtrn']],how='left',on='date')
	#estimate idiosyncratic return as r-r_m
	crsp['e'] = crsp['ret']-crsp['sprtrn']
	#calculate 3 and 5 day centered mean returns (this seems quicker than .rolling())
	for i in range(-2,3):
		crsp['ret_'+str(i)] = crsp.groupby('permno')['ret'].shift(-i)
		crsp['e_'+str(i)] = crsp.groupby('permno')['e'].shift(-i)
	crsp['momentum_earn_3'] = crsp[['ret_'+str(i) for i in range(-1,2)]].mean(axis=1)
	crsp['momentum_earn_5'] = crsp[['ret_'+str(i) for i in range(-2,3)]].mean(axis=1)
	crsp['momentum_idio_3'] = crsp[['e_'+str(i) for i in range(-1,2)]].mean(axis=1)
	crsp['momentum_idio_5'] = crsp[['e_'+str(i) for i in range(-2,3)]].mean(axis=1)
	#features to output
	cols = ['momentum_earn_3','momentum_earn_5','momentum_idio_3','momentum_idio_5']
	#get earnings dates as quarterly filing dates and merge 3/5 day returns/idiosyncratic returns on
	aod = get_comp_dates()
	aod['jdate'] = pd.to_datetime(aod['rdq'])
	aod = aod.dropna(subset=['jdate','gvkey'])
	aod = aod.sort_values(by=['jdate','gvkey'])
	crsp['jdate'] = pd.to_datetime(crsp.date)
	out = pd.merge_asof(aod,crsp[['gvkey','jdate','permno','date']+cols],on='jdate',by=['gvkey'])
	out = out.groupby(['permno','rdq'])[['date']+cols].last().reset_index()
	#lag features by 5 business days + 1 month to align with t+1 return
	out['date'] = out['date']+BDay(5)+MonthBegin(1)
	out = out[['date','permno']+cols].dropna().drop_duplicates(subset=['permno','date'])
	output_features(out,cols,feat_path=feat_path,feat_file=feat_file)

def calc_estimates(feat_path='',feat_file='features.pkl'):
	#calculate signals from earnings estimate date (IBES), including earnings surprises, fwd P/E, change in analyst coverage, fwd EPS growth
	ibes = get_ibes_summary()
	#keep estimates for all fwd financial periods which will be used later to calculate fwd P/E ratios
	all_fpi = ibes.loc[ibes.fpi.isin(['0','1','2','3','4','6','7','8','9'])]
	#lag by a month to align with t+1 return
	all_fpi['date'] = pd.to_datetime(all_fpi['statpers']) + DateOffset(months=1)
	#reshape to create columns for each fwd financial period, indexed by company/date
	all_fpi = all_fpi[['ticker','date','fpi','meanest']].pivot_table(index=['ticker','date'],columns=['fpi'],values='meanest',aggfunc=np.sum)
	all_fpi = all_fpi.sort_values(by=['date','ticker'])
	#rename columns 
	replace_dict = dict(zip([str(i) for i in range(5)]+[str(i) for i in range(6,10)],['ltg']+['ann'+str(i) for i in range(1,5)]+['qtr'+str(i) for i in range(1,5)]))
	all_fpi.columns = [replace_dict[x] for x in all_fpi.columns]
	#subset ibes data to 1 qtr fwd EPS estimate and select the most recent estimate before the earnings date
	ibes = ibes.loc[(ibes.measure=='EPS')&(ibes.fpi=='6')]
	ibes['fpedats'] = pd.to_datetime(ibes.fpedats)
	ibes = ibes.sort_values(by=['statpers','ticker'])
	ibes = ibes.groupby(['fpedats','ticker'])[['statpers','meanest','medest','stdev','numest']].last().reset_index()
	#merge on the actual earnings release figures
	actuals = get_ibes_actual()
	actuals = actuals.rename(columns={'value':'actual'})
	actuals = actuals.loc[(actuals.measure=='EPS')&(actuals.pdicity=='QTR')]
	actuals['fpedats'] = pd.to_datetime(actuals.pends) 
	ibes = ibes.merge(actuals[['fpedats','ticker','actual','anndats']],how='left',on=['fpedats','ticker'])
	#set the date index to the earnings releast date and lag by 2 business days + 1 month to align with t+1 return
	ibes['anndats'] = pd.to_datetime(ibes['anndats'])
	ibes['date'] = ibes['anndats'] + BDay(2) + MonthBegin(1)
	ibes = ibes.loc[pd.notnull(ibes.anndats)]
	ibes = ibes.sort_values(by=['date','ticker']).drop_duplicates(subset=['ticker','date'])
	#calculate standardized unexpected earnings (SUE) and earnings surprise z-score
	ibes['sue'] = (ibes['actual']-ibes['meanest'])/ibes['stdev']
	ibes['sue'] = ibes['sue'].where(ibes.stdev!=0,np.nan)
	ibes['surprise'] = ibes['actual']-ibes['meanest']
	ibes['surprise_z'] = ibes.groupby('ticker')['surprise'].rolling(8,min_periods=6).apply(lambda x: (x[-1]-x.mean())/x.std(),raw=True).reset_index(0,drop=True)
	#calculate change in analyst coverage
	ibes['numest_lag'] = ibes.groupby('ticker')['numest'].shift(1)
	ibes['chanalyst'] = ibes['numest']-ibes['numest_lag'] 
	ibes['pchanalyst'] = ibes['numest']/ibes['numest_lag']-1 
	#features to output
	cols = ['sue','surprise_z','chanalyst','pchanalyst']
	#get crsp file,  merge with IBES linking table, and merge features
	crsp = get_crsp_m()
	crsp = crsp.sort_values(by=['date','permno'])
	link = get_crspibes_link()
	crsp = crsp.merge(link[['permno','ticker','sdate','edate']],how='left',on='permno')
	crsp['date'] = pd.to_datetime(crsp.date)
	crsp['ticker'] = crsp.ticker.where(~((crsp.date<crsp.sdate)|(crsp.date>crsp.edate)),np.nan)
	crsp['date'] = pd.to_datetime(crsp.date)-MonthBegin(1)
	crsp = crsp.dropna(subset=['ticker']).drop_duplicates(subset=['permno','date'])
	crsp = crsp.merge(ibes[['ticker','date']+cols],how='left',on=['ticker','date'])
	#merge all fwd earning period estimates onto crsp data
	crsp = pd.merge_asof(crsp,all_fpi,on='date',by=['ticker'])
	#fwd earnings yield at various horizons
	crsp['prc_lag'] = crsp.groupby('permno')['prc'].shift(1)
	crsp['prc_lag'] = crsp.prc_lag.abs()
	crsp['pe0'] = crsp['ann1']/crsp['prc_lag']
	crsp['pe1'] = crsp['ann2']/crsp['prc_lag']
	crsp['pe2'] = crsp['ann3']/crsp['prc_lag']
	crsp['pe3'] = crsp['ann4']/crsp['prc_lag']
	crsp['pe4'] = crsp['qtr1']/crsp['prc_lag']
	crsp['pe5'] = crsp[['qtr1','qtr2','qtr3','qtr4']].sum(axis=1)/crsp['prc_lag']
	#add to list of features to output
	cols += ['pe'+str(x) for x in range(6)]+['ltg']
	output_features(crsp,cols,feat_path=feat_path,feat_file=feat_file)

if __name__ == '__main__':
	df = get_features(overwrite=True)