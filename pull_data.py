import os
import pandas as pd
from pandas.tseries.offsets import *
import pandas_datareader as pdr
import numpy as np
import wrds
from utils import expanding_garch

#----------------------------------------------------------
# functions to retrieve input data from WRDS (login needed) 
#----------------------------------------------------------

def get_crsp_m(input_path='input/'):
	#download and process monthly CRSP file from WRDS api and save as pandas df
	if os.path.isfile(input_path+'crsp/crsp_m.pkl'):
		crsp_m = pd.read_pickle(input_path+'crsp/crsp_m.pkl')
	else:
		db = wrds.Connection()
		#CRSP file with share/exchange data joined on
		crsp_m = db.raw_sql('''
		                  select a.permno, a.permco, a.cusip, a.date, 
	                      	b.shrcd, b.exchcd, a.ret, a.retx, a.shrout, 
	                      	a.prc, a.bid, a.ask, a.vol, a.hsiccd, a.cfacpr, a.cfacshr
	                      from crspq.msf as a
	                      left join crspq.msenames as b
	                      on a.permno=b.permno
	                      	and b.namedt<=a.date
	                      	and a.date<=b.nameendt
	                      ''') 
		#de-listing returns to be joined to crsp file
		dlret = db.raw_sql('''
		                   select permno, dlret, dlstdt 
		                   from crspq.msedelist
		                   ''')
		#merge de-listing returns and calcualte adjusted return 
		crsp_m['jdate'] = crsp_m['date']+MonthEnd(0)
		crsp_m = crsp_m.set_index(['permno','jdate'])
		dlret['jdate'] = dlret['dlstdt']+MonthEnd(0)
		dlret = dlret.set_index(['permno','jdate'])
		crsp_m['dlret'] = dlret.dlret
		crsp_m.reset_index(inplace=True)
		crsp_m = crsp_m.loc[pd.notnull(crsp_m.ret)|pd.notnull(crsp_m.dlret)]
		crsp_m['dlret'] = crsp_m['dlret'].fillna(0)
		crsp_m['ret'] = crsp_m['ret'].fillna(0)
		crsp_m['retadj'] = (1+crsp_m['ret'])*(1+crsp_m['dlret'])-1
		#calculate market equity
		crsp_m['me'] = crsp_m['prc'].abs()*crsp_m['shrout']
		crsp_m.to_pickle(input_path+'crsp/crsp_m.pkl')
		db.close()
	return crsp_m

def get_crsp_d(input_path='input/'):
	#download daily CRSP file from WRDS api and save as pandas df
	if os.path.isfile(input_path+'crsp/crsp_d.pkl'):
		crsp_d = pd.read_pickle(input_path+'crsp/crsp_d.pkl')
	else:
		db = wrds.Connection()
		crsp_d = db.raw_sql('''
		                    select a.permno, a.permco, a.cusip, a.date, 
		                    	b.shrcd, b.exchcd, a.ret, a.retx, a.shrout, 
		                      	a.prc, a.bid, a.ask, a.vol
		                    from crspq.dsf as a
		                    left join crspq.msenames as b
		                    on a.permno=b.permno
		                     	and b.namedt<=a.date
		                      	and a.date<=b.nameendt
		                    ''') 
		crsp_d.to_pickle(input_path+'crsp/crsp_d.pkl')
		db.close()
	return crsp_d

def get_indices_m(input_path='input/'):
	#download monthly index data from CRSP with WRDS api and save as pandas df
	if os.path.isfile(input_path+'crsp/indices_m.pkl'):
		indices_m = pd.read_pickle(input_path+'crsp/indices_m.pkl')
	else:
		db = wrds.Connection()
		indices_m = db.raw_sql('''
		                       select *
		                       from crsp.msi
		                       ''') 
		indices_m.to_pickle(input_path+'crsp/indices_m.pkl')
		db.close()
	return indices_m

def get_indices_d(input_path='input/'):
	#download daily index data from CRSP with WRDS api and save as pandas df
	if os.path.isfile(input_path+'crsp/indices_d.pkl'):
		indices_d = pd.read_pickle(input_path+'crsp/indices_d.pkl')
	else:
		db = wrds.Connection()
		indices_d = db.raw_sql('''
		                      select *
		                      from crsp.dsi
		                   	   ''') 
		indices_d.to_pickle(input_path+'crsp/indices_d.pkl')
		db.close()
	return indices_d

def get_crsp_divs(input_path='input/'):
	#download CRSP dividend data from WRDS api and save as pandas df
	if os.path.isfile(input_path+'crsp/crsp_divs.pkl'):
		crsp_divs = pd.read_pickle(input_path+'crsp/crsp_divs.pkl')
	else:
		db = wrds.Connection()
		crsp_divs = db.raw_sql('''
		                      select a.permno, a.divamt, a.dclrdt, a.exdt, a.paydt
		                      from crsp.msedist as a
			                   ''') 
		crsp_divs.to_pickle(input_path+'crsp/crsp_divs.pkl')
		db.close()
	return crsp_divs

def get_crspcomp_link(input_path='input/'):
	#download CRSP/Compustat linking table from WRDS api and save as pandas df
	if os.path.isfile(input_path+'linking/crspcomp.pkl'):
		link = pd.read_pickle(input_path+'linking/crspcomp.pkl')
	else:
		db = wrds.Connection()
		link = db.raw_sql('''
		                      select gvkey,lpermno,linkdt,linkenddt
		                      from crspq.ccmxpf_lnkhist 
		                      ''') 
		link.to_pickle(input_path+'linking/crspcomp.pkl')
		db.close()
	#make sure there's a 1-1 mapping
	link = link.rename(columns={'lpermno':'permno'}).drop_duplicates()
	link['linkdt'] = pd.to_datetime(link.linkdt)
	link['linkenddt'] = pd.to_datetime(link.linkenddt)
	link['linkenddt'] = link.linkenddt.where(pd.notnull(link.linkenddt),pd.to_datetime('today'))
	link = link.groupby(['permno','gvkey']).agg({'linkdt':'min','linkenddt':'max'}).reset_index()
	return link

def get_compustat_q(input_path='input/'):
	#download quarterly Compustat file from WRDS api and save as pandas df
	if os.path.isfile(input_path+'compustat/compustat_q.pkl'):
		compustat_q = pd.read_pickle(input_path+'compustat/compustat_q.pkl')
	else:
		db = wrds.Connection()
		compustat_q = db.raw_sql('''
			                      select gvkey,cusip,datadate,fqtr,atq,actq,ltq,cheq,teqq,seqq,lseq,lctq,dlttq,dlcq,
			                      	chq,gdwlq,req,revtq,saleq,cogsq,xsgaq,txtq,capxy,dpq,niq,rectq,invtq,
			                      	apq,apalchy,recchy,xaccq,xrdq,epspxq,aqcy,pstkq,txditcq,invchy,ibq,
			                      	dvy,cshoq
			                      from comp.fundq
		                         ''') 
		compustat_q.to_pickle(input_path+'compustat/compustat_q.pkl')
		db.close()
	return compustat_q

def get_comp_dates(input_path='input/'):
	#download and process mapping table for Compustat filing dates from WRDS api and save as pandas df
	if os.path.isfile(input_path+'compustat/as_of_dates.pkl'):
		comp_dates = pd.read_pickle(input_path+'compustat/as_of_dates.pkl')
	else:
		db = wrds.Connection()
		comp_dates = db.raw_sql('''
			                      select distinct gvkey,datadate,rdq
			                      from comp.fundq
		                    	''') 
		comp_dates.to_pickle(input_path+'compustat/as_of_dates.pkl')
		db.close()
	return comp_dates

def get_ibes_summary(input_path='input/'):
	#download IBES summary file from WRDS api and save as pandas df
	if os.path.isfile(input_path+'ibes/ibes_summary.pkl'):
		ibes_summary = pd.read_pickle(input_path+'ibes/ibes_summary.pkl')
	else:
		db = wrds.Connection()
		#EPS measure estimates
		ibes_summary = db.raw_sql('''
				                      select ticker,cusip,statpers,fpedats,fiscalp,fpi,measure,
				                      	numest,numup,numdown,medest,meanest,stdev,highest,lowest
				                      from ibes.statsumu_epsus
		                    	  ''') 
		#non-EPS measure estimates
		ibes_summary2 = db.raw_sql('''
				                      select ticker,cusip,statpers,fpedats,fiscalp,fpi,measure,
				                      	numest,numup,numdown,medest,meanest,stdev,highest,lowest
				                      from ibes.statsumu_xepsus
		                    	   ''') 
		
		ibes_summary = pd.concat([ibes_summary,ibes_summary2],axis=0)
		ibes_summary.to_pickle(input_path+'ibes/ibes_summary.pkl')
		db.close()
	return ibes_summary

def get_ibes_actual(input_path='input/'):
	#download IBES actuals file from WRDS api and save as pandas df
	if os.path.isfile(input_path+'ibes/ibes_actual.pkl'):
		ibes_actual = pd.read_pickle(input_path+'ibes/ibes_actual.pkl')
	else:
		db = wrds.Connection()
		#EPS measure actuals
		ibes_actual = db.raw_sql('''
			                      select ticker,cusip,anndats,pends,measure,value,pdicity
			                      from ibes.actu_epsus
	  		                     ''') 

		#non-EPS measure actuals
		ibes_actual2 = db.raw_sql('''
			                      select ticker,cusip,anndats,pends,measure,value,pdicity
			                      from ibes.actu_xepsus
			              	      ''') 

		ibes_actual = pd.concat([ibes_actual,ibes_actual2],axis=0)
		ibes_actual.to_pickle(input_path+'ibes/ibes_actual.pkl')
		db.close()
	return ibes_actual

def get_crspibes_link(input_path='input/'):
	#get crsp/ibes linking table from https://wrds-www.wharton.upenn.edu/pages/grid-items/linking-suite-wrds/
	link = pd.read_csv(input_path+'linking/crspibes.csv')
	link = link.rename(columns = {'TICKER':'ticker','PERMNO':'permno'})
	link = link.drop_duplicates()
	link['sdate'] = pd.to_datetime(link.sdate)
	link['edate'] = pd.to_datetime(link.edate)
	link = link.groupby(['permno','ticker']).agg({'sdate':'min','edate':'max'}).reset_index()
	return link

def get_inst_hold(input_path='input/'):
	#download aggregated TR institutional ownership data from WRDS api and save as pandas df
	if os.path.isfile(input_path+'tr/13f_agg.pkl'):
		inst_hold = pd.read_pickle(input_path+'tr/13f_agg.pkl')
	else:
		inst_hold = db.raw_sql('''
		                      select rdate, cusip, avg(shrout1) shrout1, avg(shrout2) shrout2, sum(shares) shares
		                      from tr_13f.s34
		                      group by cusip, rdate
		                     ''') 
		inst_hold.to_pickle(input_path+'tr/13f_agg.pkl')
		db.close()
	return inst_hold

def get_ff_model(input_path='input/'):
	if os.path.isfile(input_path+'ff_ret.pkl'):
		ff = pd.read_pickle(input_path+'ff_ret.pkl')
	else:
		ff = pdr.famafrench.FamaFrenchReader('F-F_Research_Data_Factors',start='1960-01-01').read()[0]/100
		ff['UMD'] = pdr.famafrench.FamaFrenchReader('F-F_Momentum_Factor',start='1960-01-01').read()[0].iloc[:,0]/100
		ff.index = pd.to_datetime(ff.index.astype(str),format='%Y-%m')
		ff.to_pickle(input_path+'ff_ret.pkl')
	return ff

def get_recession_dates(input_path='input/'):
	if os.path.isfile(input_path+'recession_dates.pkl'):
		rec = pd.read_pickle(input_path+'recession_dates.pkl')
	else:
		rec = pdr.fred.FredReader('USREC', start='1950-01-01', end='2020-01-01').read()
		rec.to_pickle(input_path+'recession_dates.pkl')
	return rec

def get_macro_df(input_path='input/'):
	if os.path.isfile(input_path+'macro.pkl'):
		df = pd.read_pickle(input_path+'macro.pkl')
	else: 
		df = pdr.fred.FredReader(['BAA','AAA'], start='1950-01-01', end='2020-01-01').read()
		df.index=pd.to_datetime(df.index)
		df['BAA-AAA'] = df['BAA']-df['AAA']
		df['BAA-AAA_pct'] = df['BAA-AAA'].pct_change()
		df2 = pdr.fred.FredReader(['DGS10','DGS2'], start='1950-01-01', end='2020-01-01').read()
		df2.index=pd.to_datetime(df2.index)
		df2['UST10-UST2'] = df2['DGS10']-df2['DGS2']
		df2['date'] = pd.to_datetime(df2.index.strftime('%Y-%m-01'))
		df[['UST10','UST2','UST10-UST2']] = df2.groupby('date')[['DGS10','DGS2','UST10-UST2']].last()
		ind = get_indices_d()
		ind['date'] = pd.to_datetime(ind['date'])
		ind = ind.sort_values(by='date')
		ind['MktVol'] = ind['sprtrn'].rolling(22).std()
		ind['date'] = pd.to_datetime(ind['date'].dt.strftime('%Y-%m-01'))
		df['MktVol'] = ind.groupby('date')['MktVol'].last()
		ind = get_indices_m()
		ind.index = pd.to_datetime(ind['date'])-MonthBegin(1)
		df['sprtrn'] = ind['sprtrn']
		out = expanding_garch(df['sprtrn'],order=(1,0,1),start_date='1950-01-01')
		df[['GarchVol','GarchVol_z']] = out[['vol','vol_z']].astype(float)
		df[['BAA-AAA_lag','UST10-UST2_lag','MktVol_lag']] = df[['BAA-AAA','UST10-UST2','MktVol']].shift(1)
		df[['BAA-AAA_lag_z','UST10-UST2_lag_z','MktVol_lag_z']] = df[['BAA-AAA_lag','UST10-UST2_lag','MktVol_lag']].expanding().apply(lambda x: (x.iloc[-1]-x.mean())/x.std())
		df.to_pickle(input_path+'macro.pkl')
	return df

if __name__ == '__main__':
	crsp_m = get_crsp_m()
	crsp_d = get_crsp_d()
	ind_d = get_indices_d()
	ind_m = get_indices_m()
	crsp_div = get_crsp_divs()
	crspcomp = get_crspcomp_link()
	comp_q = get_compustat_q()
	comp_d = get_comp_dates()
	ibes_summary = get_ibes_summary()
	ibes_actual = get_ibes_actual()
	crspibes = get_crspibes_link()
	inst_hold = get_inst_hold()