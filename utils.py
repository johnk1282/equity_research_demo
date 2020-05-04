import multiprocessing as mp
import pandas as pd
from pandas.tseries.offsets import *
import numpy as np
import os
from statsmodels.tsa.arima_model import ARIMA
from arch import arch_model

def multi_groupby(df,groupby_cols,func):
    #helper function to do a pandas groupby().apply() using multi-processing
    pool = mp.Pool(processes=mp.cpu_count())
    results = pool.map(func,[df_sub for _,df_sub in df.groupby(groupby_cols)])
    pool.close()
    pool.join()
    return pd.concat(results)

def normalize_signal_ranks(x):
	return 2*(x.rank()-x.rank().mean())/(x.rank()-x.rank().mean()).abs().sum()

def create_macro_factor(series,normalizing_func=None,winsorize_cutoff=None,scale=1,shift=0,rolling_window=120,min_periods=120,order=(1,0,1)):
	if normalizing_func==None:
		pass
	elif normalizing_func=='expanding_arima':
		series = expanding_arima(series,order=order,min_periods=min_periods)['pred_z']
	elif normalizing_func=='expanding_garch':
		series = expanding_garch(series,order=order,min_periods=min_periods)['vol_z']
	elif normalizing_func=='expanding_z':
		series = series.expanding(min_periods=min_periods).apply(lambda x: (x.iloc[-1]-x.mean())/x.std())
	elif normalizing_func=='rolling_z':
		series = series.rolling(rolling_window,min_periods=min_periods).apply(lambda x: (x.iloc[-1]-x.mean())/x.std())
	elif normalizing_func=='expanding_ptile':
		series = 2*series.expanding(min_periods=min_periods).apply(lambda x: (x.dropna()<x.iloc[-1]).mean())-1
	elif normalizing_func=='rolling_ptile':
		series = 2*series.rolling(rolling_window,min_periods=min_periods).apply(lambda x: (x.dropna()<x.iloc[-1]).mean())-1
	else:
		raise ValueError('normalizing_func not recognized')
	series = series.dropna()
	if winsorize_cutoff!=None:
		series = scale*series.where(series.abs()<winsorize_cutoff,winsorize_cutoff*np.sign(series))+shift
	else:
		series = scale*series+shift
	return series

def expanding_arima(data,order=(1,0,1),start_date='1980-01-01',min_periods=120):
    temp = data.dropna()
    d = pd.to_datetime(start_date)
    end_date = temp.index[-1]
    out = pd.DataFrame(columns=['pred','pred_z'],index=temp.index)
    while d<=end_date:
        temp2 = temp.loc[temp.index<d]
        if len(temp2)<min_periods:
            d+=DateOffset(months=1)
            continue
        mod = ARIMA(temp2,order=order,freq='MS')
        res = mod.fit()
        pred = res.predict(d)[0]
        out.loc[d]=[pred,(pred-res.fittedvalues.mean())/res.fittedvalues.std()]
        d+=DateOffset(months=1)
    return out.astype(float)

def expanding_garch(data,order=(1,0,1),start_date='1980-01-01',min_periods=120):
    temp = data.dropna()
    d = pd.to_datetime(start_date)
    end_date = data.index[-1]
    out = pd.DataFrame(columns=['vol','vol_z'],index=data.index)
    while d<=end_date:
        temp2 = temp.loc[temp.index<d]       
        if len(temp2)<min_periods:
            d+=DateOffset(months=1)
            continue
        mod = arch_model(100*temp2,mean='AR',lags=1,vol='garch',p=order[0],o=order[1],q=order[2],dist='Normal')
        res = mod.fit(disp='off')
        std = res.conditional_volatility.std()
        mean = res.conditional_volatility.mean()
        forecasts = res.forecast(horizon=1)
        vol = np.sqrt(forecasts.variance['h.1'].values[-1])
        out.loc[d]=[vol,(vol-mean)/std]
        d+=DateOffset(months=1)
    return out.astype(float)

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