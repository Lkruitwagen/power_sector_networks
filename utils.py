import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
import networkx as nx
import random, math, glob, os, pickle, time, copy, ternary
import pandas as pd
import statsmodels.api as sm
import scipy.stats as s
import numpy as np
from PIL import Image
from sklearn import linear_model
from sklearn.cluster import KMeans
from helpers import *
from scipy.optimize import minimize, curve_fit
from scipy.misc import factorial
from collections import OrderedDict


""" fix all the wepp stuff, fix categories, interpolate dates, add all the columns """
def prep_wepp(wepp_df):
    # merge with ISO, country budgets and load factors 
    #print '~~~~~~ GENERATING DF ~~~~~~~'
    #print 'loading df...'
    df_iso = pd.read_csv('country_ISO_update.csv')
    #fuel_class = 'fuel_classification_database.dta'
    df_fuel_class = pd.read_csv('fuel_class_db.csv')
    heat_rates_xls = 'Heat rates_v3.xls'
    df_heatrates = pd.read_excel(heat_rates_xls, sheet_name='CSV_output')
    df_load_factor = pd.io.stata.read_stata('load_factor_database.dta')

    #print 'loaded dfs: '
    #print 'merging dfs and filling missing years...'
    #df_fuel_load = pd.merge(df_fuel_class, df_load_factor, on='fuel_class')
    #print df_iso
    #print df_fuel_class
    #print df_heatrates
    #print df_load_factor
    #print list(wepp_df)
    #print wepp_df['FUEL']

    df_fuel_class.rename(columns = {'fuel': 'FUEL'}, inplace = True)
    
    #fix fuel classes
    wepp_df = wepp_df.merge(df_fuel_class, on='FUEL', how='left')
    
    #print wepp_df[pd.isnull(wepp_df.FUEL)]
    df_wepp_em_fact = pd.read_csv('wepp_em_fact.csv')

    #merge emissions factors
    wepp_df = wepp_df.merge(df_wepp_em_fact, left_on='FUEL', right_on='fuel', how='left')

    #prepare lookup indexer
    wepp_df['FORMAT_HR'] = wepp_df.apply(lambda row: format_hr(row), axis=1)

    #standardise statuses
    wepp_df.loc[wepp_df.STATUS=='DEF', 'STATUS'] = 'PLN'
    wepp_df.loc[wepp_df.STATUS=='DEL', 'STATUS'] = 'CON'
    wepp_df.loc[wepp_df.STATUS=='UNK', 'STATUS'] = 'PLN'
    wepp_df.loc[wepp_df.STATUS=='DAC', 'STATUS'] = 'STN'

    #print  list(df_iso)

    #add ISO
    wepp_df = wepp_df.merge(df_iso[['Caps','ISO','Region']], left_on='COUNTRY', right_on='Caps', how='left')

    #fill in missing years
    all_training = wepp_df[['YEAR','fuel_class','STATUS','Region','FORMAT_HR']]
    all_training['fuel_class'] = all_training['fuel_class'].astype('category')
    all_training['STATUS'] = all_training['STATUS'].astype('category')
    all_training['Region'] = all_training['Region'].astype('category')
    all_training['FORMAT_HR'] = all_training['FORMAT_HR'].astype('category')
    all_training = pd.get_dummies(all_training[['YEAR','fuel_class','STATUS','Region','FORMAT_HR']], columns = ['fuel_class','STATUS','Region','FORMAT_HR'])

    year_train_X = all_training[all_training.YEAR.notnull()].drop('YEAR', axis=1)
    year_train_Y = all_training.loc[all_training.YEAR.notnull(),'YEAR']
    year_train_X = sm.add_constant(year_train_X)

    test_data = all_training.loc[all_training.YEAR.isnull()].drop('YEAR', axis=1)
    test_data = sm.add_constant(test_data)

    est = sm.OLS(year_train_Y, year_train_X)
    est = est.fit()

    wepp_df['YEAR_EST_FLAG'] = 0
    wepp_df.loc[wepp_df.YEAR.isnull(),'YEAR_EST_FLAG'] = 1
    wepp_df.loc[wepp_df.YEAR.isnull(),'YEAR'] = est.predict(test_data)
    


    #get heatrates
    wepp_df = wepp_df.merge(df_heatrates, left_on='FORMAT_HR', right_on='unique_id', how='left')
    wepp_df['HEATRATE'] = wepp_df.apply(lambda row: get_hr(row), axis=1)
    drop_cols = [col for col in list(wepp_df) if isinstance(col,int)]
    wepp_df.drop(drop_cols, axis=1, inplace=True)

    #get CO2 int, CCCE
    wepp_df = wepp_df.merge(df_load_factor, on='fuel_class', how='left')
    wepp_df['YEARS_LEFT'] = np.where(wepp_df['STATUS']=='OPR', wepp_df['YEAR']+40-2017, 0)
    wepp_df.YEARS_LEFT.clip(lower=0.0, inplace=True) #set min years left to 0
    #print 'dfs merged and interped: '


    #print 'calculating carbon and MWs...'

    wepp_df['CO2_INT'] = wepp_df['em_fact'] /2.205 * wepp_df['HEATRATE'] / 1000
    wepp_df['CCCE'] = 8760 * wepp_df['MW'] * wepp_df['YEARS_LEFT'] * wepp_df['load_factor'] * wepp_df['CO2_INT'] /1000 #tonnes 
    #wepp_df.sort_values('CCCE', inplace=True)

    #print wepp_df
    #print list(wepp_df)
    #print wepp_df.CCCE


    #print all_countries
    #exit()

    #sort WEPP
    wepp_df.sort_values('CCCE', inplace=True, ascending=False)

    wepp_df['green']=wepp_df.fuel_class.isin(['SUN','BIOGAS','BIOOIL','WIND','BIOMASS','GEOTHERMAL'])
    wepp_df['green_MW'] = wepp_df.MW*wepp_df.green
    wepp_df['blue']=wepp_df.fuel_class.isin(['WATER','NUCLEAR'])
    wepp_df['blue_MW'] = wepp_df.MW*wepp_df.blue
    wepp_df['solar']=wepp_df.fuel_class.isin(['SUN'])
    wepp_df['solar_MW'] = wepp_df.MW*wepp_df.solar
    wepp_df['wind']=wepp_df.fuel_class.isin(['WIND'])
    wepp_df['wind_MW'] = wepp_df.MW*wepp_df.wind
    wepp_df['ff']=~wepp_df.fuel_class.isin(['SUN','BIOGAS','BIOOIL','WIND','BIOMASS','GEOTHERMAL','WATER','NUCLEAR'])
    wepp_df['ff_MW'] = wepp_df.MW*wepp_df.ff
    
    return wepp_df

def prep_iso_slices(wepp_df, select_iso):
    prep_slice = wepp_df[wepp_df.ISO==select_iso]
    
    #group the units by plant, add the 
    prep_slice_plants = prep_slice[['PLANT','COMPANY']].groupby(['PLANT']).agg(lambda x:x.value_counts().index[0])

    #print prep_slice.loc[pd.isnull(prep_slice.fuel_class)]

    prep_slice_plants['fuel_class'] = prep_slice[['PLANT','fuel_class','MW']].sort_values('MW', ascending=False).groupby(['PLANT']).nth(0).fuel_class#.agg(lambda x:x.value_counts().index[0])
    prep_slice_plants['YEAR'] = prep_slice[['PLANT','YEAR']].groupby(['PLANT']).mean()

    for c in ['MW','CCCE','green_MW','blue_MW','solar_MW','wind_MW','ff_MW']:
        prep_slice_plants[c] = prep_slice[['PLANT',c]].groupby(['PLANT']).sum()
        
    prep_slice_plants['solar'] = prep_slice[['PLANT','solar_MW']].groupby(['PLANT']).sum()>10**-4
    prep_slice_plants['wind'] = prep_slice[['PLANT','wind_MW']].groupby(['PLANT']).sum()>10**-4
    #print prep_slice_plants[['green_MW','blue_MW','ff_MW']].idxmax(axis=1)
    prep_slice_plants['green'] = prep_slice_plants[['green_MW','blue_MW','ff_MW']].idxmax(axis=1)=='green_MW'
    prep_slice_plants['blue'] = prep_slice_plants[['green_MW','blue_MW','ff_MW']].idxmax(axis=1)=='blue_MW'
    prep_slice_plants['ff'] = prep_slice_plants[['green_MW','blue_MW','ff_MW']].idxmax(axis=1)=='ff_MW'
        
    #for c in ['green','blue','solar','wind','ff']:
    #    prep_slice_plants[c] = prep_slice[['PLANT',c]].groupby(['PLANT']).agg(lambda x:x.value_counts().index[0])

    
    
    prep_slice_meta = prep_slice_plants[['COMPANY','MW']].groupby(['COMPANY']).sum()
    prep_slice_meta['COUNT'] = prep_slice_plants[['COMPANY','MW']].groupby(['COMPANY']).count()
    for c in ['CCCE','green_MW','blue_MW','solar_MW','wind_MW','ff_MW']:
        prep_slice_meta[c] = prep_slice[['COMPANY',c]].groupby(['COMPANY']).sum()
        
    return prep_slice, prep_slice_plants, prep_slice_meta

def make_null_model(iso_slice_plants, years):
    
    color_dict = {}
    color_dict['name'] = {
        'g':'green',
        'b':'blue',
        'f':'ff'
        }
    color_dict['color_arr'] = {
        'g':[1.0,0.0,0.0],
        'b':[0.0,1.0,0.0],
        'f':[0.0,0.0,1.0],
        }

    plant_ii = 0
    company_ii = 0

    null_df = {}
    null_df[years[0]] = iso_slice_plants[years[0]]

    new_assets = {}
    logreg = {}

    retired_plants = {}
    extra_plants = {}
    epsilons = {}
    new_plants = {}
    reso = {}

    for y in years[1:]:
    
        epsilons[y] = {}
    
    
        ### GENERATE NEW PLANTS - BASED ON PR_G/B/FF, WEIBULL SIZE DIST, 
       
        new_plants_ini = [p for p in iso_slice_plants[y].index.unique() if p not in iso_slice_plants[y-1].index.unique()]
    
        #print iso_slice_plants[y].ix[new_plants_ini]
    
        ADD_MW_ALL = iso_slice_plants[y].ix[new_plants_ini].MW.sum()
        new_plants[y] = list(iso_slice_plants[y].ix[new_plants_ini].loc[np.abs(iso_slice_plants[y].ix[new_plants_ini].YEAR-y)<5.0].index.values)

        extra_plants[y] = [p for p in new_plants_ini if p not in new_plants[y]]
        #print extra_plants[y]
    
        for c in ['green_MW','blue_MW','ff_MW']:
            epsilons[y][c] = len([p for p in new_plants[y] if (iso_slice_plants[y].at[p,c]>10**-4.0)&(iso_slice_plants[y].at[p,'COMPANY'] not in iso_slice_plants[y-1].COMPANY.unique())])/float(len(new_plants[y]))
    
        retired_plants[y] = [p for p in iso_slice_plants[y-1].index.unique() if p not in iso_slice_plants[y].index.unique()]
        #print retired_plants
        
        #print iso_slice_plants[y].ix[new_plants].loc[iso_slice_plants[y].ix[new_plants].green==True].sort_values('green_MW')
        ADD_MW = iso_slice_plants[y].ix[new_plants[y]].MW.sum()
        #print 'add_MW', ADD_MW, 'all added MW', ADD_MW_ALL, 'retired MW', iso_slice_plants[y-1].ix[retired_plants[y]].MW.sum()
    
        #static model - reproduce distribution for each time step just to test preferential attachment model
        Pr_green = len(iso_slice_plants[y].ix[new_plants[y]].loc[iso_slice_plants[y].ix[new_plants[y]].green==True])/float(len(new_plants[y]))
        Pr_blue = len(iso_slice_plants[y].ix[new_plants[y]].loc[iso_slice_plants[y].ix[new_plants[y]].blue==True])/float(len(new_plants[y]))
        Pr_ff = len(iso_slice_plants[y].ix[new_plants[y]].loc[iso_slice_plants[y].ix[new_plants[y]].ff==True])/float(len(new_plants[y]))
    
        #print iso_slice_plants[y].ix[new_plants].loc[iso_slice_plants[y].ix[new_plants].blue==True]
        #print Pr_green, Pr_blue, Pr_ff
    
        reso[y] = {}
    
        #use full population for size of asset
        #print 'green, blue, ff plants', (iso_slice_plants[y-1].green==True).sum(), (iso_slice_plants[y-1].blue==True).sum(), (iso_slice_plants[y].ff==True).sum() 
        #reso['g'] = s.exponweib.fit(iso_slice_plants[y-1][iso_slice_plants[y-1].green==True].MW.values,floc=0, fa=1)
        #reso['b'] = s.exponweib.fit(iso_slice_plants[y-1][iso_slice_plants[y-1].blue==True].MW.values,floc=0, fa=1)
        #reso['f'] = s.exponweib.fit(iso_slice_plants[y-1][iso_slice_plants[y-1].ff==True].MW.values,floc=0, fa=1)
    
        reso[y]['g'] = s.exponweib.fit(iso_slice_plants[y].ix[new_plants[y]].loc[iso_slice_plants[y].ix[new_plants[y]].green==True].MW.values)#,floc=0, fa=1)
        reso[y]['b'] = s.exponweib.fit(iso_slice_plants[y].ix[new_plants[y]].loc[iso_slice_plants[y].ix[new_plants[y]].blue==True].MW.values)#,floc=0, fa=1)
        reso[y]['f'] = s.exponweib.fit(iso_slice_plants[y].ix[new_plants[y]].loc[iso_slice_plants[y].ix[new_plants[y]].ff==True].MW.values)#,floc=0, fa=1)
    
        new_assets[y] = []
    
        NEW_MW=0.0
    
        while NEW_MW<=ADD_MW:
            c = np.random.choice(['g','b','f'],p=np.array([Pr_green,Pr_blue,Pr_ff])/np.sum([Pr_green,Pr_blue,Pr_ff]))
            MW = s.exponweib.rvs(*reso[y][c],size=1)[0]
            new_assets[y].append({'color':c, 'MW':MW})
            NEW_MW +=MW
        
        #print max([n['MW'] for n in new_assets[y]])
        #print len(new_assets[y])
        #print list(iso_slice_plants[y].ix[new_plants])
        
        null_df[y] = copy.deepcopy(null_df[y-1])
    
        #drop retired plants
        null_df[y] = null_df[y][~null_df[y].index.isin(retired_plants[y])]
    
        cols = list(null_df[y-1])
    
        X = []
    
    
        ### attach new assets
        x = null_df[y-1][['MW','COMPANY','green','blue','ff']]
    
        #x['_size'] = (np.log10(x['MW'])-np.log10(null_df[y-1]['MW'].min()))/float(np.log10(null_df[y-1]['MW'].max())-np.log10(null_df[y-1]['MW'].min()))
        x['_size'] = (x['MW']-null_df[y-1]['MW'].min())/float(null_df[y-1]['MW'].max()-null_df[y-1]['MW'].min())
        #x['_size'] = x['MW'].sort_values().cumsum()/x.MW.sum()
        x['_green'] = x['green'].astype(int)
        x['_blue'] = x['blue'].astype(int)
        x['_ff'] = x['ff'].astype(int)
        x['_norm'] = np.sqrt(np.square(x[['_size','_green','_blue','_ff']]).sum(axis=1))
    
        n = pd.DataFrame(new_assets[y])
        #n['_size'] = (np.log10(n['MW'])-np.log10(null_df[y-1]['MW'].min()))/float(np.log10(null_df[y-1]['MW'].max())-np.log10(null_df[y-1]['MW'].min()))
        n['_size'] = (n['MW']-null_df[y-1]['MW'].min())/float(null_df[y-1]['MW'].max()-null_df[y-1]['MW'].min())
        n['_green'] = (n['color']=='g').astype(int)
        n['_blue'] = (n['color']=='b').astype(int)  
        n['_ff'] = (n['color']=='f').astype(int) 
        n['_norm'] = np.sqrt(np.square(n[['_size','_green','_blue','_ff']]).sum(axis=1))
    
        for c in ['_size','_green','_blue','_ff']:
            x[c] = x[c]/x['_norm']
            n[c] = n[c]/n['_norm']
        
        
        X = x[['_size','_green','_blue','_ff']].values
        N = n[['_size','_green','_blue','_ff']].values
    
    
        ### Difference method
        X = np.stack([X for _ in range(N.shape[0])], axis=0)
        N =np.stack([N for _ in range(X.shape[1])], axis=1)
    
        P = 1.0/(((X-N)**2).sum(axis=-1))
        P = P/P.sum(axis=-1)[:, np.newaxis]
  
        select = random_choice_prob_index(P, axis=1)
    
        for ii, n_a in enumerate(new_assets[y]):
            if np.random.rand()<epsilons[y][color_dict['name'][n_a['color']]+'_MW']:
            
                name = 'NEW_COMPANY_'+str(company_ii)
                company_ii+=1
                #print 'new company'
            else:
                #print 'n_MW: ', n_a['MW'], 'chosen_MW: ',x.iloc[select[ii]].MW, 'color:', n_a['color'], 'ch_col:', x.iloc[select[ii]].green, x.iloc[select[ii]].blue,x.iloc[select[ii]].ff
                name = x.iloc[select[ii]].COMPANY
            
        
            null_df[y] = null_df[y].append(pd.DataFrame({
                'COMPANY':name, 
                'fuel_class':'N/A_GEN', 
                'YEAR':y, 
                'green':color_dict['color_arr'][n_a['color']][0], 
                'blue':color_dict['color_arr'][n_a['color']][1], 
                'solar':0.0, 
                'wind':0.0, 
                'ff':color_dict['color_arr'][n_a['color']][2], 
                'MW':n_a['MW'], 
                'CCCE':0.0, 
                'green_MW':color_dict['color_arr'][n_a['color']][0]*n_a['MW'], 
                'blue_MW':color_dict['color_arr'][n_a['color']][1]*n_a['MW'], 
                'solar_MW':0.0, 
                'wind_MW':0.0, 
                'ff_MW':color_dict['color_arr'][n_a['color']][2]*n_a['MW'],
            }, index=['NEW_PLANT_'+str(plant_ii)]))
        
            plant_ii+=1
        
        for p in extra_plants[y]:
            null_df[y] = null_df[y].append(iso_slice_plants[y].ix[p])
            
    return null_df, retired_plants, new_plants, extra_plants, epsilons, reso
    
def get_stats(df, years):
    genned_stats = {
        'n_companies': [len(df[y].COMPANY.unique()) for y in years[1:]],
        'n_plants': [(len(df[y])) for y in years[1:]],
        'n_green': [(df[y].green_MW>0.0).sum() for y in years[1:]],
        'n_blue':[(df[y].blue_MW>0.0).sum() for y in years[1:]],
        'n_ff':[(df[y].ff_MW>0.0).sum() for y in years[1:]],
        'MW_total':[df[y].MW.sum() for y in years[1:]],
        'MW_green':[df[y].green_MW.sum() for y in years[1:]],
        'MW_blue':[df[y].blue_MW.sum() for y in years[1:]],
        'MW_ff':[df[y].ff_MW.sum() for y in years[1:]],
        'mean_green_MW':[df[y].groupby('COMPANY').sum().green_MW.mean() for y in years[1:]],
        'mean_blue_MW':[df[y].groupby('COMPANY').sum().blue_MW.mean()  for y in years[1:]],
        'mean_ff_MW':[df[y].groupby('COMPANY').sum().ff_MW.mean()  for y in years[1:]],
        'std_green_MW':[df[y].groupby('COMPANY').sum().green_MW.std()  for y in years[1:]],
        'std_blue_MW':[df[y].groupby('COMPANY').sum().blue_MW.std() for y in years[1:]],
        'std_ff_MW':[df[y].groupby('COMPANY').sum().ff_MW.std() for y in years[1:]],
        'mean_green_count':[df[y].groupby('COMPANY').sum().green.mean() for y in years[1:]],
        'mean_blue_count':[df[y].groupby('COMPANY').sum().blue.mean()  for y in years[1:]],
        'mean_ff_count':[df[y].groupby('COMPANY').sum().ff.mean()  for y in years[1:]],
        'std_green_count':[df[y].groupby('COMPANY').sum().green.std()  for y in years[1:]],
        'std_blue_count':[df[y].groupby('COMPANY').sum().blue.std() for y in years[1:]],
        'std_ff_count':[df[y].groupby('COMPANY').sum().ff.std() for y in years[1:]],
        
        'mean_green_MW_f':[df[y].groupby('COMPANY').filter(lambda x: x.green_MW.sum()>10**-4.).groupby('COMPANY').sum().green_MW.mean() for y in years[1:]],
        'mean_blue_MW_f':[df[y].groupby('COMPANY').filter(lambda x: x.blue_MW.sum()>10**-4.).groupby('COMPANY').sum().blue_MW.mean()  for y in years[1:]],
        'mean_ff_MW_f':[df[y].groupby('COMPANY').filter(lambda x: x.ff_MW.sum()>10**-4.).groupby('COMPANY').sum().ff_MW.mean()  for y in years[1:]],
        'std_green_MW_f':[df[y].groupby('COMPANY').filter(lambda x: x.green_MW.sum()>10**-4.).groupby('COMPANY').sum().green_MW.std()  for y in years[1:]],
        'std_blue_MW_f':[df[y].groupby('COMPANY').filter(lambda x: x.blue_MW.sum()>10**-4.).groupby('COMPANY').sum().blue_MW.std() for y in years[1:]],
        'std_ff_MW_f':[df[y].groupby('COMPANY').filter(lambda x: x.ff_MW.sum()>10**-4.).groupby('COMPANY').sum().ff_MW.std() for y in years[1:]],
        'mean_green_count_f':[df[y].groupby('COMPANY').filter(lambda x: x.green.sum()>10**-4.).groupby('COMPANY').sum().green.mean() for y in years[1:]],
        'mean_blue_count_f':[df[y].groupby('COMPANY').filter(lambda x: x.blue.sum()>10**-4.).groupby('COMPANY').sum().blue.mean()  for y in years[1:]],
        'mean_ff_count_f':[df[y].groupby('COMPANY').filter(lambda x: x.ff.sum()>10**-4.).groupby('COMPANY').sum().ff.mean()  for y in years[1:]],
        'std_green_count_f':[df[y].groupby('COMPANY').filter(lambda x: x.green.sum()>10**-4.).groupby('COMPANY').sum().green.std()  for y in years[1:]],
        'std_blue_count_f':[df[y].groupby('COMPANY').filter(lambda x: x.blue.sum()>10**-4.).groupby('COMPANY').sum().blue.std() for y in years[1:]],
        'std_ff_count_f':[df[y].groupby('COMPANY').filter(lambda x: x.ff.sum()>10**-4.).groupby('COMPANY').sum().ff.std() for y in years[1:]],
        }
    

    return genned_stats

def plot_stats(truth_stats, synth_stats, null_1_stats, years):

    custom_lines = [Line2D([0], [0], color='gray', lw=2, linestyle='-'),
                    Line2D([0], [0], color='gray', lw=2, linestyle=':'),
                    Line2D([0], [0], color='gray', lw=2, linestyle='--')]

    fig, axs = plt.subplots(5,2,figsize=(16,20))
    for ci,cj,ck in zip(['n_green','n_blue','n_ff'],['MW_green','MW_blue','MW_ff'],['green','blue','black']):
        axs[0,0].plot(years[1:],truth_stats[ci], color=ck, linestyle='-')
        axs[0,0].plot(years[1:],synth_stats[ci], color=ck, linestyle='--')
        axs[0,0].plot(years[1:],null_1_stats[ci], color=ck, linestyle=':')

        axs[0,1].plot(years[1:],truth_stats[cj], color=ck, linestyle='-')
        axs[0,1].plot(years[1:],synth_stats[cj], color=ck, linestyle='--')
        axs[0,1].plot(years[1:],null_1_stats[cj], color=ck, linestyle=':')
    
    for ci,cj,ck in zip(['mean_green_MW','mean_blue_MW','mean_ff_MW'],['std_green_MW','std_blue_MW','std_ff_MW'],['green','blue','black']):
        axs[1,0].plot(years[1:],truth_stats[ci], color=ck, linestyle='-')
        axs[1,0].plot(years[1:],synth_stats[ci], color=ck, linestyle='--')
        axs[1,0].plot(years[1:],null_1_stats[ci], color=ck, linestyle=':')

        axs[1,1].plot(years[1:],truth_stats[cj], color=ck, linestyle='-')
        axs[1,1].plot(years[1:],synth_stats[cj], color=ck, linestyle='--')
        axs[1,1].plot(years[1:],null_1_stats[cj], color=ck, linestyle=':')
    
    for ci,cj,ck in zip(['mean_green_count','mean_blue_count','mean_ff_count'],['std_green_count','std_blue_count','std_ff_count'],['green','blue','black']):
        axs[2,0].plot(years[1:],truth_stats[ci], color=ck, linestyle='-')
        axs[2,0].plot(years[1:],synth_stats[ci], color=ck, linestyle='--')
        axs[2,0].plot(years[1:],null_1_stats[ci], color=ck, linestyle=':')

        axs[2,1].plot(years[1:],truth_stats[cj], color=ck, linestyle='-')
        axs[2,1].plot(years[1:],synth_stats[cj], color=ck, linestyle='--')
        axs[2,1].plot(years[1:],null_1_stats[cj], color=ck, linestyle=':')
    
    for ci,cj,ck in zip(['mean_green_MW_f','mean_blue_MW_f','mean_ff_MW_f'],['std_green_MW_f','std_blue_MW_f','std_ff_MW_f'],['green','blue','black']):
        axs[3,0].plot(years[1:],truth_stats[ci], color=ck, linestyle='-')
        axs[3,0].plot(years[1:],synth_stats[ci], color=ck, linestyle='--')
        axs[3,0].plot(years[1:],null_1_stats[ci], color=ck, linestyle=':')

        axs[3,1].plot(years[1:],truth_stats[cj], color=ck, linestyle='-')
        axs[3,1].plot(years[1:],synth_stats[cj], color=ck, linestyle='--')
        axs[3,1].plot(years[1:],null_1_stats[cj], color=ck, linestyle=':')
    
    for ci,cj,ck in zip(['mean_green_count_f','mean_blue_count_f','mean_ff_count_f'],['std_green_count_f','std_blue_count_f','std_ff_count_f'],['green','blue','black']):
        axs[4,0].plot(years[1:],truth_stats[ci], color=ck, linestyle='-')
        axs[4,0].plot(years[1:],synth_stats[ci], color=ck, linestyle='--')
        axs[4,0].plot(years[1:],null_1_stats[ci], color=ck, linestyle=':')

        axs[4,1].plot(years[1:],truth_stats[cj], color=ck, linestyle='-')
        axs[4,1].plot(years[1:],synth_stats[cj], color=ck, linestyle='--')
        axs[4,1].plot(years[1:],null_1_stats[cj], color=ck, linestyle=':')

    for ii in range(5):
        for jj in range(2):
            axs[ii,jj].legend(custom_lines, ['Truth', 'Null Model','Synth'])
    axs[0,0].set_title('Number of Plants')
    axs[0,1].set_title('Sum of MW')
    axs[1,0].set_title('Degree (MW) - Mean')
    axs[1,1].set_title('Degree (MW) - Std')
    axs[2,0].set_title('Degree (Count) - Mean')
    axs[2,1].set_title('Degree (Count) - Std')
    axs[3,0].set_title('Degree (MW-Filtered) - Mean')
    axs[3,1].set_title('Degree (MW-Filtered) - Std')
    axs[4,0].set_title('Degree (Count-Filtered) - Mean')
    axs[4,1].set_title('Degree (Count-Filtered) - Std')

    plt.show()
    return 0


def draw_all_components(df_1yr, MW=False, N_GT=3):
    if MW:
        all_cos = df_1yr[['COMPANY','MW']].groupby('COMPANY').filter(lambda x: x['MW'].count()>N_GT).groupby('COMPANY').sum().sort_values('MW',ascending=False).index.values
        
    else:
        all_cos = df_1yr[['COMPANY','MW']].groupby('COMPANY').filter(lambda x: x['MW'].count()>N_GT).groupby('COMPANY').count().sort_values('MW',ascending=False).index.values
        
    #print len(all_cos)
    fig, axs = plt.subplots(int(len(all_cos)/ 10) +1,10,figsize=(16,(int(len(all_cos)/ 10) +1 )*1.5), dpi=100, sharex=True, sharey=True)
    ii=0
    jj=0

    for c in all_cos:
        slice_company = df_1yr.loc[df_1yr['COMPANY']==c]
        #print slice_company

        # makem nodes

        nodes_list = []

        for plant in list(slice_company.index):
            g = int(slice_company.get_value(plant,'green_MW')/slice_company.get_value(plant,'MW')*255)
            b = int(slice_company.get_value(plant,'blue_MW')/slice_company.get_value(plant,'MW')*255)
            size=np.log10(slice_company.get_value(plant,'MW'))*10


            nodes_list.append(
                (plant,{
                    'pos': np.random.rand(2),
                    'type': 'power_station',
                    'n_color':"#{0:02x}{1:02x}{2:02x}".format(clamp(0), clamp(g), clamp(b)),
                    'n_alpha':1.0,
                    'n_size':size
                    })            
                )

        #print nodes_list

        # makem edges

        edges_list = []

        pi=0

        for plant1 in list(slice_company.index):

            pi+=1

            for plant2 in list(slice_company.index)[pi:]:

                if plant1 != plant2:
                    edges_list.append(
                        (plant1, plant2, {
                            'weight': 10.0*np.log10(min(slice_company.get_value(plant1,'MW'),slice_company.get_value(plant2,'MW'))),
                            'type': 'plants',
                            'e_color': 'gray',
                            'e_alpha': 1.0
                        })
                    )


        ### make graph diagram

        G = nx.Graph()

        G.add_nodes_from(nodes_list)
        G.add_edges_from(edges_list)

        if len(edges_list)>0:
            max_weight= max([e[2]['weight'] for e in edges_list])
        else:
            max_weight=1





        #print nx.get_edge_attributes(G,'e_color')

        pos = nx.get_node_attributes(G,'pos')
        #print pos

        pos = nx.spring_layout(G, k=max_weight/8.0, iterations=10, pos=pos, center=[0,0], weight='weight')#fixed=fixed_nodes, 


        ### Draw Edges
        nx.draw_networkx_edges(G,
                                   pos,
                                   edgelist =G.edges(),
                                   ax=axs[jj,ii],
                                   edge_color=[nx.get_edge_attributes(G,'e_color')[e] for e in G.edges()],
                                   alpha=0.2
                                  )

        ### Draw Nodes
        nodes_ax = nx.draw_networkx_nodes(G,
                                              pos,
                                              ax=axs[jj,ii],
                                              nodelist=G.nodes(),
                                              node_size = [nx.get_node_attributes(G,'n_size')[n]*15 for n in G.nodes()],
                                              node_color = [nx.get_node_attributes(G,'n_color')[n] for n in G.nodes()],
                                              alpha= [nx.get_node_attributes(G,'n_alpha')[n] for n in G.nodes()]
                                              )



        nodes_ax.set_edgecolor('w')

        #axs[jj,ii].set_title(c)


        #axs[jj,ii].axis('off')

        ii+=1
        if ii==10:
            ii=0
            jj+=1
            
    for jj in range(axs.shape[0]):
        for ii in range(axs.shape[1]):
            axs[jj,ii].axis('off')
            
            
    plt.show()

    return None
            
def draw_IADS(df_1yr, N_iad=3):
    iad_cos = df_1yr[['COMPANY','MW']].groupby('COMPANY').filter(lambda x: x['MW'].count()==N_iad).groupby('COMPANY').sum().sort_values('MW',ascending=False).index.values
          
    fig, axs = plt.subplots(1,3,figsize=(16,5), dpi=100)
    ii=0
    jj=0

    for c in iad_cos[0:1]:
        slice_company = df_1yr.loc[df_1yr['COMPANY']==c]

        nodes_list = []

        for plant in list(slice_company.index):
            size=np.log10(slice_company.get_value(plant,'MW'))*10

            nodes_list.append(
                (plant,{
                    'pos': np.random.rand(2),
                    'type': 'power_station',
                    'n_color':"#FFFFFF",
                    'n_alpha':1.0,
                    'n_size':size
                    })            
                )

        edges_list = []

        pi=0

        for plant1 in list(slice_company.index):

            pi+=1

            for plant2 in list(slice_company.index)[pi:]:

                if plant1 != plant2:
                    edges_list.append(
                        (plant1, plant2, {
                            'weight': 10.0*np.log10(min(slice_company.get_value(plant1,'MW'),slice_company.get_value(plant2,'MW'))),
                            'type': 'plants',
                            'e_color': 'gray',
                            'e_alpha': 1.0
                        })
                    )
                    
        ### make graph diagram

        G = nx.Graph()

        G.add_nodes_from(nodes_list)
        G.add_edges_from(edges_list)

        if len(edges_list)>0:
            max_weight= max([e[2]['weight'] for e in edges_list])
        else:
            max_weight=1

        pos = nx.get_node_attributes(G,'pos')

        pos = nx.spring_layout(G, k=max_weight/8.0, iterations=10, pos=pos, center=[0,0], weight='weight')#fixed=fixed_nodes, 


        ### Draw Edges
        nx.draw_networkx_edges(G,
                                   pos,
                                   edgelist =G.edges(),
                                   ax=axs[0],
                                   edge_color=[nx.get_edge_attributes(G,'e_color')[e] for e in G.edges()],
                                   alpha=0.2
                                  )

        ### Draw Nodes
        nodes_ax = nx.draw_networkx_nodes(G,
                                              pos,
                                              ax=axs[0],
                                              nodelist=G.nodes(),
                                              node_size = [nx.get_node_attributes(G,'n_size')[n]*15 for n in G.nodes()],
                                              node_color = [nx.get_node_attributes(G,'n_color')[n] for n in G.nodes()],
                                              alpha= [nx.get_node_attributes(G,'n_alpha')[n] for n in G.nodes()]
                                              )



        nodes_ax.set_edgecolor('k')
        
        axs[0].axis('off')
        axs[0].set_ylim([-1.2,1.2])
        axs[0].set_ylim([-1.2,1.2])
        axs[1].axis('off')
        
        tfig, tax = ternary.figure(ax=axs[1], scale=100.0)
        tax.boundary(linewidth=2.0)
        tax.gridlines(color='black', multiple=20)
        # Remove default Matplotlib Axes
        tax.clear_matplotlib_ticks()
        #tax.set_title(str(y)+ ' - '+labels[ii], fontsize=12)
        tax.left_axis_label("FF", fontsize=12)
        tax.right_axis_label("BLUE", fontsize=12)
        tax.bottom_axis_label("GREEN", fontsize=12)

        tax.ticks(axis='lbr', multiple=20, linewidth=1)
        
        IADS = (df_1yr[['COMPANY','MW','green_MW', 'blue_MW','ff_MW']]
                    .groupby('COMPANY').filter(lambda x: x['MW'].count()==N_iad)
                    .groupby('COMPANY').sum())
        IADS['green_norm'] = IADS.green_MW / IADS.MW *100.0
        IADS['blue_norm'] = IADS.blue_MW / IADS.MW * 100.0
        IADS['ff_norm'] = IADS.ff_MW / IADS.MW * 100.0

        IADS_NORM = IADS.as_matrix(columns=['green_norm','blue_norm','ff_norm'])

        color_IADS = []
        for ci in range(IADS_NORM.shape[0]):
            g = int(IADS_NORM[ci,0]/100.*255)
            b = int(IADS_NORM[ci,1]/100.*255)
            color_IADS.append("#{0:02x}{1:02x}{2:02x}".format(clamp(0), clamp(g), clamp(b)))
        n_IADS = len(color_IADS)

        #print DIADS
        tax.scatter(IADS_NORM, marker='1', s=1000, color=color_IADS)#, label='IADS (N='+str(len(color_IADS))+')')

        box_data = [df_1yr[df_1yr.COMPANY.isin(iad_cos)&df_1yr.green_MW>10**-4].green_MW.values,
                    df_1yr[df_1yr.COMPANY.isin(iad_cos)&df_1yr.blue_MW>10**-4].blue_MW.values,
                    df_1yr[df_1yr.COMPANY.isin(iad_cos)&df_1yr.ff_MW>10**-4].ff_MW.values]
        _patches = axs[2].boxplot(box_data,0,'+', whis=[10.,90.], patch_artist=True, widths=0.8)
        
        x_labels = ['N_g = '+str(len(box_data[0])),
                    'N_b = '+str(len(box_data[1])),
                    'N_ff = '+str(len(box_data[2])),]
        
        
        axs[2].set_xticklabels(x_labels)
        axs[2].set_ylabel('MW')
        
        
        color_dict = {
            0:'#00FF00',
            1:'#0000FF',
            2:'#000000',
        }
        for pi in range(3):
            _patches['boxes'][pi].set(facecolor=color_dict[pi])
            _patches['fliers'][pi].set(markeredgecolor=color_dict[pi])

            
    plt.show()

    return None



def draw_full_network(df_1yr, iters=500, k_scale=1.0, weight_scale=1.0):


    fig, ax = plt.subplots(1,1,figsize=(16,9), dpi=100)
    ii=0
    jj=0

    nodes_list = []

    for plant in list(df_1yr.index):
        g = int(df_1yr.get_value(plant,'green_MW')/df_1yr.get_value(plant,'MW')*255)
        b = int(df_1yr.get_value(plant,'blue_MW')/df_1yr.get_value(plant,'MW')*255)
        size=np.log10(df_1yr.get_value(plant,'MW'))*10


        nodes_list.append(
                (plant,{
                    'pos': np.random.rand(2),
                    'type': 'power_station',
                    'n_color':"#{0:02x}{1:02x}{2:02x}".format(clamp(0), clamp(g), clamp(b)),
                    'n_alpha':1.0,
                    'n_size':size
                    })            
                )

        #print nodes_list

        # makem edges
        
    print 'done nodes'

    edges_list = []
    
    
    all_companies = df_1yr.COMPANY.unique()
    ii=0
    print len(all_companies)
    
    
    for c in all_companies:
        
        plants = df_1yr[df_1yr.COMPANY==c].index.values
        pi=0
        
        for p1 in plants:
            
            for p2 in plants[pi:]:
                edges_list.append(
                        (p1, p2, {
                            'weight': 10.0*np.log10(min(df_1yr.get_value(p1,'MW'),df_1yr.get_value(p2,'MW'))),
                            'type': 'plants',
                            'e_color': 'gray',
                            'e_alpha': 1.0
                        }))
                
        print ii
        ii+=1



    ### make graph diagram

    G = nx.Graph()

    G.add_nodes_from(nodes_list)
    G.add_edges_from(edges_list)

    if len(edges_list)>0:
        max_weight= max([e[2]['weight'] for e in edges_list])
    else:
        max_weight=1





    #print nx.get_edge_attributes(G,'e_color')

    pos = nx.get_node_attributes(G,'pos')
    #print pos

    pos = nx.spring_layout(G, k=max_weight/8.0*k_scale, iterations=iters, pos=pos, center=[0,0], weight=weight_scale)#'weight')#fixed=fixed_nodes, 


    ### Draw Edges
    nx.draw_networkx_edges(G,
                        pos,
                        edgelist =G.edges(),
                        ax=ax,
                        edge_color=[nx.get_edge_attributes(G,'e_color')[e] for e in G.edges()],
                        alpha=0.2
                        )

    ### Draw Nodes
    nodes_ax = nx.draw_networkx_nodes(G,
                                pos,
                                ax=ax,
                                nodelist=G.nodes(),
                                node_size = [nx.get_node_attributes(G,'n_size')[n]*15 for n in G.nodes()],
                                node_color = [nx.get_node_attributes(G,'n_color')[n] for n in G.nodes()],
                                alpha= [nx.get_node_attributes(G,'n_alpha')[n] for n in G.nodes()]
                                )



    nodes_ax.set_edgecolor('w')

    ax.set_title('All Companies')


    ax.axis('off')

            
    plt.show()

    return None

def mse_sim(v1,v2):
    return 1.0-np.sqrt(np.sum(np.nan_to_num(((v1-v2)/np.amax([v1,v2], axis=0))**2))/4.)

def draw_basics(isos, wepp_dfs, years, iso_colors):
    ### 
    #n_units, #n_plants, #n_companies, #opr MW, %green_n, %blue, %ff, %MW_g/b/ff
    
    fig, axs = plt.subplots(3,3,figsize=(16,15))
    for iso in isos:
        n_units = [len(wepp_dfs[y][wepp_dfs[y].ISO==iso]) for y in years]
        n_plants = [len(wepp_dfs[y][wepp_dfs[y].ISO==iso].groupby('PLANT')) for y in years]
        n_companies = [len(wepp_dfs[y][wepp_dfs[y].ISO==iso].groupby('COMPANY')) for y in years]
        N_tot =dict(zip(years,[(wepp_dfs[y][wepp_dfs[y].ISO==iso].MW>10**-4).sum() for y in years]))
        green_n = [(wepp_dfs[y][wepp_dfs[y].ISO==iso].green_MW>10**-4).sum()/float(N_tot[y]) for y in years]
        blue_n = [(wepp_dfs[y][wepp_dfs[y].ISO==iso].blue_MW>10**-4).sum()/float(N_tot[y]) for y in years]
        ff_n = [(wepp_dfs[y][wepp_dfs[y].ISO==iso].ff_MW>10**-4).sum()/float(N_tot[y]) for y in years]
        MW_tot = dict(zip(years,[wepp_dfs[y][wepp_dfs[y].ISO==iso].MW.sum() for y in years]))
        green_MW = [wepp_dfs[y][wepp_dfs[y].ISO==iso].green_MW.sum()/MW_tot[y] for y in years]
        blue_MW = [wepp_dfs[y][wepp_dfs[y].ISO==iso].blue_MW.sum()/MW_tot[y] for y in years]
        ff_MW = [wepp_dfs[y][wepp_dfs[y].ISO==iso].ff_MW.sum()/MW_tot[y] for y in years]
        axs[0,0].plot(years, n_units, color=iso_colors[iso])
        axs[0,1].plot(years, n_plants, color=iso_colors[iso])
        axs[0,2].plot(years, n_companies, color=iso_colors[iso])
        axs[0,0].set_yscale("log", nonposy='clip')
        axs[0,1].set_yscale("log", nonposy='clip')
        axs[0,2].set_yscale("log", nonposy='clip')
        
        axs[1,0].plot(years, green_n, color=iso_colors[iso])
        axs[1,1].plot(years, blue_n, color=iso_colors[iso])
        axs[1,2].plot(years, ff_n, color=iso_colors[iso])
        axs[2,0].plot(years, green_MW, color=iso_colors[iso])
        axs[2,1].plot(years, blue_MW, color=iso_colors[iso])
        axs[2,2].plot(years, ff_MW, color=iso_colors[iso])
        
        axs[0,0].set_title('n_units')
        axs[0,1].set_title('n_plant')
        axs[0,2].set_title('n_companies')
        axs[1,0].set_title('units % green')
        axs[1,1].set_title('units % blue')
        axs[1,2].set_title('units % ff')
        axs[2,0].set_title('MW % green')
        axs[2,1].set_title('MW % blue')
        axs[2,2].set_title('MW % ff')
        
    custom_lines = [Line2D([0], [0], color=iso_colors[iso], lw=4, label=iso) for iso in isos]
    
    fig.legend(custom_lines,isos,loc='lower center', ncol=len(isos))
        
    plt.show()

def draw_top_companies(df_1yr, MW=False, N=10, co_list=None):
    
    if co_list is not None:
        top_cos = co_list
        N = len(co_list)
    else:    
        if MW:
            top_cos = df_1yr[['COMPANY','MW']].groupby('COMPANY').sum().sort_values('MW', ascending=False).index.values[0:N]
        
        else:
            top_cos = df_1yr[['COMPANY','MW']].groupby('COMPANY').count().sort_values('MW', ascending=False).index.values[0:N]
    

    fig, axs = plt.subplots(int(math.ceil(N/5.)),5,figsize=(16,int(math.ceil(N/5.))*4 ), dpi=100)
    
    #print 'axs shape',axs.shape
    
    if len(axs.shape)==1:
        axs = axs.reshape(-1,5)
    #print 'axs shape',axs.shape, len(top_cos),axs
    ii=0
    jj=0

    for c in top_cos:
        slice_company = df_1yr.loc[df_1yr['COMPANY']==c]
        #print slice_company

        # makem nodes

        nodes_list = []

        for plant in list(slice_company.index):
            g = int(slice_company.get_value(plant,'green_MW')/slice_company.get_value(plant,'MW')*255)
            b = int(slice_company.get_value(plant,'blue_MW')/slice_company.get_value(plant,'MW')*255)
            size=np.log10(slice_company.get_value(plant,'MW'))*10


            nodes_list.append(
                (plant,{
                    'pos': np.random.rand(2),
                    'type': 'power_station',
                    'n_color':"#{0:02x}{1:02x}{2:02x}".format(clamp(0), clamp(g), clamp(b)),
                    'n_alpha':1.0,
                    'n_size':size
                    })            
                )
            

        #print nodes_list

        # makem edges

        edges_list = []

        pi=0

        for plant1 in list(slice_company.index):

            pi+=1

            for plant2 in list(slice_company.index)[pi:]:

                if plant1 != plant2:
                    edges_list.append(
                        (plant1, plant2, {
                            'weight': 10.0*np.log10(min(slice_company.get_value(plant1,'MW'),slice_company.get_value(plant2,'MW'))),
                            'type': 'plants',
                            'e_color': 'gray',
                            'e_alpha': 1.0
                        })
                    )
                    
        


        ### make graph diagram

        G = nx.Graph()

        G.add_nodes_from(nodes_list)
        G.add_edges_from(edges_list)

        if len(edges_list)>0:
            max_weight= max([e[2]['weight'] for e in edges_list])
        else:
            max_weight=1


        pos = nx.get_node_attributes(G,'pos')
        #print pos
        

        pos = nx.spring_layout(G, k=max_weight/8.0, iterations=50, pos=pos, center=[0,0], weight='weight')#fixed=fixed_nodes, 
            ### Draw Edges
        nx.draw_networkx_edges(G,
                               pos,
                               edgelist =G.edges(),
                               ax=axs[jj,ii],
                               edge_color=[nx.get_edge_attributes(G,'e_color')[e] for e in G.edges()],
                               alpha=0.2
                              )

        ### Draw Nodes
        #print 'g nodes', G.nodes()

            
            
        nodes_ax = nx.draw_networkx_nodes(G,
                                          pos,
                                          ax=axs[jj,ii],
                                          nodelist=G.nodes(),
                                          node_size = [nx.get_node_attributes(G,'n_size')[n]*15 for n in G.nodes()],
                                          node_color = [nx.get_node_attributes(G,'n_color')[n] for n in G.nodes()],
                                          alpha= [nx.get_node_attributes(G,'n_alpha')[n] for n in G.nodes()]
                                          )
        #print nodes_ax



        nodes_ax.set_edgecolor('w')

        axs[jj,ii].set_title(c)



        ii+=1
        if ii==5:
            ii=0
            jj+=1
            
    for ii in range(int(math.ceil(N/5.))):
        for jj in range(5):
            axs[ii,jj].axis('off')
            
    plt.show()

    return None

def get_fitness(stats_a, stats_b, years):
    fitness = []
    for ii in range(len(years)-1):
        ii_fitness = 0
        for k,v in stats_a.iteritems():
            ii_fitness+= ((stats_b[k][ii]-v[ii]) / float(v[ii]))**2.0
        fitness.append(ii_fitness)
    return np.sqrt(np.array(fitness))

def plot_fitness(stats_a, stats_b, years):
    ff = get_fitness(stats_a, stats_b, years)
    fig,ax = plt.subplots(1,1, figsize=(6,4))
    ax.plot(years[1:],ff)
    ax.set_xticks(years[1:])
    ax.set_ylabel('Fitness - root sum square error')
    ax.set_title('Model Fitness')
    plt.show()


def plot_DD_line(iso_slice_meta, select_iso, iso_color):
    fig,axs = plt.subplots(1,2, figsize=(12,4.5))
    axs[0].plot(iso_slice_meta[2007].sort_values('COUNT', ascending=False).COUNT.values, color=iso_color, alpha=1.0, label='2007')
    axs[0].plot(iso_slice_meta[2012].sort_values('COUNT', ascending=False).COUNT.values, color=iso_color, alpha=0.7, label='2012')
    axs[0].plot(iso_slice_meta[2017].sort_values('COUNT', ascending=False).COUNT.values, color=iso_color, alpha=0.5, label='2017')
    axs[0].legend(); axs[0].set_xlabel('companies'); axs[0].set_ylabel('Degree - n_plants')
    axs[1].plot(iso_slice_meta[2007].sort_values('MW', ascending=False).MW.values, color=iso_color, alpha=1.0, label='2007')
    axs[1].plot(iso_slice_meta[2012].sort_values('MW', ascending=False).MW.values, color=iso_color, alpha=0.7, label='2012')
    axs[1].plot(iso_slice_meta[2017].sort_values('MW', ascending=False).MW.values, color=iso_color, alpha=0.5, label='2017')
    axs[1].legend(); axs[1].set_xlabel('companies'); axs[1].set_ylabel('Degree - MW')
    plt.suptitle('Degree Distributions - '+select_iso)
    plt.show()
    
def plot_DD_boxplot(df, label, years, MW = False):
    ### PLOT DEGREE DISTS WITH BOX & WHISK - COUNT
    
    if MW:
        cols_list = ['green_MW', 'blue_MW', 'ff_MW']
    else:
        cols_list = ['green','blue','ff']

    fig, axs = plt.subplots(1,3,figsize=(16,5), sharey=True, sharex=True)

    data_sup = {}


    N_sup = {}


    for c in cols_list:
        data_sup[c] = []

        for y in years[1:]:
            df_t = df[y][['COMPANY',c]].groupby('COMPANY').sum()

            data_sup[c].append(df_t[df_t[c]>10**-4][c].values)  #companies with positive counts for those assets

    for c in cols_list:
        #%companies that have that don't have that color asset
        N_sup[c] = [(1.0 - (float(ii)/jj)) for ii,jj in zip([len(qq) for qq in data_sup[c]],[len(df[y][['COMPANY',c]].groupby('COMPANY').sum()) for y in years[1:]])]

    _patches = {}


    _patches['green'] = axs[0].boxplot(data_sup[cols_list[0]],0,'g+', whis=[2.5,97.5], patch_artist=True, widths=0.8)
    _patches['blue'] = axs[1].boxplot(data_sup[cols_list[1]],0,'b+', whis=[2.5,97.5],patch_artist=True, widths=0.8)
    _patches['black'] = axs[2].boxplot(data_sup[cols_list[2]],0,'k+', whis=[2.5,97.5], patch_artist=True, widths=0.8)
    axs[0].set_yscale("log", nonposy='clip')
    

    if MW:
        axs[0].set_ylim(bottom=10**-4)
        axs[0].set_ylabel(label)

    else:
        axs[0].set_ylim(bottom=10**-0.5, top=10**2)
        axs[0].set_ylabel(label)
    
    
    axs[0].set_xticklabels([y for y in years[1:]])
    axs[1].set_xticklabels([y for y in years[1:]])
    axs[2].set_xticklabels([y for y in years[1:]])


    cols_dict = {
        'green':'#008000',
        'blue':'#4040ff',
        'black':'#404040',
    }
    colors = ['green', 'blue', 'black']
    for c in colors:
        for patch in _patches[c]['boxes']:
            patch.set_facecolor(cols_dict[c])
            
    if MW:
        txt_h = 10**-3.5
        axs[0].set_ylabel('Degree - MW')
    else:
        txt_h = 10**-0.3
        axs[0].set_ylabel('Degree - Count')
        
    for ia,c in enumerate(cols_list):
        for iy in range(len(years)-1):
            axs[ia].text(iy+1,txt_h,"{0:.0%}".format(N_sup[c][iy]), ha='center')

        
    axs[0].set_xticklabels([str(y) for y in years[1:]])
    axs[0].set_title('DD for companies with Green Assets')
    axs[1].set_title('DD for companies with Blue Assets')
    axs[2].set_title('DD for companies with FF Assets')
    plt.suptitle(label)
        
    plt.show()
    return None
    
def summarize_iso_slice(iso_slice, iso_slice_meta, iso_slice_plants, years):
    print '{:>5} {:>7} {:>7} {:>11} {:>12} {:>8} {:>7} {:>5} {:>9} {:>8} {:>6}'.format('year', 'n_units', 'n_plants', 'n_companies', 'total_OPR_MW', '%n_green', '%n_blue', '%n_ff', '%MW_green', '%MW_blue', '%MW_ff')
    print '----'*24

    for y in years:
        print '{:>5} {:>7} {:>7} {:>11} {:>12} {:>8} {:>7} {:>5} {:>9} {:>8} {:>6}'.format(
            y,
            len(iso_slice[y]),
            len(iso_slice_plants[y]),
            len(iso_slice_meta[y]),int(iso_slice_plants[y].MW.sum()), 
            int(iso_slice_plants[y].green.sum()),int(iso_slice_plants[y].blue.sum()), int(iso_slice_plants[y].ff.sum()),
            int(iso_slice_plants[y].green_MW.sum() / iso_slice_plants[y].MW.sum()*100.),
            int(iso_slice_plants[y].blue_MW.sum() / iso_slice_plants[y].MW.sum()*100.),
            int(iso_slice_plants[y].ff_MW.sum() / iso_slice_plants[y].MW.sum()*100.)        
            )
    
    
def plot_ternary(dfs, labels):
    
    ### DRAW TERNARY - ISOLATES - mean MW., DIADS; TRIADS; TOP 10 x 2 - count, MW, with lines for years
    fig,axs = plt.subplots(len(dfs),3,figsize=(16,len(dfs)*5))


    for ii,df in enumerate(dfs):
        for jj,y in enumerate([2008,2012,2017]):
            
            if len(dfs)==1:
                axs[jj].axis('off')
                tfig, tax = ternary.figure(ax=axs[jj], scale=100.0)
            else:
                axs[ii,jj].axis('off')
                tfig, tax = ternary.figure(ax=axs[ii,jj], scale=100.0)

            tax.boundary(linewidth=2.0)
            tax.gridlines(color='black', multiple=20)
            # Remove default Matplotlib Axes
            tax.clear_matplotlib_ticks()
            tax.set_title(str(y)+ ' - '+labels[ii], fontsize=12)
            tax.left_axis_label("FF", fontsize=12)
            tax.right_axis_label("BLUE", fontsize=12)
            tax.bottom_axis_label("GREEN", fontsize=12)

            tax.ticks(axis='lbr', multiple=20, linewidth=1)
        
            df_t = (df[y][['COMPANY','MW','green_MW', 'blue_MW','ff_MW']]
                    .groupby('COMPANY').filter(lambda x: x['MW'].count()==1)
                    .groupby('COMPANY').sum())
            n_isolates = len(df_t)
        
            ISOLATES_MEAN = (
                (df_t.mean().green_MW / df_t.mean().MW *100.0),
                (df_t.mean().blue_MW / df_t.mean().MW *100.0),
                (df_t.mean().ff_MW / df_t.mean().MW *100.0))
            g = int(ISOLATES_MEAN[0]/100.*255)
            b = int(ISOLATES_MEAN[1]/100.*255)
            
            #print ISOLATES_MEAN
            tax.scatter([ISOLATES_MEAN], marker='x', s=500, color="#{0:02x}{1:02x}{2:02x}".format(clamp(0), clamp(g), clamp(b)))#, label='Isolates Mean (N='+str(n_isolates)+')')
            
            DIADS = (df[y][['COMPANY','MW','green_MW', 'blue_MW','ff_MW']]
                    .groupby('COMPANY').filter(lambda x: x['MW'].count()==2)
                    .groupby('COMPANY').sum())
            DIADS['green_norm'] = DIADS.green_MW / DIADS.MW *100.0
            DIADS['blue_norm'] = DIADS.blue_MW / DIADS.MW * 100.0
            DIADS['ff_norm'] = DIADS.ff_MW / DIADS.MW * 100.0
        
            DIADS = DIADS.as_matrix(columns=['green_norm','blue_norm','ff_norm'])
        
            color_diads = []
            for ci in range(DIADS.shape[0]):
                g = int(DIADS[ci,0]/100.*255)
                b = int(DIADS[ci,1]/100.*255)
                color_diads.append("#{0:02x}{1:02x}{2:02x}".format(clamp(0), clamp(g), clamp(b)))
            n_diads = len(color_diads)
        
            #print DIADS
            tax.scatter(DIADS, marker='d', s=50, color=color_diads)#, label='Diads (N='+str(len(color_diads))+')')
        
        
            TRIADS = (df[y][['COMPANY','MW','green_MW', 'blue_MW','ff_MW']]
                    .groupby('COMPANY').filter(lambda x: x['MW'].count()==3)
                    .groupby('COMPANY').sum())
            TRIADS['green_norm'] = TRIADS.green_MW / TRIADS.MW *100.0
            TRIADS['blue_norm'] = TRIADS.blue_MW / TRIADS.MW * 100.0
            TRIADS['ff_norm'] = TRIADS.ff_MW / TRIADS.MW * 100.0
        
            TRIADS = TRIADS.as_matrix(columns=['green_norm','blue_norm','ff_norm'])
        
            color_triads = []
            for ci in range(TRIADS.shape[0]):
                g = int(TRIADS[ci,0]/100.*255)
                b = int(TRIADS[ci,1]/100.*255)
                color_triads.append("#{0:02x}{1:02x}{2:02x}".format(clamp(0), clamp(g), clamp(b)))
            n_triads = len(color_triads)
        
            #print DIADS
            tax.scatter(TRIADS, marker='1', s=100, color=color_triads)#, label='Triads (N='+str(len(color_triads))+')')
        
        
            ALL = (df[y][['COMPANY','MW','green_MW', 'blue_MW','ff_MW']]
                    .groupby('COMPANY').filter(lambda x: x['MW'].count()>3)
                    .groupby('COMPANY').sum())
        
            #print ALL
            ALL['green_norm'] = ALL.green_MW / ALL.MW *100.0
            ALL['blue_norm'] = ALL.blue_MW / ALL.MW * 100.0
            ALL['ff_norm'] = ALL.ff_MW / ALL.MW * 100.0
        
            s_all = ALL.as_matrix(columns=['MW'])
        
            ALL = ALL.as_matrix(columns=['green_norm','blue_norm','ff_norm'])
        
            #print s_all
        
            color_all = []
            for ci in range(ALL.shape[0]):
                g = int(ALL[ci,0]/100.*255)
                b = int(ALL[ci,1]/100.*255)
                color_all.append("#{0:02x}{1:02x}{2:02x}".format(clamp(0), clamp(g), clamp(b)))
            
            n_all = len(color_all)
        
            #print DIADS
        
            tax.scatter(ALL, marker='o', s=50, edgecolors=color_all, facecolors='None')#, markeredgewidth=1,markerfacecolor='None')#, label='All others (N='+str(len(color_all))+')')
        
            custom_markers = [Line2D([0], [0], color='gray', marker='x', lw=0,linestyle=None),
                    Line2D([0], [0], color='gray', lw=0,marker='d'),
                    Line2D([0], [0], color='gray', lw=0,marker='1'),
                    Line2D([0], [0], color='gray', lw=0,marker='o'),]
        
            leg_labels = [
                'N_isolates = '+str(n_isolates),
                'N_diads = '+str(n_diads),
                'N_triads = '+str(n_triads),
                'N_all>r3 = '+str(n_all),
            ]
        
        
        
            tax.legend(custom_markers, leg_labels, loc='upper right')
        
    plt.show()
    
def plot_n_clusters(iso_slice_plants, years, thresholds):
    
    fig,ax = plt.subplots(1,1,figsize=(6,6))
    
    for thresh in thresholds:
        
        thresh_list = []
    
        for y in years:

            k_cluster = iso_slice_plants[y][['COMPANY','MW','green_MW','blue_MW','ff_MW','green','blue','ff']].groupby(['COMPANY']).sum()
            k_cluster['green_avg'] = iso_slice_plants[y][['COMPANY','green_MW']].groupby(['COMPANY']).mean()
            k_cluster['blue_avg'] = iso_slice_plants[y][['COMPANY','blue_MW']].groupby(['COMPANY']).mean()
            k_cluster['ff_avg'] = iso_slice_plants[y][['COMPANY','ff_MW']].groupby(['COMPANY']).mean()
            k_cluster['tot_avg'] = iso_slice_plants[y][['COMPANY','MW']].groupby(['COMPANY']).mean()
            k_cluster['tot_count'] = k_cluster[['green','blue','ff']].sum(axis=1)
            k_cluster.rename(columns={'MW':'tot_MW',
                                      'green':'green_count',
                                      'blue':'blue_count',
                                      'ff':'ff_count',
                                     }, inplace=True)


            k_cluster_std =  s.zscore(k_cluster[list(k_cluster)])

            r_inertia = 0.
            k=1
            kmeans = KMeans(n_clusters=k, random_state=0).fit(k_cluster_std)

            while (k<40) and (r_inertia<(1.0-thresh)):
                k+=1
                try_kmeans = KMeans(n_clusters=k, random_state=0).fit(k_cluster_std)
                r_inertia = try_kmeans.inertia_/kmeans.inertia_
                kmeans = try_kmeans

            
            thresh_list.append(k-1)
        
        ax.plot(years,thresh_list,label=str(thresh))
        ax.legend(loc='upper right')
        ax.set_xticks(years)
        ax.set_ylabel('N_clusters')
        ax.set_title('Number of K-means clusters')
        
    plt.show()
    
def get_clusters(iso_slice_plants, years, threshold):
    cluster_stats = {}
    cluster_dfs = {}

    for y in years:

        k_cluster = iso_slice_plants[y][['COMPANY','MW','green_MW','blue_MW','ff_MW','green','blue','ff']].groupby(['COMPANY']).sum()
        k_cluster['green_avg'] = iso_slice_plants[y][['COMPANY','green_MW']].groupby(['COMPANY']).mean()
        k_cluster['blue_avg'] = iso_slice_plants[y][['COMPANY','blue_MW']].groupby(['COMPANY']).mean()
        k_cluster['ff_avg'] = iso_slice_plants[y][['COMPANY','ff_MW']].groupby(['COMPANY']).mean()
        k_cluster['tot_avg'] = iso_slice_plants[y][['COMPANY','MW']].groupby(['COMPANY']).mean()
        k_cluster['tot_count'] = k_cluster[['green','blue','ff']].sum(axis=1)
        k_cluster.rename(columns={'MW':'tot_MW',
                                  'green':'green_count',
                                  'blue':'blue_count',
                                  'ff':'ff_count',
                                 }, inplace=True)


        k_cluster_std =  s.zscore(k_cluster[list(k_cluster)])

        r_inertia = 0.
        k=1
        kmeans = KMeans(n_clusters=k, random_state=0).fit(k_cluster_std)

        while (k<40) and (r_inertia<(1.0-threshold)):
            k+=1
            try_kmeans = KMeans(n_clusters=k, random_state=0).fit(k_cluster_std)
            r_inertia = try_kmeans.inertia_/kmeans.inertia_
            kmeans = try_kmeans


        kmeans = KMeans(n_clusters=k-1, random_state=0).fit(k_cluster_std)
        labels = kmeans.labels_

        #Glue back to originaal data
        k_cluster['clusters'] = labels
        
        #print k_cluster


        cluster_stats[y] = k_cluster.groupby(['clusters']).mean()
        cluster_stats[y]['COUNT'] = k_cluster.groupby(['clusters']).count().tot_count
        #cluster_dfs[y] = k_cluster
        cluster_dfs[y] = iso_slice_plants[y].merge(pd.DataFrame(k_cluster.clusters), how='left', left_on='COMPANY',right_index=True)
        #print cluster_dfs[y]

    for y in years:
        #print cluster_stats[y].sort_values(['COUNT','tot_MW'], ascending=False)#.index.values
        di = dict(zip(cluster_stats[y].sort_values(['COUNT','tot_MW'], ascending=False).index.values,range(len(cluster_stats[y]))))

        cluster_dfs[y]['clusters'] = cluster_dfs[y].clusters.map(di)
        
        company_cluster = cluster_dfs[y][['COMPANY','MW','green_MW','blue_MW','ff_MW','green','blue','ff']].groupby(['COMPANY']).sum()
        
        company_cluster['green_avg'] = cluster_dfs[y][['COMPANY','green_MW']].groupby(['COMPANY']).mean()
        company_cluster['blue_avg'] = cluster_dfs[y][['COMPANY','blue_MW']].groupby(['COMPANY']).mean()
        company_cluster['ff_avg'] = cluster_dfs[y][['COMPANY','ff_MW']].groupby(['COMPANY']).mean()
        company_cluster['tot_avg'] = cluster_dfs[y][['COMPANY','MW']].groupby(['COMPANY']).mean()
        company_cluster['clusters'] = cluster_dfs[y][['COMPANY','clusters']].groupby(['COMPANY']).nth(0)
        company_cluster['tot_count'] = company_cluster[['green','blue','ff']].sum(axis=1)
        company_cluster.rename(columns={'MW':'tot_MW',
                                  'green':'green_count',
                                  'blue':'blue_count',
                                  'ff':'ff_count',
                                 }, inplace=True)
        cluster_stats[y] = company_cluster.groupby(['clusters']).mean()
        cluster_stats[y]['COUNT'] = company_cluster.groupby(['clusters']).count().tot_MW
        


        #cluster_stats[y] = cluster_dfs[y].groupby(['clusters']).mean()
        #cluster_stats[y]['COUNT'] = cluster_dfs[y].groupby(['clusters']).count().MW
        
    return cluster_stats, cluster_dfs

def draw_cluster_transitions(cluster_dfs,cluster_stats, years):

    fix,ax = plt.subplots(1,1,figsize=(16,16))

    for y in years[0:-1]:
        for c in cluster_stats[y].index.values:
            #print c, len(cluster_dfs[y+1][cluster_dfs[y+1].index.isin(cluster_dfs[y][cluster_dfs[y].clusters==c].index)]), cluster_dfs[y].loc[cluster_dfs[y].clusters==c,'tot_MW'].sum()
            c_slice = cluster_dfs[y+1][cluster_dfs[y+1].index.isin(cluster_dfs[y][cluster_dfs[y].clusters==c].index)]
            slice_mat = c_slice.groupby(['clusters']).sum()
            for ii in slice_mat.index.values:
                x = [y,y+1]
                z = [c,ii]
                #width = 5.*c_slice.groupby(['clusters']).sum().loc[ii,'MW']/cluster_dfs[y].loc[cluster_dfs[y].clusters==c,'MW'].sum()
                width = np.sqrt(c_slice.groupby(['clusters']).sum().loc[ii,'MW'])/7.#/cluster_dfs[y].loc[cluster_dfs[y].clusters==c,'MW'].sum()

                #print c, ii, width
                g = int(c_slice.groupby(['clusters']).sum().loc[ii,'green_MW'] / c_slice.groupby(['clusters']).sum().loc[ii,'MW'] * 255.)
                b = int(c_slice.groupby(['clusters']).sum().loc[ii,'blue_MW'] / c_slice.groupby(['clusters']).sum().loc[ii,'MW'] * 255.)
                #"#{0:02x}{1:02x}{2:02x}".format(clamp(0), clamp(g), clamp(b))
                ax.plot(x,z, color="#{0:02x}{1:02x}{2:02x}".format(clamp(0), clamp(g), clamp(b)), linewidth=width, alpha=0.2, zorder=1)

    patches = []        
    for y in years[0:-1]:
        for c in cluster_stats[y].index.values:

            #print ~(cluster_dfs[y][cluster_dfs[y].clusters==c].index.isin(cluster_dfs[y+1].index))

            retire_slice = cluster_dfs[y][(cluster_dfs[y].clusters==c) & ~(cluster_dfs[y].index.isin(cluster_dfs[y+1].index))]
            if len(retire_slice)>0:
                w_tot = retire_slice.MW.sum()
                w_green = retire_slice.green_MW.sum()
                w_blue = retire_slice.blue_MW.sum()
                w_ff = retire_slice.ff_MW.sum()
                w_center = (y+0.1,c)
                #print w_tot
                w_size = w_tot/8000.

                patches.append(Wedge(w_center,w_size,-90,  -90.+w_ff/w_tot*180,color='black', alpha=1., zorder=2))
                patches.append(Wedge(w_center,w_size,-90+w_ff/w_tot*180,  -90.+w_ff/w_tot*180+w_blue/w_tot*180,color='#0000ff', alpha=1., zorder=2))
                patches.append(Wedge(w_center,w_size,-90+w_ff/w_tot*180+w_blue/w_tot*180,  -90.+w_blue/w_tot*180+w_green/w_tot*180+w_ff/w_tot*180,color='#00ff00', alpha=1.,zorder=2))

            if y>years[0]:
                birth_slice = cluster_dfs[y][(cluster_dfs[y].clusters==c) & ~(cluster_dfs[y].index.isin(cluster_dfs[y-1].index))]
                if len(birth_slice)>0:
                    w_tot = birth_slice.MW.sum()
                    w_green = birth_slice.green_MW.sum()
                    w_blue = birth_slice.blue_MW.sum()
                    w_ff = birth_slice.ff_MW.sum()
                    w_center = (y-0.1,c)
                    #print w_tot
                    w_size = w_tot/8000.

                    patches.append(Wedge(w_center,w_size,90,  90.+w_ff/w_tot*180,color='black', alpha=1.,zorder=2))
                    patches.append(Wedge(w_center,w_size,90+w_ff/w_tot*180,  90.+w_ff/w_tot*180+w_blue/w_tot*180,color='#0000ff', alpha=1.,zorder=2))
                    patches.append(Wedge(w_center,w_size,90+w_ff/w_tot*180+w_blue/w_tot*180,  90.+w_blue/w_tot*180+w_green/w_tot*180+w_ff/w_tot*180,color='#00ff00', alpha=1.,zorder=2))

    patches.append(Wedge((2015.6,15),0.5,-90,  -30.,color='black', alpha=1.))
    patches.append(Wedge((2015.6,15),0.5,-30,  30.,color='#0000ff', alpha=1.))
    patches.append(Wedge((2015.6,15),0.5,30,  90.,color='#00ff00', alpha=1.))


    patches.append(Wedge((2015.4,15),0.5,90,  150.,color='black', alpha=1.))
    patches.append(Wedge((2015.4,15),0.5,150,  210,color='#0000ff', alpha=1.))
    patches.append(Wedge((2015.4,15),0.5,210,  270,color='#00ff00', alpha=1.))

    ax.text(2015.4,16,'Births', horizontalalignment='right')
    ax.text(2015.6,16,'Retirements')

    P = PatchCollection(patches, match_original=True)

    ax.add_collection(P)
    ax.set_xticks(years)
    ax.set_ylabel('Clusters')

    plt.show()



def random_choice_prob_index(a, axis=1):
    r = np.expand_dims(np.random.rand(a.shape[1-axis]), axis=axis)
    return (a.cumsum(axis=axis) > r).argmax(axis=axis)