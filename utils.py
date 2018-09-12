import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import random, math, glob, os, pickle, time, copy, ternary
import pandas as pd
import statsmodels.api as sm
import scipy.stats as s
import numpy as np
from PIL import Image
from sklearn import linear_model
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
    fuel_class = 'fuel_classification_database.dta'
    df_fuel_class = pd.io.stata.read_stata(fuel_class)
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

    wepp_df['green']=wepp_df.fuel_class.isin(['SUN','BIOGAS','WASTE','BIOOIL','WIND','BIOMASS','GEOTHERMAL'])
    wepp_df['green_MW'] = wepp_df.MW*wepp_df.green
    wepp_df['blue']=wepp_df.fuel_class.isin(['WATER','NUCLEAR'])
    wepp_df['blue_MW'] = wepp_df.MW*wepp_df.blue
    wepp_df['solar']=wepp_df.fuel_class.isin(['SUN'])
    wepp_df['solar_MW'] = wepp_df.MW*wepp_df.solar
    wepp_df['wind']=wepp_df.fuel_class.isin(['WIND'])
    wepp_df['wind_MW'] = wepp_df.MW*wepp_df.wind
    wepp_df['ff']=~wepp_df.fuel_class.isin(['SUN','BIOGAS','WASTE','BIOOIL','WIND','BIOMASS','GEOTHERMAL','WATER','NUCLEAR'])
    wepp_df['ff_MW'] = wepp_df.MW*wepp_df.ff
    
    return wepp_df

def prep_iso_slices(wepp_df, select_iso):
    prep_slice = wepp_df[wepp_df.ISO==select_iso]
    
    #group the units by plant, add the 
    prep_slice_plants = prep_slice[['PLANT','COMPANY']].groupby(['PLANT']).agg(lambda x:x.value_counts().index[0])

    prep_slice.loc[pd.isnull(prep_slice.fuel_class),['fuel_class']]='OIL'

    prep_slice_plants['fuel_class'] = prep_slice[['PLANT','fuel_class']].groupby(['PLANT']).agg(lambda x:x.value_counts().index[0])
    prep_slice_plants['YEAR'] = prep_slice[['PLANT','YEAR']].groupby(['PLANT']).mean()

    for c in ['green','blue','solar','wind','ff']:
        prep_slice_plants[c] = prep_slice[['PLANT',c]].groupby(['PLANT']).agg(lambda x:x.value_counts().index[0])

    for c in ['MW','CCCE','green_MW','blue_MW','solar_MW','wind_MW','ff_MW']:
        prep_slice_plants[c] = prep_slice[['PLANT',c]].groupby(['PLANT']).sum()
    
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

def plot_DD(iso_slice_plants, null_df, synthetic_df, years, MW = False):
    ### PLOT DEGREE DISTS WITH BOX & WHISK - COUNT
    
    if MW:
        cols_list = ['green_MW', 'blue_MW', 'ff_MW']
    else:
        cols_list = ['green','blue','ff']

    fig, axs = plt.subplots(3,3,figsize=(16,9), sharey=True, sharex=True)

    data_true = {}
    data_synth = {}
    data_null = {}

    N_t = {}
    N_s = {}
    N_n = {}

    for c in cols_list:
        data_true[c] = []
        data_synth[c] = []
        data_null[c] = []
        for y in years[1:]:
            df_t = iso_slice_plants[y][['COMPANY',c]].groupby('COMPANY').sum()
            df_s = synthetic_df[y][['COMPANY',c]].groupby('COMPANY').sum()
            df_n = null_df[y][['COMPANY',c]].groupby('COMPANY').sum()
            data_true[c].append(df_t[df_t[c]>10**-4][c].values)
            data_synth[c].append(df_s[df_s[c]>10**-4][c].values)
            data_null[c].append(df_n[df_n[c]>10**-4][c].values)  #companies with positive counts for those assets

    for c in cols_list:
        #%companies that have that don't have that color asset
        N_t[c] = [(1.0 - (float(ii)/jj)) for ii,jj in zip([len(qq) for qq in data_true[c]],[len(iso_slice_plants[y][['COMPANY',c]].groupby('COMPANY').sum()) for y in years[1:]])]
        N_s[c] = [(1.0 - (float(ii)/jj)) for ii,jj in zip([len(qq) for qq in data_synth[c]],[len(synthetic_df[y][['COMPANY',c]].groupby('COMPANY').sum()) for y in years[1:]])]
        N_n[c] = [(1.0 - (float(ii)/jj)) for ii,jj in zip([len(qq) for qq in data_null[c]],[len(null_df[y][['COMPANY',c]].groupby('COMPANY').sum()) for y in years[1:]])]


    t_patches = {}
    s_patches = {}
    n_patches = {}

    t_patches['green'] = axs[0,0].boxplot(data_true[cols_list[0]],0,'g+', whis=[2.5,97.5], patch_artist=True, widths=0.8)
    t_patches['blue'] = axs[0,1].boxplot(data_true[cols_list[1]],0,'b+', whis=[2.5,97.5],patch_artist=True, widths=0.8)
    t_patches['black'] = axs[0,2].boxplot(data_true[cols_list[2]],0,'k+', whis=[2.5,97.5], patch_artist=True, widths=0.8)
    axs[0,0].set_yscale("log", nonposy='clip')
    
    

    #axs[2,1].set_xticklabels([str(y) for y in years[1:]])
    #axs[2,2].set_xticklabels([str(y) for y in years[1:]])
    s_patches['green'] = axs[1,0].boxplot(data_synth[cols_list[0]],0,'g+', whis=[2.5,97.5],patch_artist=True, widths=0.8)
    s_patches['blue'] = axs[1,1].boxplot(data_synth[cols_list[1]],0,'b+', whis=[2.5,97.5],patch_artist=True, widths=0.8)
    s_patches['black'] = axs[1,2].boxplot(data_synth[cols_list[2]],0,'k+', whis=[2.5,97.5], patch_artist=True, widths=0.8)
    
    n_patches['green'] = axs[2,0].boxplot(data_null[cols_list[0]],0,'g+', whis=[2.5,97.5],patch_artist=True, widths=0.8)
    n_patches['blue'] = axs[2,1].boxplot(data_null[cols_list[1]],0,'b+', whis=[2.5,97.5],patch_artist=True, widths=0.8)
    n_patches['black'] = axs[2,2].boxplot(data_null[cols_list[2]],0,'k+', whis=[2.5,97.5], patch_artist=True, widths=0.8)
    
    if MW:
        axs[0,0].set_ylim(bottom=10**-4)
        axs[0,0].set_ylabel('True Degree - MW')
        axs[1,0].set_ylabel('Synthetic Degree - MW')
        axs[2,0].set_ylabel('Null Model - MW')
    else:
        axs[0,0].set_ylim(bottom=10**-0.5, top=10**2)
        axs[0,0].set_ylabel('True Degree - Count')
        axs[1,0].set_ylabel('Synthetic Degree - Count')
        axs[2,0].set_ylabel('Null Model - Count')
    
    
    axs[2,0].set_xticklabels([y for y in years[1:]])
    axs[2,1].set_xticklabels([y for y in years[1:]])
    axs[2,2].set_xticklabels([y for y in years[1:]])


    cols_dict = {
        'green':'#008000',
        'blue':'#4040ff',
        'black':'#404040',
    }
    colors = ['green', 'blue', 'black']
    for c in colors:
        for patch in t_patches[c]['boxes']:
            patch.set_facecolor(cols_dict[c])
        
        for patch in s_patches[c]['boxes']:
            patch.set_facecolor(cols_dict[c])
        
        for patch in n_patches[c]['boxes']:
            patch.set_facecolor(cols_dict[c])
            
    if MW:
        txt_h = 10**-3.5
    else:
        txt_h = 10**-0.3
        
    for ia,c in enumerate(cols_list):
        for iy in range(len(years)-1):
            axs[0,ia].text(iy+1,txt_h,"{0:.0%}".format(N_t[c][iy]), ha='center')
            axs[1,ia].text(iy+1,txt_h,"{0:.0%}".format(N_s[c][iy]), ha='center')
            axs[2,ia].text(iy+1,txt_h,"{0:.0%}".format(N_n[c][iy]), ha='center')

        
    axs[2,0].set_xticklabels([str(y) for y in years[1:]])
        
        
    #axs[1,0].set_yscale("log", nonposy='clip')
    #axs[1].set_ylim(bottom=10**-3, top=10**4)

    plt.savefig('./figures/growth_model/DD_count_n1b2-log10-eps.png')
    plt.show()
    return 0
    
    
def plot_ternary(iso_slice_plants, synthetic_df, null_df):
    
    ### DRAW TERNARY - ISOLATES - mean MW., DIADS; TRIADS; TOP 10 x 2 - count, MW, with lines for years
    fig,axs = plt.subplots(3,3,figsize=(16,16))
    model_dict = {
        0:'True Data',
        1:'Synthetic Model',
        2:'Null Model'
    }


    for ii,df in enumerate([iso_slice_plants, synthetic_df, null_df]):
        for jj,y in enumerate([2008,2012,2017]):
            axs[ii,jj].axis('off')
            tfig, tax = ternary.figure(ax=axs[ii,jj], scale=100.0)

            tax.boundary(linewidth=2.0)
            tax.gridlines(color='black', multiple=20)
            # Remove default Matplotlib Axes
            tax.clear_matplotlib_ticks()
            tax.set_title(str(y)+ ' - '+model_dict[ii], fontsize=12)
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
        
            labels = [
                'N_isolates = '+str(n_isolates),
                'N_diads = '+str(n_diads),
                'N_triads = '+str(n_triads),
                'N_all>r3 = '+str(n_all),
            ]
        
        
        
            tax.legend(custom_markers, labels, loc='upper right')
        
    plt.show()
    
    
def random_choice_prob_index(a, axis=1):
    r = np.expand_dims(np.random.rand(a.shape[1-axis]), axis=axis)
    return (a.cumsum(axis=axis) > r).argmax(axis=axis)