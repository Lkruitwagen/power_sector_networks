import matplotlib.pyplot as plt
import networkx as nx
import random
import math
import pandas as pd
import statsmodels.api as sm
import glob
import os
import numpy as np
from PIL import Image
import helpers
import pickle
import time

""" some helpers """
def format_hr(row):
    if row['MW']<=400:
        row_MW = '0-400'
    elif row['MW']<=500:
        row_MW = '401-500'
    elif row['MW']<=700:
        row_MW = '501-700'
    elif row['MW']<=900:
        row_MW = '701-900'
    elif row['MW']>900:
        row_MW = '901-10000'


    if not row['fuel_class'] in ['COAL','OIL','GAS']:
        return np.nan
    elif row['fuel_class']=='COAL':
        #no fuel type or stype
        if row['FUELTYPE']=='' or pd.isnull(row['FUELTYPE']):
            row_ftype = 'BIT'
        elif row['FUELTYPE'].split('/')[0].upper() == 'ANTH':
            row_ftype = 'BIT'
        elif row['FUELTYPE'].split('/')[0].upper() in ['BIT','SUB','LIG']:
            row_ftype = row['FUELTYPE'].split('/')[0].upper()
        else:
            row_ftype = 'BIT'

        if row['STYPE']=='' or pd.isnull(row['STYPE']):
            row_stype = 'SUBCR'
        else:
            row_stype = row['STYPE']
        try:
            return row['fuel_class']+'+'+row_ftype.upper()+'+'+row_stype.upper()+'+'+str(row_MW)
        except:
            print row['fuel_class'], row_ftype, row_stype, row_MW
            exit()
    else: #oil or gas:
        if row['UTYPE'].split('/')[0].upper() not in ['ST','GT','IC','CC']:
            row_utype = 'ST'
        else:
            row_utype = row['UTYPE'].split('/')[0].upper()
        try:
            return row['fuel_class']+'+'+row_utype
        except:
            print row['fuel_class'], row_utype
            exit()

def get_hr(row):
    if row['YEAR']<1949:
        return row[1949]
    elif row['YEAR']>2016:
        return row[2016]
    else:
        return row[int(row['YEAR'])]

def fill_year(row, est, cols):
    test = pd.DataFrame(np.zeros((1,len(cols))),columns = cols)
    #print row
    for k,v in row.items():
        #print k,v
        if not pd.isnull(v):
            test[k+'_'+v] = 1
    #print test
    test = sm.add_constant(test, has_constant='add')
    #print est.predict(test)
    return est.predict(test)

    #print pd.get_dummies(row)
    #inp = pd.get_dummies(row.T).reindex(columns=cols, fill_value = 0)
    #inp = sm.add_constant(inp)
    #print inp.T
    #print est.predict(inp)
    #return est.predict(inp)
    
def clamp(x): 
    return max(0, min(x, 255))

""" get our wepp db """
def get_wepp(folder):
    files = glob.glob(os.path.join(folder,'*'))
    for file in files:
        directory, fname = os.path.split(file)
        ext = fname.split('.')[-1]
        fbase = fname.split('.')[0]
        print ext, fbase

        if fbase.upper() == 'ALLUNITS':
            if ext == 'csv':
                return pd.read_csv(file, encoding='latin1')
            elif ext == 'xls' or 'xlsx':
                return pd.read_excel(file,encoding='latin1')