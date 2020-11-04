# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 14:08:41 2020

@author: nwillems
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def assemble_test(base_schema):  

    #out_beta = pd.read_sql(sql % ('diffusion_results_' + base_schema + '_beta'), con)
    out_beta = pd.read_csv(base_schema + '_beta.csv')
    out_exp = pd.read_csv(base_schema + '_adv.csv')
    #out_prop = pd.read_csv(base_schema + '_prop.csv')
    
    historic_county = pd.read_csv('installed_count_capacity_kw_by_county_sector.csv', index_col=0)
    historic_county = historic_county.query("state_abbr in @out_exp.state_abbr.unique()")[['state_abbr','county_id','fips','sector_abbr','year','imputed_cum_count','imputed_cum_size_kW']]
    
    
    out = out_beta.groupby(['state_abbr','county_id','sector_abbr','year']).agg(size_kW_beta=('system_kw_cum','sum')).join(
            [out_exp.groupby(['state_abbr','county_id','sector_abbr','year']).agg(size_kW_adv=('system_kw_cum','sum'))#,
             #out_prop.groupby(['state_abbr','county_id','sector_abbr','year']).agg(size_kW_prop=('system_kw_cum','sum'))
             ]).reset_index()
    
    
    df = pd.merge(historic_county.replace({'sector_abbr': 'commercial'}, 'com'), out,
                  on=['state_abbr','county_id','sector_abbr','year'])
    
    return df

####****---- END OF FUNCTION assemble_test ----****####
# ==================================================================================================================== #


def scorer(test_set, level, score_type="all", forecast_years=1):
    # Test the goodness of fit
    test_set.sort_values(by=['year'], inplace=True)
    
    fit_years = test_set.year.unique()[:-forecast_years]
    if score_type=="fit":
        test_set = test_set[test_set.year.isin(fit_years)]
    elif score_type=="prediction":
        test_set = test_set[~test_set.year.isin(fit_years)]
    
    if level=="fips":
        y_true = test_set.groupby(['fips','sector_abbr'])['true'].apply(list)
        y_pred = test_set.groupby(['fips','sector_abbr'])['pred'].apply(list)
    elif level =="state":
        y_true = test_set.groupby(['state_abbr','sector_abbr'])['true'].apply(list)
        y_pred = test_set.groupby(['state_abbr','sector_abbr'])['pred'].apply(list)
    else:
        level="group"
        y_true = test_set.groupby(['group','sector_abbr'])['true'].apply(list)
        y_pred = test_set.groupby(['group','sector_abbr'])['pred'].apply(list)

    
    print(score_type, 'scores for', level, 'level')
    keys = y_pred.index.get_level_values(level).unique()

    scores = []
    for key in keys:
        for sector in y_pred.index.get_level_values('sector_abbr').unique():
            true_arr = np.array(y_true.loc[key, sector])
            pred_arr = np.array(y_pred.loc[key, sector])

            ## Mean Square Error
            mse = mean_squared_error(true_arr, pred_arr)          
            
            ## Relative Percent Difference
            rpd = np.where((true_arr==0) & (pred_arr==0), 0, 2*(pred_arr - true_arr) / (pred_arr + true_arr)) # 0/0 = 0              
            
            if level=="fips":
                scores.append([key, sector, mse, rpd.mean()])
                #scores.append([*list(y_pred.loc[key,sector].index), key, sector, mse, rpd.mean()])
            elif level=="group":
                scores.append([key, sector, mse, rpd.mean()])


    print('Compiling dataframe of', level, 'level', score_type, 'scores')
    ## Composite Mean Squared Error
    if level =="fips":
        scores = pd.DataFrame(data=scores, columns=['fips','sector_abbr','mse','rpd'])

        #state level RMSE
        state_RMSE = test_set.groupby(['state_abbr','year','sector_abbr'], as_index=False).agg({'true':'sum','pred':'sum'})
        state_RMSE = {group: mean_squared_error(vals['true'], vals['pred'])   
                        for group, vals in state_RMSE.sort_values(by='year').groupby(['state_abbr','sector_abbr'])}
        state_RMSE = pd.DataFrame(state_RMSE.keys(), columns=['state_abbr','sector_abbr']).assign(smse=state_RMSE.values())
        state_RMSE = state_RMSE.merge(test_set[['state_abbr','fips','sector_abbr']].drop_duplicates(), on=['state_abbr','sector_abbr'], how='right')
        scores = scores.merge(state_RMSE, on=['fips','sector_abbr'], how='left')
        scores['crmse'] = np.sqrt((scores['mse'] + scores['smse'])/2)
    elif level=="group":
        scores = pd.DataFrame(data=scores, columns=['group','sector_abbr','mse','rpd'])
    
    scores['rmse'] = np.sqrt(scores['mse'])
    
    return scores

####****---- END OF FUNCTION scorer ----****####
# ==================================================================================================================== #


def propensity_eval(base_schema):
    coeff = pd.read_csv('../runs/' + 'results_' + base_schema + '_prop/' + 'propensities.csv')
    val = pd.concat([(coeff != 0).sum(axis=0).rename('num'), coeff.sum(axis=0).rename('mean')], axis=1)
    val['mean'] = val['mean'] / np.maximum(val['num'], 1)
    val.sort_values(by="num", ascending=False, inplace=True)
    
    return (coeff, val)

####****---- END OF FUNCTION propensity_eval ----****####
# ==================================================================================================================== #

