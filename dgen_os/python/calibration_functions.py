# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 16:15:44 2020

@author: nwillems
"""

import config
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso


#simple bass curve to fit nonlinear regression to
def differential_Bass(x, p, q, mms):
    #x is a vector of market share at time t (Cumulative installs * Max market size)
    #y is a vector of market share at time t+1
    return np.clip(-q*x*x + (1+q-p)*x + p, 0, mms)

####****---- END OF FUNCTION differential_Bass ----****####
# ==================================================================================================================== #


# calibrate p & q values for the differential_Bass function
def calibrate_Bass(df_grouped):
    market = df_grouped.groupby(['group','year','sector_abbr'], as_index=False).agg({'customers_in_bin_initial':'sum', 'number_of_adopters':'sum', 'pct_of_bldgs_developable':'mean'})
    market['f'] = np.clip(market['number_of_adopters'] / market['customers_in_bin_initial'], 0, 1) 

    fitted_Bass = []
    for group in market.group.unique():
        data = market.query("group==@group").sort_values(by="year")
        for sector in data.sector_abbr.unique():
            xdata = data.query("sector_abbr==@sector & year<year.max()")['f']
            ydata = data.query("sector_abbr==@sector & year>year.min()")['f']
            mms = data.query("sector_abbr==@sector & year>year.min()")['pct_of_bldgs_developable'].to_numpy()
    
            [p,q], _ = curve_fit(lambda x, p, q: differential_Bass(x, p, q, mms=mms), xdata, ydata, bounds=(1e-5, [0.2, 0.8]))

            fitted_Bass.append([group, sector, p, q])
    
    fitted_Bass = pd.DataFrame(data=fitted_Bass, columns=['group','sector_abbr','bass_param_p','bass_param_q'])
    
    fitted_Bass['teq_yr1'] = 1
    fitted_Bass['tech'] = 'solar'
    
    return fitted_Bass

####****---- END OF FUNCTION calibrate_Bass ----****####
# ==================================================================================================================== #


# define groups based on the data in agent_attr
def market_grouper(agent_attr, df, grouping_method, nclusters = 20, kmeans_vars=[], exclude_zeros=True, verbose=False):
    
    state_capacity_total = (df[['state_abbr', 'sector_abbr', 'pgid', 'developable_roof_sqft']].groupby(['state_abbr', 'sector_abbr'])
                                                                        .agg({'developable_roof_sqft':'sum', 'pgid':'count'})
                                                                        .rename(columns={'developable_roof_sqft':'state_developable_roof_sqft', 'pgid':'agent_count'})
                                                                        .reset_index())
    
    # coerce dtypes
    state_capacity_total.state_developable_roof_sqft = state_capacity_total.state_developable_roof_sqft.astype(np.float64) 
    df.developable_roof_sqft = df.developable_roof_sqft.astype(np.float64)
    df.pct_of_bldgs_developable = df.pct_of_bldgs_developable.astype(np.float64)
    
    # merge state totals back to agent df
    df = pd.merge(df, state_capacity_total, how = 'left', on = ['state_abbr', 'sector_abbr'])
    
    ## Bring in market data for calibration    
    historical_state_capacity_df = pd.read_csv(config.INSTALLED_CAPACITY_BY_STATE)
    #historical_county_capacity_df = pd.read_csv(config.INSTALLED_CAPACITY_BY_COUNTY)
    
    # join historical data to agent df
    df = pd.merge(df, historical_state_capacity_df, how='left', on=['state_abbr', 'sector_abbr'])
    
    # calculate scale factor - weight that is given to each agent based on proportion of state total
    # where state cumulative capacity is 0, proportion evenly to all agents
    df['scale_factor'] =  np.where(df['state_developable_roof_sqft'] == 0, 1.0/df['agent_count'], df['developable_roof_sqft'] / df['state_developable_roof_sqft'])
    
    # use scale factor to constrain agent capacity values to historical values
    df['system_kw_cum'] = df['scale_factor'] * df['observed_capacity_mw'] * 1000.
    
    # recalculate number of adopters using anecdotal values
    df['number_of_adopters'] = np.where(df['sector_abbr'] == 'res', df['system_kw_cum']/5.0, df['system_kw_cum']/100.0)

    # recalculate market share
    #df['market_share'] = np.where(df['developable_agent_weight'] == 0, 0.0, 
     #                  df['number_of_adopters'] / df['developable_agent_weight'])
    #df['market_share'] = df['market_share'].astype(np.float64)
    
    df.drop(columns=['agent_count', 'state_developable_roof_sqft', 'state', 'observed_capacity_mw', 'scale_factor'], inplace=True)

    #use the first year, residential as the grouping data (if they exist)
    agent_group = pd.merge(agent_attr, df[['county_id','pgid']].drop_duplicates(), how='inner', on='county_id')
    if "year" in agent_attr.columns:
        agent_group = agent_group.query("year == year.min()")
    if "sector_abbr" in agent_attr.columns:
        agent_group = agent_group.query("sector_abbr=='res'")
    
    agent_group_zero = pd.DataFrame()
    if exclude_zeros==True:
        print("Seperating agents with 0 installations into their own group")
        zero_counties = (df.groupby(['county_id','sector_abbr'], as_index=False)
                         .agg({'number_of_adopters':'sum'})
                         .query("sector_abbr=='res' & number_of_adopters==0")['county_id'])
        agent_group_zero = agent_group[agent_group.county_id.isin(zero_counties)].copy()
        agent_group_zero.insert(0,'group',0)

        agent_group = agent_group[~agent_group.county_id.isin(zero_counties)]
    
    ##GROUPING
    #group by **MAXIMUM MARKET SHARE**
    if grouping_method == "mms":
        print("Grouping by Maximum Market Share")

        #scale data around mean and to unit variance before clustering
        data = agent_group.loc[:, "max_market_share"].to_numpy(copy=True).reshape(-1,1)
        scaled = StandardScaler().fit(data)
        clusters = KMeans(n_clusters=nclusters, random_state=0).fit(scaled.transform(data))
        agent_group.insert(0, 'group', clusters.labels_+1)
        agent_group = agent_group.append(agent_group_zero)
    
    #group using **K MEANS** clustering
    elif grouping_method == "kmeans":
        print("Grouping using k means clustering")

        #scale data around mean and to unit variance before clustering
        data = agent_group.loc[:, kmeans_vars]
        scaled = StandardScaler().fit(data)
        clusters = KMeans(n_clusters=nclusters, random_state=0).fit(scaled.transform(data))
        agent_group.insert(0, 'group', clusters.labels_+1)
        agent_group = agent_group.append(agent_group_zero)

    #**MANUAL** grouping method
    elif grouping_method == "manual":
        print("Grouping by Phase of Adoption")
        agent_group = df.loc[(df["sector_abbr"] == 'res')].copy()
        agent_group.insert(0, 'group', np.nan)
        agent_group.sort_values(by=['year'], inplace=True)
        #print(agent_group.head())

        for agentid in agent_group.pgid.unique():
            phase = agent_group.loc[agent_group["pgid"]==agentid, 'number_of_adopters'].copy().apply(lambda x: 'O' if x == 0 else 'X')
            phase = "".join(phase) #make sure the character length matches the training group names
            agent_group.loc[agent_group["pgid"]==agentid, 'group'] = phase + "-" + \
                        agent_group.loc[agent_group["pgid"]==agentid, 'census_division']

            #distinguish between pre and post adoption counties
            if sum(agent_group.query("pgid==@agentid & year<2018")['number_of_adopters'])==0:
                agent_group.loc[agent_group["pgid"]==agentid, 'group'] = '-'*5
            
        #print(agent_group.head())
        if "year" in agent_attr.columns:
            agent_group = agent_group.loc[agent_group["year"] == agent_attr["year"].min()]
        else:
            agent_group.query("year==year.min()", inplace=True)

    #default groups are **STATES**
    else:
        print("Grouping by State")
        if exclude_zeros==True:
            agent_group = pd.concat([agent_group, agent_group_zero.drop(columns={"group"})])
        agent_group.insert(0, 'group', agent_group["state_abbr"])
    
    groups = pd.value_counts(agent_group.loc[:,"group"])
    #agent_group = agent_group.loc[:,["county_id","state_abbr","group"]]
    #agent_group = agent_group.loc[:,["pgid","group"]]
    
    
    if verbose==True:
        print("The smallest group has", groups.min(), "members")
        print(groups)
       
    agent_group = pd.merge(agent_group, df[['agent_id','pgid','year','sector_abbr','number_of_adopters','customers_in_bin_initial','pct_of_bldgs_developable']], on='pgid')
    agent_group = agent_group[['group','agent_id','pgid','state_abbr','county_id','year','sector_abbr','number_of_adopters','customers_in_bin_initial','pct_of_bldgs_developable']]

    return agent_group

####****---- END OF FUNCTION market_grouper ----****####
# ==================================================================================================================== #



#fit model to the distribution of counts within each group
def lasso_disagg(df_grouped, county_attr, a=1000, verbose=False):
    counts = df_grouped[['group','agent_id','state_abbr','county_id','year','sector_abbr','number_of_adopters']].copy()
    counts = counts.set_index('agent_id')
    counts['total'] = counts.groupby(['group','year','sector_abbr'], sort=False).number_of_adopters.transform('sum')
    counts['prop'] = np.clip(counts['number_of_adopters'] / np.maximum(counts['total'],0.001), 0, 1)
    counts = counts.sort_values(by="year")
    counts.reset_index(level=0, inplace=True)

    # merge in historical non-economic attributes
    data = counts.merge(county_attr, on=['state_abbr','county_id','year'])
    data = data.sort_values(by="year")
    data.drop(columns=['state_abbr','county_id'], inplace=True)

    county_val = pd.DataFrame(columns=['agent_id','group','sector_abbr','year','number_of_adopters','pred_prop'])
    coeff = []
    for group in list(set(data.group.unique()) - set([0,'-----'])):
        
        pred = data.query("group==@group")[['agent_id','fips_code','group','sector_abbr','year','number_of_adopters']]
        
        if pred.agent_id.nunique()==1:            
            print("\nOnly one agent in group", group)
            pred['pred_prop'] = 1
            county_val = county_val.append(pred)
            continue
        elif pred.fips_code.nunique()==1:
            print("\nOnly one county in group", group)
            pred['pred_prop'] = 1 / pred.agent_id.nunique()
            county_val = county_val.append(pred)
            continue
        else:
            Y_train = data.query("group==@group & year<year.max()")['prop']
            
            # Training data. Identifier variable should be excluded
            X_train = data.query("group==@group & year<year.max()").drop(
                            columns=['agent_id','fips_code','group','sector_abbr','year','prop','number_of_adopters','total'])
            
            pred.drop(columns='fips_code', inplace=True)
            X_pred = data.query("group==@group").drop(
                            columns=['agent_id','fips_code','group','sector_abbr','year','prop','number_of_adopters','total'])
        
        lasso = Lasso(alpha=a)
        lasso.fit(X_train, -np.log(1/(Y_train+1e-40) - 1)) #logit
        pred['pred_prop'] = 1/(1+np.exp(-lasso.predict(X_pred))) - 1e-40
        coeff.append(lasso.coef_)


        if verbose==True:
            print("\nFor group:", group)
            print("number of features used:", np.sum(lasso.coef_!=0))
            
            Y_test = data.query("group==@group & year==year.max()")['prop']
            X_test = data.query("group==@group & year==year.max()").drop(columns=['agent_id','group','sector_abbr','year','prop','number_of_adopters','total'])
            
            print("training score for alpha={0}: {1}".format(a, lasso.score(X_train,-np.log(1/(Y_train+1e-40) - 1))))
            print("test score for alpha={0}: {1}".format(a, lasso.score(X_test,-np.log(1/(Y_test+1e-40) - 1))))

        county_val = county_val.append(pred)
    
    # for the "0" group, the proportion for all agents should be 0
    pred = data.query("group in [0,'-----']")[['agent_id','group','sector_abbr','year','number_of_adopters']]
    pred['pred_prop'] = 0
    county_val = county_val.append(pred)
    county_val = county_val.merge(counts[['agent_id','county_id','state_abbr']].drop_duplicates(), on='agent_id')

    coeff = pd.DataFrame(data=coeff, columns=data.drop(columns=['agent_id','fips_code','group','sector_abbr','year','prop','number_of_adopters','total']).columns)

    return (county_val, coeff)

####****---- END OF FUNCTION lasso_disagg ----****####
# ==================================================================================================================== #
