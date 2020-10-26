"""
Name: calibration_functions
Purpose: Determine market groups and calibrate diffusion parameters

    (1) Group agents into markets
    (2) Calibrate Bass parameters at market level
    (3) Estimate adoption propensity for each agent across groups

"""

import config
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso


def tune_maximum_market_share(df):
    """
    Fit maximum market share (MMS) and payback period (PP) to MMS = exp(alpha * PP)

    Parameters
    ----------
    df : pandas dataframe
        Main df

    Returns
    -------
    Pandas dataframe with revised maximum_market_share column
    """

    MMS = np.maximum(df['adopters_cum_last_year'] / df['developable_agent_weight'], df['max_market_share'])
    alpha, _ = curve_fit(lambda x, a: np.exp(-a * x), df.loc[MMS > 0].payback_period, MMS[MMS > 0], bounds=(0,np.inf))

    df.max_market_share.where(MMS==0, np.exp(-alpha * df['payback_period']), inplace=True)

    return df

####****---- END OF FUNCTION tune_maximum_market_share ----****####
# ==================================================================================================================== #



def differential_Bass(x, p, q, mms):
    """
    Simple, differential form of a Bass curve to regress onto

    Parameters
    ----------
    x : pandas Series of floats
        vector of market share at time t (Cumulative installs * Max market size)
    p : float
        Bass parameter p
    q : float
        Bass parameter q
    mms : numpy array
        vector of maximum market share values to constrain fit

    Returns
    -------
    Pandas series of flaots: vector of market share at time t+1
    """

    return np.clip(-q*x*x + (1+q-p)*x + p, 0, mms)

####****---- END OF FUNCTION differential_Bass ----****####
# ==================================================================================================================== #



def calibrate_Bass(df_grouped):
    """
    Calibrate Bass parameters using the differential form of the Bass model

    Parameters
    ----------
    df_grouped : pandas dataframe
        matches group (along with historic market data) to agent_id's

    Returns
    -------
    pandas dataframe of p & q Bass parameters for each group and sector
    """

    df_grouped['max_market_size'] = df_grouped['max_market_share'] * df_grouped['developable_agent_weight']
    market = df_grouped.groupby(['group','year','sector_abbr'], as_index=False).agg({'developable_agent_weight':'sum', 'number_of_adopters':'sum','max_market_size':'sum','pct_of_bldgs_developable':'mean'})
    market['f'] = np.clip(market['number_of_adopters'] / market['developable_agent_weight'], 0, 1)
    market['mms'] = np.clip(market['max_market_size'] / market['developable_agent_weight'], 0, 1)

    fitted_Bass = []
    for group in market.group.unique():
        data = market.query("group==@group").sort_values(by="year")
        for sector in data.sector_abbr.unique():
            xdata = data.query("sector_abbr==@sector & year<year.max()")['f']
            ydata = data.query("sector_abbr==@sector & year>year.min()")['f']
            mms = data.query("sector_abbr==@sector & year>year.min()")['mms'].to_numpy()

            [p,q], _ = curve_fit(lambda x, p, q: differential_Bass(x, p, q, mms=mms), xdata, ydata, bounds=([1e-4, 0.2], [2e-3, 0.4]))

            fitted_Bass.append([group, sector, p, q])
    
    fitted_Bass = pd.DataFrame(data=fitted_Bass, columns=['group','sector_abbr','bass_param_p','bass_param_q'])
    
    fitted_Bass['teq_yr1'] = 1
    fitted_Bass['tech'] = 'solar'
    
    return fitted_Bass

####****---- END OF FUNCTION calibrate_Bass ----****####
# ==================================================================================================================== #



# format data for use in market_grouper
def assemble_market_data(df):
    """
    Format solar_agents.df for market_grouper
    (ie merge historic market data scaled to agent level)

    Parameters
    ----------
    df : pandas dataframe
        Main df

    Returns
    -------
    pandas dataframe with additional historic columns
    'system_kw_cum', 'number_of_adopters' for each historic year
    """

    # calculate scale factors by county
    df = df.astype({'developable_roof_sqft': 'float64', 'pct_of_bldgs_developable': 'float64'})
    df['county_developable_roof_sqft'] = df.groupby(['state_abbr', 'county_id', 'sector_abbr']).developable_roof_sqft.transform('sum')
    df['agent_count'] = df.groupby(['state_abbr', 'county_id', 'sector_abbr']).agent_id.transform('count')
    # calculate scale factor - weight that is given to each agent based on proportion of county total
    # where county cumulative capacity is 0, proportion evenly to all agents
    df['scale_factor'] =  np.where(df['county_developable_roof_sqft'] == 0, 1.0/df['agent_count'], df['developable_roof_sqft'] / df['county_developable_roof_sqft'])
 
    ## Bring in market data for calibration    
    historical_county_capacity_df = pd.read_csv(config.INSTALLED_CAPACITY_BY_COUNTY)[['state_abbr','county_id','sector_abbr','year','imputed_cum_size_kW']]
    historical_county_capacity_df = (historical_county_capacity_df
                                    .groupby(['state_abbr','county_id','sector_abbr','year'])
                                    .sum().rename(columns={'imputed_cum_size_kW':'observed_capacity_kw'})
                                    .reset_index())#.query("year in [2014,2016,2018]"))
    historical_county_capacity_df.sector_abbr.replace('commercial', 'com', inplace=True)

    historical_county_capacity_df = historical_county_capacity_df.query("sector_abbr in @df.sector_abbr.unique() & year in [2014,2016,2018]")

    # join historical data to agent df
    market_data = pd.merge(df, historical_county_capacity_df, how='left', on=['state_abbr', 'county_id', 'sector_abbr'])
    
    # use scale factor to constrain agent capacity values to historical values
    market_data['system_kw_cum'] = market_data['scale_factor'] * market_data['observed_capacity_kw']
    
    # recalculate number of adopters using anecdotal values
    market_data['number_of_adopters'] = np.where(market_data['sector_abbr'] == 'res', market_data['system_kw_cum']/5.0, market_data['system_kw_cum']/100.0)
    
    market_data.drop(columns=['agent_count', 'county_developable_roof_sqft', 'observed_capacity_kw', 'scale_factor'], inplace=True)
    
    return market_data

####****---- END OF FUNCTION assemble_market_data ----****####
# ==================================================================================================================== #



def market_grouper(county_attr, df, grouping_method, kmeans_vars=[], exclude_zeros=True, verbose=False):
    """
    Group agents into markets using a grouping_method and additional attributes

    Parameters
    ----------
    county_attr : pandas dataframe
        (non-economic) county level data containing additional attributes
    df : pandas dataframe
        Main df with market_data merged in
    grouping_method : string
        One of "kmeans","manual","state"
    kmeans_vars : list
        column names in df or county_attr on which to cluster when using
        grouping_method "kmeans"
    exclude_zeros : boolean
        should agents with 0 adoption be in their own group?
    verbose : boolean
        enable verbose printing

    Returns
    -------
    pandas dataframe of agent_id's assigned to groups
    """

    # check to make sure vars exist in dataframes
    df_vars = set(kmeans_vars) - set(county_attr.columns)
    if len(df_vars - set(df.columns)) != 0:
        print('Dropping the following variables which were not found in agent attributes or solar_agents.df')
        print('\t', df_vars - set(df.columns))
        kmeans_vars = list(set(kmeans_vars) - (df_vars - set(df.columns)))

    agent_group = pd.merge(county_attr, df[['county_id','agent_id', *df_vars]].drop_duplicates(), how='inner', on='county_id')
    
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
    nclusters = np.clip(agent_group.county_id.nunique()//2, 2, 20)
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
        print("grouping using kmeans clustering")

        #scale data around mean and to unit variance before clustering
        data = agent_group.loc[:, kmeans_vars]
        scaled = StandardScaler().fit(data)
        X = scaled.transform(data)
        
        silhouette_best = -1
        for n_clusters in range(2, agent_group.county_id.nunique()//2):
        
            # Initialize the clusterer with n_clusters value and a random seed
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            
            cluster_labels = clusterer.fit_predict(X)
        
            # calculate the silhouette score to evaluate this number of clusters
            silhouette_val = silhouette_score(X, cluster_labels)
            if silhouette_val > silhouette_best:
                silhouette_best = silhouette_val
                n_clusters_best = n_clusters
                labels_best = cluster_labels
            
        print("using", n_clusters_best, "clusters")
        agent_group.insert(0, 'group', labels_best+1)
        agent_group = agent_group.append(agent_group_zero)
        
    #**MANUAL** grouping method
    elif grouping_method == "manual":
        print("Grouping by Phase of Adoption")
        agent_group = df.loc[(df["sector_abbr"] == 'res')].copy()
        agent_group.insert(0, 'group', np.nan)
        agent_group.sort_values(by=['year'], inplace=True)
        #print(agent_group.head())

        for agentid in agent_group.agent_id.unique():
            phase = agent_group.loc[agent_group['agent_id']==agentid, 'number_of_adopters'].copy().apply(lambda x: 'O' if x == 0 else 'X')
            phase = "".join(phase) #make sure the character length matches the training group names
            agent_group.loc[agent_group['agent_id']==agentid, 'group'] = phase + "-" + \
                        agent_group.loc[agent_group['agent_id']==agentid, 'census_division']

            #distinguish between pre and post adoption counties
            if sum(agent_group.query("agent_id==@agentid & year<2018")['number_of_adopters'])==0:
                agent_group.loc[agent_group['agent_id']==agentid, 'group'] = '-'*5
            
        #print(agent_group.head())
        if "year" in county_attr.columns:
            agent_group = agent_group.loc[agent_group["year"] == county_attr["year"].min()]
        else:
            agent_group.query("year==year.min()", inplace=True)

    #default groups are **STATES**
    else:
        print("Grouping by State")
        if exclude_zeros==True:
            agent_group = pd.concat([agent_group, agent_group_zero.drop(columns='group')])
        agent_group.insert(0, 'group', agent_group.state_abbr.factorize()[0] + 1)
    
    groups = pd.value_counts(agent_group.loc[:,"group"])

    
    if verbose==True:
        print("The smallest group has", groups.min(), "members")
        print(groups)
    
    agent_group = pd.merge(agent_group, df, on='agent_id', suffixes=(None, '_redundant'))
    agent_group = agent_group[['group','agent_id','developable_agent_weight','max_market_share']]

    return agent_group

####****---- END OF FUNCTION market_grouper ----****####
# ==================================================================================================================== #



#fit model to the distribution of counts within each group
def lasso_disagg(df_grouped, county_attr, a=2000, verbose=False):
    """
    fit a model to the distribution of counts among agents in each group.
    Use the lasso method to select regression variables

    Parameters
    ----------
    df_grouped : pandas dataframe
        matches group (along with historic market data) to agent_id's
    county_attr : pandas dataframe
        (non-economic) county level data containing additional attributes
    a : int
        alpha parameter for sklearn.linear_model.Lasso
    verbose : boolean
        enable verbose printing

    Returns
    -------
    pandas dataframe of agent_id with groups and yearly predicted proportions of adoption
    pandas dataframe of the regression coefficients for the variables in county_attr
    agent_val, propensities
    """

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
            county_val = county_val.append(pred.drop(columns='fips_code'))
            continue
        elif pred.fips_code.nunique()==1:
            print("\nOnly one county in group", group)
            pred['pred_prop'] = 1 / pred.agent_id.nunique()
            county_val = county_val.append(pred.drop(columns='fips_code'))
            continue

        Y_train = data.query("group==@group & year<year.max()")['prop']
            
        # Training data. Identifier variable should be excluded
        X_train = data.query("group==@group & year<year.max()").drop(
                            columns=['agent_id','fips_code','group','sector_abbr','year','prop','number_of_adopters','total'])
            
        pred.drop(columns='fips_code', inplace=True)
        X_pred = data.query("group==@group").drop(
                            columns=['agent_id','fips_code','group','sector_abbr','year','prop','number_of_adopters','total'])
        
        lasso = Lasso().fit(X_train, Y_train)
        pred['pred_prop'] = np.clip(lasso.predict(X_pred), 0, 1)
        pred['pred_prop'] = pred.groupby(['year','sector_abbr']).pred_prop.transform(lambda x: x / x.sum())

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
    county_val = county_val.astype({'group':'int64'})

    coeff = pd.DataFrame(data=coeff, columns=data.drop(columns=['agent_id','fips_code','group','sector_abbr','year','prop','number_of_adopters','total']).columns)

    return (county_val, coeff)

####****---- END OF FUNCTION lasso_disagg ----****####
# ==================================================================================================================== #
