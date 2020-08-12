"""
Distributed Generation Market Demand Model (dGen) - Final Release
National Renewable Energy Lab
"""

import time
import os
import pandas as pd
import psycopg2.extras as pgx
import numpy as np
import data_functions as datfunc
import utility_functions as utilfunc
import settings
import agent_mutation
import diffusion_functions_elec
import financial_functions
import input_data_functions as iFuncs
import calibration_functions as calib

#==============================================================================
# raise  numpy and pandas warnings as exceptions
#==============================================================================
pd.set_option('mode.chained_assignment', None)
#==============================================================================


def main(mode = None, resume_year = None, endyear = None, ReEDS_inputs = None):

    try:
        # =====================================================================
        # SET UP THE MODEL TO RUN
        # =====================================================================
        # initialize Model Settings object
        # (this controls settings that apply to all scenarios to be executed)
        model_settings = settings.init_model_settings()

        # make output directory
        os.makedirs(model_settings.out_dir)
        # create the logger
        logger = utilfunc.get_logger(os.path.join(model_settings.out_dir, 'dg_model.log'))

        # connect to Postgres and configure connection
        con, cur = utilfunc.make_con(model_settings.pg_conn_string, model_settings.role)
        engine = utilfunc.make_engine(model_settings.pg_engine_string)

        # register access to hstore in postgres
        pgx.register_hstore(con)  

        logger.info("Connected to Postgres with the following params:\n{}".format(model_settings.pg_params_log))
        owner = model_settings.role

        # =====================================================================
        # LOOP OVER SCENARIOS
        # =====================================================================
        # variables used to track outputs
        scenario_names = []
        dup_n = 1
        out_subfolders = {'wind': [], 'solar': []}
        for i, scenario_file in enumerate(model_settings.input_scenarios):
            logger.info('============================================')
            logger.info('============================================')
            logger.info("Running Scenario {i} of {n}".format(i=i + 1,n=len(model_settings.input_scenarios)))
            # initialize ScenarioSettings object
            # (this controls settings that apply only to this specific scenario)
            scenario_settings = settings.init_scenario_settings(scenario_file, model_settings, con, cur)
            scenario_settings.input_data_dir = model_settings.input_data_dir

            # summarize high level secenario settings
            datfunc.summarize_scenario(scenario_settings, model_settings)

            # create output folder for this scenario
            input_scenario = scenario_settings.input_scenario
            scen_name = scenario_settings.scen_name
            out_dir = model_settings.out_dir
            (out_scen_path, scenario_names, dup_n) = datfunc.create_scenario_results_folder(input_scenario, scen_name,
                                                             scenario_names, out_dir, dup_n)
                                                             
            # create folder for input data csvs for this scenario
            scenario_settings.dir_to_write_input_data = out_scen_path + '/input_data'
            scenario_settings.scen_output_dir = out_scen_path
            os.makedirs(scenario_settings.dir_to_write_input_data)
                                                             
            # get other datasets needed for the model run
            logger.info('Getting various scenario parameters')

            schema = scenario_settings.schema
            max_market_share = datfunc.get_max_market_share(con, schema)
            load_growth_scenario = scenario_settings.load_growth.lower()
            inflation_rate = datfunc.get_annual_inflation(con, scenario_settings.schema)
            
            if model_settings.realtime_calibration == False:
                bass_params = datfunc.get_bass_params(con, scenario_settings.schema)
            #import non-economic agent attributes - /input_data or postgres?
            ### acs5 = datfun.acs5()
            acs5 = pd.read_csv('../input_data/non_economic/acs5_processed.csv', index_col=0)
            refUSA = pd.read_csv('../input_data/non_economic/refUSA_processed.csv', index_col=0)

            # get settings whether to use pre-generated agent file ('User Defined'- provide pkl file name) or generate new agents
            agent_file_status = scenario_settings.agent_file_status

            #==========================================================================================================
            # CREATE AGENTS
            #==========================================================================================================
            logger.info("--------------Creating Agents---------------")
            
            if scenario_settings.techs in [['wind'], ['solar']]:

                # =========================================================
                # Initialize agents
                # =========================================================   
                           
                solar_agents = iFuncs.import_agent_file(scenario_settings, con, cur, engine, model_settings, agent_file_status, input_name='agent_file')   

                # Get set of columns that define agent's immutable attributes
                cols_base = list(solar_agents.df.columns)
                
                # Calculate Bass parameters based on historic adoption
                if model_settings.realtime_calibration == True:
                    calibration_time = time.time()
                    # group agents together into larger markets
                    agent_groups = calib.market_grouper(refUSA, solar_agents.df.reset_index(), "kmeans",\
                                           kmeans_vars=['OWNER_RENTER_STATUS','MARITAL_STATUS','LENGTH_OF_RESIDENCE','CHILDREN_IND','CHILDRENHHCOUNT', 'MAILABILITY_SCORE','WEALTH_FINDER_SCORE','FIND_DIV_1000','ESTMTD_HOME_VAL_DIV_1000','PPI_DIV_1000'], verbose=True)
                    # calibrate Bass parameters at the market level
                    bass_params = calib.calibrate_Bass(agent_groups)
                    bass_params = pd.merge(agent_groups[['group','pgid','sector_abbr']].drop_duplicates(), bass_params, how='left', on=['group','sector_abbr'])                        
                    
                    if model_settings.propensity_model == True:                 
                        agent_val, propensities = calib.lasso_disagg(agent_groups, acs5.drop(columns='NAME'), a=2000)
                        agent_val.group = agent_val.group.astype(np.int64)
                        propensities.to_csv(out_dir + '/propensities.csv', index=False)
                    
                    print("CALIBRATION TIME", time.time()-calibration_time)

            #==============================================================================
            # TECHNOLOGY DEPLOYMENT
            #==============================================================================

            if scenario_settings.techs == ['solar']:
                # get incentives and itc inputs
                state_incentives = datfunc.get_state_incentives(con)
                itc_options = datfunc.get_itc_incentives(con, scenario_settings.schema)
                nem_state_capacity_limits = datfunc.get_nem_state(con, scenario_settings.schema)
                nem_state_and_sector_attributes = datfunc.get_nem_state_by_sector(con, scenario_settings.schema)
                nem_utility_and_sector_attributes = datfunc.get_nem_utility_by_sector(con, scenario_settings.schema)
                nem_selected_scenario = datfunc.get_selected_scenario(con, scenario_settings.schema)
                rate_switch_table = agent_mutation.elec.get_rate_switch_table(con)

                #==========================================================================================================
                # INGEST SCENARIO ENVIRONMENTAL VARIABLES
                #==========================================================================================================
                deprec_sch = iFuncs.import_table( scenario_settings, con, engine, owner, input_name ='depreciation_schedules', csv_import_function=iFuncs.deprec_schedule)
                carbon_intensities = iFuncs.import_table( scenario_settings, con, engine,owner, input_name='carbon_intensities', csv_import_function=iFuncs.melt_year('grid_carbon_intensity_tco2_per_kwh'))
                wholesale_elec_prices = iFuncs.import_table( scenario_settings, con, engine, owner, input_name='wholesale_electricity_prices', csv_import_function=iFuncs.process_wholesale_elec_prices)
                pv_tech_traj = iFuncs.import_table( scenario_settings, con, engine, owner,input_name='pv_tech_performance', csv_import_function=iFuncs.stacked_sectors)
                elec_price_change_traj = iFuncs.import_table( scenario_settings, con, engine, owner,input_name='elec_prices', csv_import_function=iFuncs.process_elec_price_trajectories)
                load_growth = iFuncs.import_table( scenario_settings, con, engine, owner,input_name='load_growth', csv_import_function=iFuncs.stacked_sectors)
                pv_price_traj = iFuncs.import_table( scenario_settings, con, engine, owner,input_name='pv_prices', csv_import_function=iFuncs.stacked_sectors)
                batt_price_traj = iFuncs.import_table( scenario_settings, con, engine,owner, input_name='batt_prices', csv_import_function=iFuncs.stacked_sectors)
                pv_plus_batt_price_traj = iFuncs.import_table( scenario_settings, con, engine,owner, input_name='pv_plus_batt_prices', csv_import_function=iFuncs.stacked_sectors)
                financing_terms = iFuncs.import_table( scenario_settings, con, engine, owner,input_name='financing_terms', csv_import_function=iFuncs.stacked_sectors)
                batt_tech_traj = iFuncs.import_table( scenario_settings, con, engine, owner,input_name='batt_tech_performance', csv_import_function=iFuncs.stacked_sectors)
                value_of_resiliency = iFuncs.import_table( scenario_settings, con, engine,owner, input_name='value_of_resiliency', csv_import_function=None)

                #==========================================================================================================
                # Calculate Tariff Components from ReEDS data
                #==========================================================================================================
                for i, year in enumerate(scenario_settings.model_years):

                    logger.info('\tWorking on {}'.format(year))

                    # determine any non-base-year columns and drop them
                    cols = list(solar_agents.df.columns)
                    cols_to_drop = [x for x in cols if x not in cols_base]
                    solar_agents.df.drop(cols_to_drop, axis=1, inplace=True)

                    # copy the core agent object and set their year
                    solar_agents.df['year'] = year

                    # is it the first model year?
                    is_first_year = year == model_settings.start_year

                    # get and apply load growth
                    solar_agents.on_frame(agent_mutation.elec.apply_load_growth, (load_growth))

                    # Update net metering and incentive expiration
                    cf_during_peak_demand = pd.read_csv('cf_during_peak_demand.csv') # Apply NEM on generation basis, i.e. solar capacity factor during peak demand
                    peak_demand_mw = pd.read_csv('peak_demand_mw.csv')
                    if is_first_year:
                        last_year_installed_capacity = agent_mutation.elec.get_state_starting_capacities(con, schema)

                    state_capacity_by_year = agent_mutation.elec.calc_state_capacity_by_year(con, schema, load_growth, peak_demand_mw, is_first_year, year,solar_agents,last_year_installed_capacity)
                    
                    #Apply net metering parameters
                    net_metering_state_df, net_metering_utility_df = agent_mutation.elec.get_nem_settings(nem_state_capacity_limits, nem_state_and_sector_attributes, nem_utility_and_sector_attributes, nem_selected_scenario, year, state_capacity_by_year, cf_during_peak_demand)
                    solar_agents.on_frame(agent_mutation.elec.apply_export_tariff_params, [net_metering_state_df, net_metering_utility_df])

                    # Apply each agent's electricity price change and assumption about increases
                    solar_agents.on_frame(agent_mutation.elec.apply_elec_price_multiplier_and_escalator, [year, elec_price_change_traj])

                    # Apply technology performance
                    solar_agents.on_frame(agent_mutation.elec.apply_batt_tech_performance, (batt_tech_traj))
                    solar_agents.on_frame(agent_mutation.elec.apply_pv_tech_performance, pv_tech_traj)

                    # Apply technology prices
                    solar_agents.on_frame(agent_mutation.elec.apply_pv_prices, pv_price_traj)
                    solar_agents.on_frame(agent_mutation.elec.apply_batt_prices, [batt_price_traj, batt_tech_traj, year])
                    solar_agents.on_frame(agent_mutation.elec.apply_pv_plus_batt_prices, [pv_plus_batt_price_traj, batt_tech_traj, year])

                    # Apply value of resiliency
                    solar_agents.on_frame(agent_mutation.elec.apply_value_of_resiliency, value_of_resiliency)

                    # Apply depreciation schedule
                    solar_agents.on_frame(agent_mutation.elec.apply_depreciation_schedule, deprec_sch)

                    # Apply carbon intensities
                    solar_agents.on_frame(agent_mutation.elec.apply_carbon_intensities, carbon_intensities)

                    # Apply wholesale electricity prices
                    solar_agents.on_frame(agent_mutation.elec.apply_wholesale_elec_prices, wholesale_elec_prices)

                    # Apply host-owned financial parameters
                    solar_agents.on_frame(agent_mutation.elec.apply_financial_params, [financing_terms, itc_options, inflation_rate])

                    if 'ix' not in os.name: 
                        cores = None
                    else:
                        cores = model_settings.local_cores

                    # Apply state incentives
                    solar_agents.on_frame(agent_mutation.elec.apply_state_incentives, [state_incentives, year, model_settings.start_year, state_capacity_by_year])
                    
                    # Calculate System Financial Performance
                    solar_agents.chunk_on_row(financial_functions.calc_system_size_and_performance, sectors=scenario_settings.sectors, cores=cores, rate_switch_table=rate_switch_table)

                    # Calculate the financial performance of the S+S systems
                    #solar_agents.on_frame(financial_functions.calc_financial_performance)

                    # Calculate Maximum Market Share
                    solar_agents.on_frame(financial_functions.calc_max_market_share, max_market_share)

                    # determine "developable" population
                    solar_agents.on_frame(agent_mutation.elec.calculate_developable_customers_and_load)

                    # Apply market_last_year
                    if is_first_year == True:
                        state_starting_capacities_df = agent_mutation.elec.get_state_starting_capacities(con, schema)
                        solar_agents.on_frame(agent_mutation.elec.estimate_initial_market_shares, state_starting_capacities_df)
                        market_last_year_df = None
                    else:
                        solar_agents.on_frame(agent_mutation.elec.apply_market_last_year, market_last_year_df)

                    if model_settings.propensity_model == True:
                        ##??? WILL NEED SOMETHING BETTER HERE FOR PREDICTION YEARS ###
                        # get the closest year in agent_val
                        propensity_year = agent_val.year.unique()[np.abs(agent_val.year.unique() - year).argmin()]
                        
                        print("Using Propensity model with fits from year ", propensity_year)
                        # Calculate diffusion at the group level based on bass diffusion
                        #aggregate solar agents to group level
                        market_cols = ['sector_abbr', 'developable_agent_weight', 'max_market_share', 'market_share_last_year', 'market_value_last_year',
                                        'system_kw', 'system_capex_per_kw', 'adopters_cum_last_year', 'system_kw_cum_last_year', 
                                        'batt_kw', 'batt_kwh', 'batt_kw_cum_last_year', 'batt_kwh_cum_last_year', 'initial_batt_kw', 'initial_batt_kwh',
                                        'initial_pv_kw', 'initial_number_of_adopters', 'initial_market_share', 'initial_market_value']
                        dissag_cols = ['number_of_adopters','new_adopters', 'new_adopt_fraction','market_value','new_market_value',
                                       'diffusion_market_share', 'market_share', 'new_market_share', 'bass_market_share',
                                       'system_kw_cum','new_system_kw','batt_kw_cum', 'batt_kwh_cum', 'new_batt_kw', 'new_batt_kwh']
                        solar_groups = pd.merge(solar_agents.df[['pgid']+market_cols], agent_groups[['group','pgid','sector_abbr']].drop_duplicates(), on=["pgid","sector_abbr"])
                        solar_groups.developable_agent_weight  = solar_groups.developable_agent_weight.astype(np.float64)
                        solar_groups.system_capex_per_kw  = solar_groups.system_capex_per_kw.astype(np.float64)
                        
                        solar_groups = solar_groups.groupby(["group","sector_abbr"], as_index=False).sum()
                        ##??? USE group AS agent_id, TO KEEP COMPATIBILITY OF AGGREGATION FUNCTIONS
                        solar_groups['agent_id'] = solar_groups['group']
                        
                        solar_groups, _ = diffusion_functions_elec.calc_diffusion_solar(solar_groups, is_first_year,
                                                                                        bass_params.drop(columns="pgid").drop_duplicates(), year,
                                                                                        id_var="group", no_constraint=True)
                        #### save propensity variables to runs folder
                        
                        # Disaggregate to agent level based on noneconomic factors
                        solar_groups = pd.merge(solar_groups.drop(columns='agent_id'), agent_val.drop(columns="number_of_adopters").query("year==@propensity_year"), on=['group','sector_abbr'], how="left")
                        solar_groups.sort_values(by=["pgid","year"], inplace=True)                      
                        
                        #group_diffusion['system_kw_cum'] = group_diffusion['system_kw_cum'] * group_diffusion['pred_prop']
                        for val in dissag_cols:
                            solar_groups[val] = solar_groups[val] * solar_groups['pred_prop']
                            # Make sure this year is >= last year
                            # if is_first_year == False:
                                #market_last_year_df[val]
                            #solar_groups[val] = solar_groups.groupby('pgid')[val].cummax()
                        
                        solar_groups.drop(columns=['index','group','pred_prop'], inplace=True)
                        solar_agents.df = pd.merge(solar_agents.df, solar_groups[['pgid','sector_abbr',*(set(solar_groups.columns) - set(solar_agents.df.columns))]], on=['pgid','sector_abbr'])
                        
                        market_last_year_df = solar_agents.df.copy()[['agent_id',
                                                                      'market_share', 'max_market_share', 'number_of_adopters','initial_batt_kw','initial_batt_kwh',
                                                                      'market_value', 'initial_number_of_adopters', 'initial_pv_kw', 'initial_market_share', 'initial_market_value',
                                                                      'system_kw_cum', 'new_system_kw', 'batt_kw_cum', 'batt_kwh_cum']]

                        market_last_year_df.rename(columns={'market_share':'market_share_last_year', 
                                                            'max_market_share':'max_market_share_last_year',
                                                            'number_of_adopters':'adopters_cum_last_year',
                                                            'market_value': 'market_value_last_year',
                                                            'system_kw_cum':'system_kw_cum_last_year',
                                                            'batt_kw_cum':'batt_kw_cum_last_year',
                                                            'batt_kwh_cum':'batt_kwh_cum_last_year'}, inplace=True)
                                               
                    else:
                        # Calculate diffusion based on economics and bass diffusion
                        if model_settings.realtime_calibration == True:
                            solar_agents.df, market_last_year_df = diffusion_functions_elec.calc_diffusion_solar(solar_agents.df, is_first_year, bass_params, year, id_var="pgid")
                        else:
                            solar_agents.df, market_last_year_df = diffusion_functions_elec.calc_diffusion_solar(solar_agents.df, is_first_year, bass_params, year)

                    # Estimate total generation
                    solar_agents.on_frame(agent_mutation.elec.estimate_total_generation)

                    # Aggregate results
                    scenario_settings.output_batt_dispatch_profiles = True

                    last_year_installed_capacity = solar_agents.df[['state_abbr','system_kw_cum','batt_kw_cum','batt_kwh_cum','year']].copy()
                    last_year_installed_capacity = last_year_installed_capacity.loc[last_year_installed_capacity['year'] == year]
                    last_year_installed_capacity = last_year_installed_capacity.groupby('state_abbr')[['system_kw_cum','batt_kw_cum','batt_kwh_cum']].sum().reset_index()

                    #==========================================================================================================
                    # WRITE AGENT DF AS PICKLES FOR POST-PROCESSING
                    #==========================================================================================================
                    write_annual_agents = True
                    drop_fields = ['index', 'reeds_reg', 'customers_in_bin_initial', 'load_kwh_per_customer_in_bin_initial',
                                   'load_kwh_in_bin_initial', 'sector', 'roof_adjustment', 'load_kwh_in_bin', 'naep',
                                   'first_year_elec_bill_savings_frac', 'metric', 'developable_load_kwh_in_bin', 'initial_number_of_adopters', 'initial_pv_kw', 
                                   'initial_market_share', 'initial_market_value', 'market_value_last_year', 'teq_yr1', 'mms_fix_zeros', 'ratio', 
                                   'teq2', 'f', 'new_adopt_fraction', 'bass_market_share', 'diffusion_market_share', 'new_market_value', 'market_value', 'total_gen_twh',
                                   'consumption_hourly', 'solar_cf_profile', 'tariff_dict', 'deprec_sch', 'batt_dispatch_profile',
                                   'cash_flow', 'cbi', 'ibi', 'pbi', 'cash_incentives', 'state_incentives', 'export_tariff_results']
                    drop_fields = [x for x in drop_fields if x in solar_agents.df.columns]
                    df_write = solar_agents.df.drop(drop_fields, axis=1)

                    if write_annual_agents==True:
                        df_write.to_pickle(out_scen_path + '/agent_df_{}.pkl'.format(year))

                    # Write Outputs to the database
                    if i == 0:
                        write_mode = 'replace'
                    else:
                        write_mode = 'append'
                    iFuncs.df_to_psql(df_write, engine, schema, owner,'agent_outputs', if_exists=write_mode, append_transformations=True)

                    del df_write

            elif scenario_settings.techs == ['wind']:
                logger.error('Wind not yet supported')
                break
            
            #==============================================================================
            #    Outputs & Visualization
            #==============================================================================
            logger.info("---------Saving Model Results---------")
            out_subfolders = datfunc.create_tech_subfolders(out_scen_path, scenario_settings.techs, out_subfolders)

            #####################################################################
            # drop the new scenario_settings.schema
            engine.dispose()
            con.close()
            datfunc.drop_output_schema(model_settings.pg_conn_string, scenario_settings.schema, model_settings.delete_output_schema)
            #####################################################################
            
            logger.info("-------------Model Run Complete-------------")
            time_to_complete = time.time() - model_settings.model_init
            logger.info('Completed in: {} seconds'.format(round(time_to_complete, 1)))

    except Exception as e:
        # close the connection (need to do this before dropping schema or query will hang)
        if 'engine' in locals():
            engine.dispose()
        if 'con' in locals():
            con.close()
        if 'logger' in locals():
            logger.error(e.__str__(), exc_info = True)
        if 'scenario_settings' in locals() and scenario_settings.schema is not None:
            # drop the output schema
            datfunc.drop_output_schema(model_settings.pg_conn_string, scenario_settings.schema, model_settings.delete_output_schema)
        if 'logger' not in locals():
            raise

    finally:
        if 'con' in locals():
            con.close()
        if 'scenario_settings' in locals() and scenario_settings.schema is not None:
            # drop the output schema
            datfunc.drop_output_schema(model_settings.pg_conn_string, scenario_settings.schema, model_settings.delete_output_schema)
        if 'logger' in locals():
            utilfunc.shutdown_log(logger)
            utilfunc.code_profiler(model_settings.out_dir)

if __name__ == '__main__':
    main()
