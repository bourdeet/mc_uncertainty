#!/usr/bin/env python

#
# Testing script of the mc_uncertainty likelihoods
# for "real-life" application on data and simulation
# from oscNext
#
# This script will produce pseudodata from the simulated 
# neutrinos and muon 
#

from pisa.core import DistributionMaker
from pisa.core.container import ContainerSet,Container
from pisa.core.map import Map,MapSet
from pisa.core.binning import OneDimBinning, MultiDimBinning
from utils.particle_tools import PARTICLE_PLOTTING_PROPERTIES
from collections import OrderedDict
import numpy as np

from pisa.utils.log import set_verbosity, Levels
from pisa.utils.profiler import profile

#from pisa.utils.stats import generalized_poisson_llh
#from llh_defs.poisson import normal_log_probability, fast_pgmix

if __name__=='__main__':

    import argparse

    parser = argparse.ArgumentParser('Real-life test case of likelihood application to PISA')

    parser.add_argument('-p','--pipelines',help='list of PISA pipelines',nargs='+',required = True)
    parser.add_argument('-o','--output',help='stem of the output filename',default='poisson_reallife_test.pdf')
    parser.add_argument('--debug',help='prints out pisa debug logs',action='store_true')

    args= parser.parse_args()

    if args.debug:
        set_verbosity(Levels.DEBUG)


    #
    # Step 1: Read and process the provided pipeline
    #
    print('\n*********************************\n')
    print('Loading input pipelines...')
    simulation_raw = DistributionMaker(args.pipelines)


    #
    # Step 2: Sum the pipelines up and generate pseudo data from it
    #         (returns a MapSet object)
    print('\n*********************************\n')
    print('Summing pipeline outputs and generate pseudo_data')
    simulation_summed = simulation_raw.get_outputs(return_sum=True) # MapSet object
    simulation_events = simulation_raw.get_outputs(return_sum=False, output_mode='events')
    simulation_mapsets= simulation_raw.get_outputs(return_sum=False)



    simulation_summed = simulation_summed['weights']
    observed_counts = simulation_summed.fluctuate(method='poisson',random_state=0)
    '''
    analysis_binning = simulation_summed.binning
    N_bins=analysis_binning.tot_num_bins


    #
    # Make a test of the Map-class likelihood implementation
    # 
    new_dict = OrderedDict()
    for S in simulation_mapsets:
        for k,v in S.items():
            if k not in new_dict.keys():
                new_dict[k] = [m for m in v.maps]
            else:
                new_dict[k]+= [m for m in v.maps]

    for k,v in new_dict.items():
        new_dict[k] = MapSet(v)
    simulation_mapsets = new_dict

    print('*******************************************\n')
    print('Getting llh form the map implementation...\n')
    llh_per_bin_map_version = observed_counts.generalized_poisson_llh(expected_values = simulation_mapsets, binned = True)
    print('\n*******************************************\n')

    #
    #  1. Find the number of MC events in each bin, for each dataset
    #     Also find the highest weight value of a dataset
    #
    n_mcevents_per_dataset = OrderedDict()
    max_of_weight_distrib_all_datasets  = OrderedDict()
    empty_bins = OrderedDict()
    for containerset in simulation_events:

        for container in containerset:

            nevents_sim = np.zeros(N_bins)
            empty_bins_set = np.zeros(N_bins,dtype=np.int64)
            max_of_weight_distrib = np.zeros(N_bins)
            
            for index in range(N_bins):
                index_mask = container.array_data['bin_indices'].get('host')==index
                current_weights = container.array_data['weights'].get('host')[index_mask]
                n_weights = current_weights.shape[0]

                # Number of MC events in each bin
                nevents_sim[index] = n_weights
                empty_bins_set[index] = 1 if n_weights<=0 else 0
                m = 1.0 if current_weights.shape[0]<=0 else max(current_weights)
                max_of_weight_distrib[index]=m 
            
            n_mcevents_per_dataset[container.name] = nevents_sim
            empty_bins[container.name] = empty_bins_set
            max_of_weight_distrib_all_datasets[container.name] = max(max_of_weight_distrib)


    #
    #  2. Check where there are bins where we need to provide a pseudo Mc event count
    #

    bins_we_need_to_fix = np.ones(N_bins,dtype=np.int64)
    for k,v in empty_bins.items():
        bins_we_need_to_fix*=v
    # This gives the bin indices where we need to set non-zero weights if the count is zero in a given dataset
    bin_indices_we_need = np.where(bins_we_need_to_fix==0)[0]

    #
    #  3. Compute the alphas and betas that will be fed into the likelihood
    #
    alpha_maps_dict = OrderedDict()
    beta_map_dict   = OrderedDict()
    mean_adjustments = OrderedDict()
    new_weight_sum = np.zeros(N_bins)
    for containerset in simulation_events:

        for container in containerset:

            #  3. calculate number of mc events / bin
            kmc_ie_nevents_sim = n_mcevents_per_dataset[container.name]
            mean_of_weight_distrib = np.zeros(N_bins)
            var_of_weight_distrib  = np.zeros(N_bins)




            for index in range(N_bins):
                index_mask = container.array_data['bin_indices'].get('host')==index
                current_weights = container.array_data['weights'].get('host')[index_mask]

                # If no weights and other datasets have some, include a pseudo weight
                if current_weights.shape[0]<=0:

                    if index in bin_indices_we_need:
                        current_weights = np.array([max_of_weight_distrib_all_datasets[container.name]])
                    else: 
                        print('WOOOO! Empty bin common to all sets: ',index)
                        current_weights = np.array([0.0])

                # New number
                n_weights = current_weights.shape[0]
                kmc_ie_nevents_sim[index] = n_weights
                new_weight_sum[index]+=sum(current_weights)

                # Mean of the current weight distribution
                mean_w = np.mean(current_weights)
                mean_of_weight_distrib[index] = mean_w

                # Maximum weight in each bin
                max_w = np.max(current_weights)
                max_of_weight_distrib[index] = max_w


                # variance of the current weight
                var_of_weight_distrib[index]=((current_weights-mean_w)**2).sum()/(float(n_weights))

            #  Calculate mean adjustment
            mean_number_of_mc_events = np.mean(kmc_ie_nevents_sim)
            mean_adjustment = -(1.0-mean_number_of_mc_events) + 1.e-3 if mean_number_of_mc_events<1.0 else 0.0
            mean_adjustments[container.name] = mean_adjustment
            #  Variance of the poisson-gamma distributed variable
            var_z=(var_of_weight_distrib+mean_of_weight_distrib**2)
            
            #  alphas and betas
            betas = mean_of_weight_distrib/var_z
            trad_alpha=(mean_of_weight_distrib**2)/var_z

            alphas = (kmc_ie_nevents_sim+mean_adjustment)*trad_alpha


            # Calculate alphas and betas
            container.add_binned_data(key='alphas', data=(analysis_binning, alphas), flat=True)
            container.add_binned_data(key='betas', data=(analysis_binning, betas), flat=True)

            # Put them into Map objects
            alpha_maps_dict[container.name] = container.get_map(key='alphas').hist.flatten()
            beta_map_dict[container.name]   = container.get_map(key='betas').hist.flatten()


    #
    # Compute the likelihood for all maps individually
    #
    tot_llh = 0.0
    llh_per_bin = []

    for bin_i in range(N_bins):

        A = observed_counts.hist.flatten()[bin_i].nominal_value
        if A >100:
            #weight_sum = simulation_summed.hist.flatten()[bin_i].nominal_value
            weight_sum = new_weight_sum[bin_i]
            if weight_sum<0:
                print('\nERROR: negative weight sum should not happen...')
                raise Exception


            logP = normal_log_probability(k=A,weight_sum=weight_sum)
            print('my llh, bin :',bin_i, weight_sum,A)
            llh_per_bin.append(logP)
            tot_llh+=logP
        

        else:
            alp = np.array([alpha_maps_dict[x][bin_i] for x in alpha_maps_dict.keys()])
            bet = np.array([beta_map_dict[x][bin_i] for x in beta_map_dict.keys()])
            new_llh=fast_pgmix(A, alp[np.isfinite(alp)], bet[np.isfinite(bet)])
            llh_per_bin.append(new_llh)
            tot_llh+=new_llh

    print('Home-made llh: ',tot_llh)




    #
    # Step 3: compute the likelihood on a poisson_fluctuated pseudo-data template
    #


    # observed counts are MapSet objects
    print('\n*********************************\n')
    print('\ntesting the likelihood on fluctuated data...')
    print('Feeding fluctuated data into the llh function...')
    output, thorstens_llh, mean_adj, data, new_weights= generalized_poisson_llh(actual_values=observed_counts,expected_values=simulation_events)
    print(output)
    for i in range(N_bins):
        print('\n Bin fucking %i'%i,' data count: ',observed_counts.hist.flatten()[i])
        
        print('bin ',i,': ',thorstens_llh[i],llh_per_bin[i],llh_per_bin_map_version[i], thorstens_llh[i]-llh_per_bin[i], thorstens_llh[i]-llh_per_bin_map_version[i])

    print('Difference with homemade: ',output-tot_llh)
    '''
    print('\n*********************************\n')

    print('\nRunning a single hypothesis fit \n')

    #
    # Step 4: Run a sample test fit from pisa
    #
    from pisa.analysis.analysis import Analysis
    from pisa.utils.fileio import from_file


    minimizer_settings = { "method": { "value": "l-bfgs-b",
                                      "desc": "The string to pass to scipy.optimize.minimize so it knows what to use"
                                      },
                          "options":{ "value": {"disp"   : 0,
                                                "ftol"   : 1.0e-6,
                                                "eps"    : 1.0e-6,
                                                "maxiter": 100
                                                },
                                      "desc": { "disp"   : "Set to True to print convergence messages",
                                      "ftol"   : "Precision goal for the value of f in the stopping criterion",
                                      "eps"    : "Step size used for numerical approximation of the jacobian.",
                                      "maxiter": "Maximum number of iteration"
                                                }
                                    }
                        }


    #
    # To make the fit faster, fix the useless Barr parameters
    #
    simulation_raw.params['barr_a_Pi'].is_fixed= True
    simulation_raw.params['barr_x_antiK'].is_fixed= True
    simulation_raw.params['barr_f_Pi'].is_fixed= True
    simulation_raw.params['barr_w_antiK'].is_fixed= True
    simulation_raw.params['delta_index'].is_fixed = True
    simulation_raw.params['pion_ratio'].is_fixed = True
    simulation_raw.params['barr_b_Pi'].is_fixed = True
    simulation_raw.params['barr_c_Pi'].is_fixed = True
    simulation_raw.params['barr_d_Pi'].is_fixed = True
    simulation_raw.params['barr_e_Pi'].is_fixed = True
    simulation_raw.params['barr_g_Pi'].is_fixed = True
    simulation_raw.params['barr_h_Pi'].is_fixed = True
    simulation_raw.params['barr_i_Pi'].is_fixed = True
    simulation_raw.params['barr_w_K'].is_fixed = True
    simulation_raw.params['barr_x_K'].is_fixed = True
    simulation_raw.params['barr_y_K'].is_fixed = True
    simulation_raw.params['barr_z_K'].is_fixed = True
    simulation_raw.params['barr_z_antiK'].is_fixed = True
    simulation_raw.params['barr_y_antiK'].is_fixed = True
    simulation_raw.params['Genie_Ma_QE'].is_fixed = True
    simulation_raw.params['Genie_Ma_RES'].is_fixed = True
    simulation_raw.params['dis_csms'].is_fixed = True
    simulation_raw.params['nu_nc_norm'].is_fixed = True
    simulation_raw.params['aeff_scale'].is_fixed = True
    simulation_raw.params['delta_gamma_mu'].is_fixed=True
    simulation_raw.params['weight_scale'].is_fixed=True

    ana = Analysis()

    result_gen_llh = ana.fit_hypo(MapSet(observed_counts),
                          simulation_raw,
                          metric='generalized_poisson_llh',
                          minimizer_settings = minimizer_settings,
                          hypo_param_selections = 'nh',
                          )

    # For comparison, perform the same fit with the other standard metrics
    result_mod_chi2 = ana.fit_hypo(MapSet(observed_counts),
                          hypo_maker=simulation_raw,
                          metric='mod_chi2',
                          minimizer_settings = minimizer_settings,
                          hypo_param_selections = 'nh',
                          )


    result_llh = ana.fit_hypo(MapSet(observed_counts),
                          hypo_maker=simulation_raw,
                          metric='llh',
                          minimizer_settings = minimizer_settings,
                          hypo_param_selections = 'nh',
                          )

    import pickle

    pickle.dump([result_gen_llh,result_mod_chi2,result_llh],open('testing_generalized_llh.pckl','wb'))