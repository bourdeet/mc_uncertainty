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
from utils.particle_tools import PARTICLE_PLOTTING_PROPERTIES


if __name__=='__main__':

    import argparse

    parser = argparse.ArgumentParser('Real-life test case of likelihood application to PISA')

    parser.add_argument('-p','--pipelines',help='list of PISA pipelines',nargs='+',required = True)
    parser.add_argument('-o','--output',help='stem of the output filename',default='poisson_reallife_test.pdf')

    args= parser.parse_args()


    #
    # Step 1: Read and process the provided pipeline
    #
    print('Loading input pipelines...')
    simulation_raw = DistributionMaker(args.pipelines)


    #
    # Step 2: Sum the pipelines up and generate pseudo data from it
    #         (returns a MapSet object)
    print('Summing pipeline outputs and generate pseudo_data')
    simulation_summed = simulation_raw.get_outputs(return_sum=True)




    print('Side step: checkout the data fluctuation methods...')
    #
    # Side Step: Generate 500 pseudo-data sets and look at their bin-by-bin event counts
    #
    data_fluctuation_checks = []
    for n in range(500):

        new_pseudo_data_set = simulation_summed[0].fluctuate(method='poisson')
        data_fluctuation_checks.append(new_pseudo_data_set)

    # Plot the side step
    from utils.plotting.analysis_plots import *
    '''
    F = AnalysisFigure(output_pdf=args.output)
    F.plot_maps(maps=data_fluctuation_checks)


    #
    # Side Step: Plot pseudo data distributions of:
    #           - weights in each bins in the total pipeline
    #           - weights in each bins in muons summed and neutrino-summed pipelines

    # To merge array data from different containers of a single pipeline, we need to create
    # a new container object
    new_neutrino_container = Container(name='neutrinos', code=None, data_specs='events')
    merged_array_data = collections.OrderedDict()
    for c in simulation_raw.pipelines[0].stages[-1].data:

        # Retrieve the array data from each container
        for k,v in c.array_data.items():
            if k not in merged_array_data.keys():
                merged_array_data[k] = v.get()
            else:
                merged_array_data[k] = np.concatenate([merged_array_data[k],v])

    # Fill out the new merged container:
    for k,v in merged_array_data.items():
        new_neutrino_container.add_array_data(k,v)

    # Muons are contained in a single container, so we jsut fetch it
    muon_container = simulation_raw.pipelines[1].stages[-1].data.containers[0]

    F.plot_container(container=new_neutrino_container,
                     what='weights',
                     histogram_kw=None,
                     plotting_kw={'color':PARTICLE_PLOTTING_PROPERTIES['nue']['color_1d'],
                                  'drawstyle':'steps-mid',
                                  'linewidth':1.5},
                    Figure_kw={'title':'Total pipeline'})

    F.plot_container(container=muon_container,
                     what='weights',
                     histogram_kw=None,
                     plotting_kw={'color':PARTICLE_PLOTTING_PROPERTIES['muon']['color_1d'],
                                  'drawstyle':'steps-mid',
                                   'linewidth':1.5},
                     Figure_kw={'title':'muon pipeline'},)


    #
    # Plot individual containers of neutrino flavours
    #
    for c in simulation_raw.pipelines[0].stages[-1].data:
        F.plot_container(container=c,
                 what='weights',
                 histogram_kw=None,
                 plotting_kw={'color':PARTICLE_PLOTTING_PROPERTIES[c.name]['color_1d'],
                              'drawstyle':'steps-mid',
                               'linewidth':1.5},
                 Figure_kw={'title':'{} class weight distributions'.format(c.name)},)



    F.close_pdf()
    '''


    #
    # Step 3: Format a single container / channel set of weights into torsten's formatting
    #
    from pisa.utils.stats import generalized_poisson_llh

    # observed counts are MapSet objects
    print(type(simulation_summed))
    observed_counts = simulation_summed.fluctuate(method='poisson',random_state=0)
    

    output = generalized_poisson_llh(actual_values=new_pseudo_data_set,expected_values=simulation_raw)
    print(output)

    #flattened_expected = .hist.flatten(order='C')


