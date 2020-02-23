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

if __name__=='__main__':

	import argparse

	parser = argparse.ArgumentParser('Real-life test case of likelihood application to PISA')

	parser.add_argument('-p','--pipelines',help='list of PISA pipelines',nargs='+',required = True)
	parser.add_argument('-o','--output',help='stem of the output filename',default='poisson_reallife_test')

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
	# Side Step: Plot pseudo data distributions of:
	#			- weights in each bins in the total pipeline
	#           - weights in each bins in muons summed and neutrino-summed pipelines
	# 			- weights in each bins in individual templates (ie divide neutrinos per channel)
	data_fluctuation_checks = {}
	for mode in ['none','gauss','poisson','gauss+poisson']:

		data_fluctuation_checks[mode] = []
		for n in range(500):

			new_pseudo_data_set = simulation_summed.fluctuate(method=mode,random_state=2)
			data_fluctuation_checks[mode].append(new_pseudo_data_set)

	# Plot the side step





	#
	# Side Step: Generate 500 pseudo-data sets and look at their bin-by-bin event counts
	#


	#
	# Step 3: Format output of pseudo-data and simulation and feed it to the likelihood. Extract output
	#


	#
	# Step 4: For a single pseudo-data set, generate 500 fluctuations of both template distribution
	#