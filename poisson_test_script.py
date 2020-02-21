#!/usr/bin/env python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
from builtins import zip
from builtins import open
from builtins import int
from builtins import range
from future import standard_library
standard_library.install_aliases()
import llh_defs.poisson as poisson
import numpy as np
import scipy.special

import collections
import llh_defs.multinomial as multinomial


import scipy.optimize as scp
import pickle
import sys



####################################################################################
def fct_to_minimize(mu,sigma,counts_data=None,n_data=None,signal_fraction=None,weight_dict=None,stats_factor=1,llh_obj=None):
	'''
	Function that minimizes the parameter mu
	Used in generating the Monte Carlo
	'''
	if weight_dict is None:
		#np.random.seed(1234)
		# Create a MC set given the stats factor desire

		weight_dict = generate_MC(n=n_data,signal_fraction=signal_fraction,mu=mu,sigma=sigma,stats_factor=stats_factor,binning=binning)


	llh = llh_obj['fct'](data = counts_data,
						 dataset_weights=weight_dict,
						 **llh_obj['kwargs'])


	return -llh
###################################################################################

def generate_MC(n,signal_fraction,mu,sigma,stats_factor,binning):

	assert isinstance(n,int),'ERROR: n should be an int'
	assert isinstance(signal_fraction,float),'ERROR: signal fraction should be a float'
	assert isinstance(mu,float) or isinstance(mu,np.ndarray),'ERROR: mu should be a float or numpy array'
	assert isinstance(sigma,float) or isinstance(sigma,np.ndarray),'ERROR: sigma should be a float'


	N = int(n*stats_factor)

	nsig = int(N*signal_fraction)
	nbkg = N-nsig


	MC_weights = np.ones(N)*1./stats_factor

	signal = np.random.normal(loc=mu,scale=sigma,size=nsig)
	background = np.random.uniform(high=0.,low=40.,size=nbkg)

	MC = np.concatenate([signal,background])
	#
	# update the weight dict
	#
	MC_weight_tracker =np.digitize(MC,bins=binning)
	bin_number = np.arange(binning.shape[0]-1)

	weight_dict = {'allMC':[]}
	for i in bin_number:
		w_for_that_bin = MC_weights[MC_weight_tracker==(i+1)]
		weight_dict['allMC'].append(w_for_that_bin)


	return weight_dict

##################################################################################

if __name__=='__main__':

	import argparse

	parser = argparse.ArgumentParser('Toy Monte Carlo to test various likelihoods')

	parser.add_argument('-nd','--ndata',help='total number of data points',type=int,default=100)
	parser.add_argument('-sf','--signal-fraction',help='fraction of the data in the signal dataset',type=float,default=1.)
	parser.add_argument('-s','--stats-factor',help='Defines how much MC weights to produce w.r.t data',type=float,default=1.)
	parser.add_argument('-nt','--ntrials',help='number of pseudo experiments in the dias study',type=int,default=100)
	parser.add_argument('--make-llh-scan',help='if chosen, will run the likelihood scan for all llh',action='store_true')
	parser.add_argument('-o','--output',help='output stem files with plots',default = 'ToyMC_LLh')

	parser.add_argument('--interactive',help='use interactive plots',action='store_true')


	args = parser.parse_args()

	import matplotlib as mpl 
	if not args.interactive:
		mpl.use('agg')


	import pylab
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_pdf import PdfPages
	output_pdf = PdfPages(args.output+'.pdf')

	#
	# Parameters of the data
	#
	n_data = args.ndata
	signal_fraction = args.signal_fraction


	nbackground_low = 0.
	nbackground_high = 40.
	mu=20.0
	sigma=3.1
	nbins = 21


	#
	# Statistical factor for the MC
	#
	stats_factor = args.stats_factor
	binning = np.linspace(0,nbackground_high,nbins)
	X = binning[:-1]+0.5*(binning[1:]-binning[:-1])

	#
	# Minimization options
	#
	Ntrials = args.ntrials



	#=============================================================
	#
	# Generate the Data sample
	#
	nsig = int(n_data*signal_fraction)
	nbkg = n_data-nsig

	signal = np.random.normal(loc=mu,scale=sigma,size=nsig)
	background = np.random.uniform(high=nbackground_high ,low=nbackground_low ,size=nbkg)
	total_data = np.concatenate([signal,background])
	counts_data,bin_edges = np.histogram(total_data,bins=binning)
	Yerr_data = np.sqrt(counts_data)



	#==============================================================
	#
	# Plot the data
	#
	fig,ax = plt.subplots(figsize=(7,7))
	ax.errorbar(X,counts_data,yerr=Yerr_data,fmt='o',color='k')
	ax.set_xlabel('Some variable')
	ax.set_ylabel('Some counts')
	if args.interactive:
		plt.show()
	output_pdf.savefig(fig)



	#===============================================================
	#
	# Generate MC sample
	#
	print((type(total_data)))
	weight_dict_lowstats = generate_MC(n=n_data,signal_fraction=signal_fraction,mu=mu,sigma=sigma,stats_factor=stats_factor,binning=binning)

	# bin weights into histogram
	counts_mc_lowstats = np.zeros(counts_data.shape[0],dtype=float)
	errors_mc_lowstats = np.zeros(counts_data.shape[0],dtype=float)
	for s in list(weight_dict_lowstats.keys()):
		counts_mc_lowstats+= np.array([x.sum() for x in weight_dict_lowstats[s]])
		errors_mc_lowstats+= np.array([(x**2.0).sum() for x in weight_dict_lowstats[s]])


	#===============================================================
	#
	# Plot Data + MC 
	#
	fig,ax = plt.subplots(figsize=(7,7))
	ax.errorbar(X,counts_data,yerr=Yerr_data,fmt='o',color='k')
	ax.errorbar(X,counts_mc_lowstats,yerr=np.sqrt(counts_mc_lowstats),fmt='--',color='g')
	ax.set_xlabel('Some variable')
	ax.set_ylabel('Some counts')
	ax.set_title('Data vs. MC, same stats')
	if args.interactive:
		plt.show()
	output_pdf.savefig(fig)


	#===============================================================
	#
	# Define the likelihood functions
	#

	llh_sets = collections.OrderedDict()

	llh_sets['dima'] = {'fct': poisson.chirkin_llh,
						'plotting':{'color' : 'g',
									'linestyle':'-',
									'marker': '',
									'label' : 'Dima"s llh'},
						 'kwargs': {}}

	
	llh_sets['SAY'] =  {'fct': poisson.asy_llh,
						'plotting':{'color' : 'm',
									'linestyle':'-',
									'marker':'.',
									'label' : 'SAY llh'},
						'kwargs':{'use_original_code':True}}

	
	llh_sets['barlow'] =  {'fct': poisson.barlow_beeston_llh,
						'plotting':{'color' : 'b',
									'linestyle':'',
									'marker':'s',
									'label' : 'Barlow-Beeston llh'},
						'kwargs':{}}

	
	#
	# Compute Thorsten's Generalized Likelihood (which works on individual event basis)
	# First generalization: Poisson-gamma mixture
	#
	# In this generalization, individual event weights are assumed to follow some PDF
	# that is approximated by a gamma function. Those gamma fcts (one per event)are therefore convoluted
	# with each other, and marginalized over all possible expected weight value. 
	'''
	llh_sets['glu1'] =  {'fct': poisson.generic_pdf,
						 'plotting':{'color' : 'b',
									 'linestyle':'--',
									 'marker': '',
									 'label' : 'llh generalization 1 (zero bin method 1)'},
						 'kwargs':{'type':'gen1',
						 		   'empty_bin_strategy': 1}}
	'''
	#
	# Second generalization:
	#
	# In this generalization, the PDF for the weight is approximated by all the weights in a dataset.
	# Then the expectation value of this dataset's PDF is convoluted with all datasets to obtain the
	# likelihood of the data
	#
	
	llh_sets['glu2'] =  {'fct': poisson.generic_pdf,
						 'plotting':{'color' : 'r',
									 'linestyle':'--',
									 'marker': '',
									 'label' : 'llh generalization 2 (zero bin method 1)'},
						  'kwargs':{'type':'gen2',
						            'empty_bin_strategy':1,
						            'mean_adjustment':True}
						            }

	
	#
	# Third generalization:
	#
	# In this generalization we go deep down the rabbit hole, Inception style.
	# The PDF of each weight is approximated by a gamma function, and the expectation value of 
	# that function is approximated itself by a gamma distribution.
	#llh_sets['glu3'] =  {'fct': poisson.generic_pdf,
	#					 'plotting':{'color' : 'm',
	#								 'linestyle':'--',
	#								 'marker': '',
	#								 'label' : 'llh generalization 3 (zero bin method 1)'},
	#					 'kwargs':{'type':'gen3'}}



	if args.make_llh_scan:
		#================================================================
		#
		# Perform Likelihood scans for all types of llh
		#
		LLH_results= {}
		for name in list(llh_sets.keys()):
			LLH_results[name] = []


		tested_mus = []
		tested_sigmas= []

		for tested_mu in np.linspace(0.,nbackground_high,122):
			
			tested_mus.append(tested_mu)
			tested_sigmas.append(sigma)
			
			#
			# Recompute the truth MC
			#
			new_weight_dict = generate_MC(n=n_data,signal_fraction=signal_fraction,mu=tested_mu,sigma=sigma,stats_factor=stats_factor,binning=binning)
			
			for llhtype,llh_obj in list(llh_sets.items()):

				llhval	= llh_obj['fct'](data =np.array(counts_data),dataset_weights=new_weight_dict,**llh_obj['kwargs'])

				# change the sign of the modified chi2 back to a positive quantity
				if llhtype=='modchi2':
					llhval = -llhval


				LLH_results[llhtype].append(-llhval)
							   
			del new_weight_dict

		#===============================================================
		#
		# Plot Likelihood scans
		#
		fig2,ax2 = plt.subplots(figsize=(9,9))

		for llh_name in list(llh_sets.keys()):

			llhvals = LLH_results[llh_name]

			ax2.plot(tested_mus,llhvals,linewidth=2.0,**llh_sets[llh_name]['plotting'])

		ax2.set_xlabel('peak center')
		ax2.set_ylabel(r'-LLH / $\chi^{2}$')
		ax2.set_title('Likelihood scan over mu')
		ax2.legend()
		ax2.set_ylim([0,100])
		ax2.set_xlim([15,25])
		if args.interactive:
			plt.show()

		output_pdf.savefig(fig2)

		print('Saved figures on pdf: ', args.output)
		output_pdf.close()
		sys.exit('Bye!')


	###################################################################################################
	####################################################################################################


	#
	# Produce a Fixed Truth MC sample 
	#===========================================================================
	# Create a MC set given the stats factor desired

	weight_dict_t        = generate_MC(n=n_data,signal_fraction=signal_fraction,mu=mu,sigma=sigma,stats_factor=1000,binning=binning)

	weight_dict_lowstats = generate_MC(n=n_data,signal_fraction=signal_fraction,mu=mu,sigma=sigma,stats_factor=stats_factor,binning=binning)

	#================================================================
	#
	#
	# Perform Toy experiments
	#
	# - Set the MC to a particular stats level
	# - Run Ntrials pseudo-experiments
	# - Minimize negative llh
	# - Compute TS = -2*(llh_minimized-llh_truth)
	#
	#
	# Note on minimizer: Nelder-Mead is super slow. Powell is 1/2 the time.
	# L-BFGS-B is the fastest

	pseudo_experiments = {}
	import time
	import cProfile, pstats, io


	# To compare the likelihood results with each other, we want the
	# generated MC to be the same for a given trial, for each evaluation.
	# 
	# therefore, for each likelihood we use to perform the minimization,
	# we re-initiate the seed to the same value for a given trial

	seed_list = np.arange(Ntrials)

	for llh_name,llh_obj in list(llh_sets.items()):
		print(('minimizing: ',llh_name))
		t0 = time.time()

		pseudo_experiments = []
		pr = cProfile.Profile()
		pr.enable() # Start profiling time usage

		for n,seed in zip(list(range(Ntrials)),seed_list):
			np.random.seed(seed)



			
			experiment_result = {}

			nsignal = int(n_data*signal_fraction)
			nbackground = n_data-nsignal

			signal = np.random.normal(loc=mu,scale=sigma,size=nsignal)
			background = np.random.uniform(high=nbackground_high,low=nbackground_low,size=nbackground)
			total_data = np.concatenate([signal,background])
			counts_data,_ = np.histogram(total_data,bins=binning)


			# Compute the truth llh value of this pseudo experiment
			# truth - if the truth comes from infinite stats MC
			experiment_result['truth_llh'] = llh_obj['fct'](data=counts_data,dataset_weights=weight_dict_t,**llh_obj['kwargs'])

			# truth if the truth comes from low stats MC
			experiment_result['lowstat_llh'] = llh_obj['fct'](data=counts_data,dataset_weights=weight_dict_lowstats,**llh_obj['kwargs'])


			# minimized llh (high stats) 
			#print '\t high statistics case...'
			Return_values = scp.minimize(fct_to_minimize,x0=mu,args=(sigma,counts_data,n_data,signal_fraction,None,1000.,llh_obj),
										 method='L-BFGS-B',
										 jac = False,
										 options={'maxiter':2000,
												  #'maxfun':2000,
												  #'approx_grad':True,
												  'ftol':0.01},
										 bounds = [(0.,None)])


			experiment_result['highstats_opt'] = Return_values


			# minimized llh (low stats)
			#print '\t low statistics case...'
			Return_values = scp.minimize(fct_to_minimize,x0=mu,args=(sigma,counts_data,n_data,signal_fraction,None,stats_factor,llh_obj),
										 method='L-BFGS-B',
										 jac = False,
										 options={'maxiter':2000,
												  #'maxfun':2000,
												  #'approx_grad':True,
												  'ftol':0.01},
										 bounds = [(0.,None)])
										 

			experiment_result['lowstats_opt'] = Return_values	
			pseudo_experiments.append(experiment_result)
			#print pseudo_experiments
		
		t1 = time.time()
		pr.disable()
		s = io.StringIO()
		sortby = 'tottime'
		ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
		ps.print_stats()
		print((s.getvalue()))
		print(("Time for ",Ntrials," minimizations: ",t1-t0," s"))
		print("Saving to file...")
		pickle.dump(pseudo_experiments,open(args.output+'_pseudo_exp_llh_%s.pckl'%llh_name,'wb'))

		print("Saved.")
