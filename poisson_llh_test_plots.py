#!/usr/bin/env python


from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals
from builtins import open
from future import standard_library
standard_library.install_aliases()
import numpy as np 
from utils.plotting.standard_modules import *
from matplotlib.backends.backend_pdf import PdfPages

if __name__=='__main__':

	import argparse
	import pickle
	import scipy.stats as scp

	parser = argparse.ArgumentParser('plot TS and fitted value distributions for all modified poissonian llhs')

	parser.add_argument('-is','--input-stem',help='input pickle file stem name',default = 'pseudo_exp_llh')
	parser.add_argument('-l','--list',help='list of llh functions you want to plot',nargs='+')
	parser.add_argument('-o','--output',help='output pdf file',default='test.pdf')

	args= parser.parse_args()

	outpdf = PdfPages(args.output)

	coverage_fig = Figure(figsize=(10,10))

	sample_chi2_distrib = np.random.chisquare(size=100,df=1)
	binning = np.linspace(0,20,31)


	for llh_name in args.list:
		print(('plotting ',llh_name))

		assert llh_name in ['modchi2','SAY','dima','glu2','barlow'],'ERROR: Available likelihood functions are: glu2 modchi2 dima SAY barlow'

		pckl_name = args.input_stem+'_'+llh_name+'.pckl'
		assert os.path.isfile(pckl_name),'ERROR: file %s not found'%(pckl_name)

		#
		# Load the pickle file containing information about pseudo-experiments
		#
		indata = pickle.load(open(pckl_name))

		container_TS_truth_high = []
		container_TS_truth_low = []
		container_TS_lowstat = []
		container_TS_highstat = []
		bias = []

		val_truth = 20.
		container_val_lowstat = []
		container_val_highstat = []

		for pseudo_exp in indata:

			val_low = pseudo_exp['lowstats_opt']['x']
			val_high =pseudo_exp['highstats_opt']['x']
			TS_low =  -pseudo_exp['lowstats_opt']['fun']
			TS_high = -pseudo_exp['highstats_opt']['fun']
			truth_low = pseudo_exp['lowstat_llh']
			truth_high = pseudo_exp['truth_llh']


			if (np.isfinite(val_low)) and (np.isfinite(val_high)) and (np.isfinite(TS_low)) and (np.isfinite(TS_high)) and (np.isfinite(truth_low)) and (np.isfinite(truth_high)):


				if len(val_low.shape)>=1:
					val_low = val_low[0]
				if len(val_high.shape)>=1:
					val_high = val_high[0]

				container_val_lowstat.append(float(val_low))
				container_val_highstat.append(float(val_high))

				container_TS_lowstat.append(2*np.abs(TS_low-truth_low))
				container_TS_highstat.append(2*np.abs(TS_high-truth_high))
				container_TS_truth_high.append(truth_high)
				container_TS_truth_low.append(truth_low)

				bias.append( 2*((TS_low-truth_low)-(TS_high-truth_high)) )
			else:
				continue


		

		fig = Figure(nx=2,ny=3,figsize=(20,30))
		fig.get_ax(x=0,y=0).set_title(llh_name)
		fig.get_ax(x=0,y=0).hist(container_TS_highstat,bins=binning,histtype='step',linewidth=2.,color='r',label='TS distribution')
		fig.get_ax(x=0,y=0).hist(sample_chi2_distrib,bins=binning,histtype='step',linewidth=2.,color='k',label=r'$\chi^{2}_{dof=1}$')
		fig.get_ax(x=0,y=0).set_xlabel(r'$-2(LLH_{opt}-LLH_{truth})$ (High statistics case)')
		fig.get_ax(x=0,y=0).legend()


		fig.get_ax(x=0,y=1).set_title(llh_name)
		fig.get_ax(x=0,y=1).hist(container_TS_lowstat,bins=binning,histtype='step',linewidth=2.,color='b',label='TS distribution')
		fig.get_ax(x=0,y=1).hist(sample_chi2_distrib,bins=binning,histtype='step',linewidth=2.,color='k',label=r'$\chi^{2}_{dof=1}$')
		fig.get_ax(x=0,y=1).set_xlabel(r'$-2(LLH_{opt}-LLH_{truth})$ (Low statistics case)')
		fig.get_ax(x=0,y=1).legend()

		fig.get_ax(x=1,y=0).set_title(llh_name)
		fig.get_ax(x=1,y=0).hist(container_val_highstat,bins=20,histtype='step',linewidth=2.,color='r')
		fig.get_ax(x=1,y=0).axvline(x=20,linewidth=2,color='k',ls='--',label=r'Truth ($\mu = 20$')
		fig.get_ax(x=1,y=0).set_xlabel('value (High statistics case)')
		fig.get_ax(x=1,y=0).legend()


		fig.get_ax(x=1,y=1).set_title(llh_name)
		fig.get_ax(x=1,y=1).hist(container_val_lowstat,bins=20,histtype='step',linewidth=2.,color='b')
		fig.get_ax(x=1,y=1).axvline(x=20,linewidth=2,color='k',ls='--',label=r'Truth ($\mu = 20$')
		fig.get_ax(x=1,y=1).set_xlabel('Value (Low statistics case)')
		fig.get_ax(x=1,y=1).legend()



		fig.get_ax(x=0,y=2).set_title(llh_name)
		fig.get_ax(x=0,y=2).hist(bias,bins=20)
		fig.get_ax(x=0,y=2).set_xlabel('Bias')
		outpdf.savefig(fig.fig)

		#
		# Coverage test
		#
		coverage_y = []
		coverage_x = np.linspace(0.0,1.0,101)

		for percent_coverage in coverage_x:
			chi2_TS_value = scp.chi2.ppf(percent_coverage,df=1)
			actual_coverage = sum(np.array(container_TS_lowstat)<=chi2_TS_value)/float(len(container_TS_lowstat))
			coverage_y.append(actual_coverage)



		coverage_fig.get_ax().plot(coverage_x,coverage_y,label=llh_name)



	coverage_fig.get_ax().set_xlabel('Expected Wilks coverage')
	coverage_fig.get_ax().set_ylabel('Actual Coverage (low statistics')
	coverage_fig.get_ax().legend()
	outpdf.savefig(coverage_fig.fig)


	outpdf.close()
