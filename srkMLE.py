#!/usr/bin/python3
'''
# Licence. October 2024.
# This script is part of ion channels research codes developed in the Welsh Laboratory, University of Western Ontario, Canada.
# The author of this research code is: Dr. Sanjay R Kharche. email: skharche@uwo.ca .
# Users may use these codes provided that the original authors are acknowledged in the form of citations &
# the authors are informed of the work that uses the codes.
# Forward all communications to the Welsh Lab. email: galina.mironova@uwo.ca
'''

import pyabf
import matplotlib.pyplot as plt
import numpy as np
import statistics

from scipy import signal

# Standard imports
import numpy as np
from numpy.linalg import eig
import matplotlib
import matplotlib.pyplot as plt
import sys
import math
import random
from random import seed
import os
import itertools
sys.setrecursionlimit(1000000) # the default depth is 1000, I made it 1M for now.
import scipy
from scipy.stats import moment # to do the central moments.
import scipy.signal # to do bessel/butterworth filters.
from scipy.signal import find_peaks

import statsmodels
import statsmodels.tsa.api as smt # alongwith the np, this is for autocovariance.
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller # stationarity test.
from numpy.fft import fft, ifft
from matplotlib import mlab
from scipy import optimize
from scipy import sparse
from scipy.sparse import linalg
import numpy as np
from numpy.linalg import norm
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import pybaselines
from pybaselines import Baseline, utils
import numpy as np
from scipy.signal import argrelextrema

debg=1

def icCountNumberOfChannels(myStates, colNN):
	colN = int(colNN)
	return int(np.max(myStates[:,colN]))


def icUnitaryChannelTimeConstants(markovChain, DELTAT):
	if(int(max(markovChain[:,0]))!=1):
		print('handling Kco_unitary and Koc_unitary for 1 channel only. This function will not return good values otherwise.')

	int_closed 		= 0
	int_open 		= 0
	int_T 			= len(markovChain[:,0])
	allopenT 		= []
	allclosedT 		= []

	if(int_T<1):
		print('not enough data to find tau_o_unitary and tau_c_unitary, exiting.')
		sys.exit()  # is this sys exit or return of some value?

	for i in range(0, int_T-1):
		if(markovChain[i,0]==0):
			int_closed 		= int_closed + 1
		if(markovChain[i,0]==1):
			int_open 		= int_open + 1
		if(markovChain[i,0]==0 and markovChain[i+1,0]==1 ):
			closeduration 	= int_closed * DELTAT
			int_closed 		= 0
			allclosedT.append(closeduration)
		if(markovChain[i,0]==1 and markovChain[i+1,0]==0 ):
			openduration 	= int_open * DELTAT
			int_open 		= 0
			allopenT.append(openduration)

	tau_o 			= np.mean(allopenT)
	tau_c 			= np.mean(allclosedT) 
	OTN = len(allopenT)
	CTN = len(allclosedT)

	allopenTNP 		= np.zeros([OTN, 1])
	allclosedTNP 	= np.zeros([CTN, 1])

	for i in range(0, OTN):
		allopenTNP[i,0] = allopenT[i]

	for i in range(0, CTN):
		allclosedTNP[i,0] = allclosedT[i]
 
	return tau_o, tau_c, allopenTNP, allclosedTNP

######################################################################################

def icAllChannelTimeConstants(markovChain, DELTAT):

	int_closed 		= 0
	int_open 		= 0
	int_T 			= len(markovChain[:,0])
	allopenT 		= []
	allclosedT 		= []

	if(int_T<1):
		print('not enough data to find tau_o_all and tau_c_all, exiting.')
		sys.exit() # is this sys exit or return of some value?

	for i in range(0, int_T-1):
		if(markovChain[i,0]==0):
			int_closed 		= int_closed + 1
		if(markovChain[i,0]>0):
			int_open 		= int_open + 1
		if(markovChain[i,0]==0 and markovChain[i+1,0]>0 ):
			closeduration 	= int_closed * DELTAT
			int_closed 		= 0
			allclosedT.append(closeduration)
		if(markovChain[i,0]>0 and markovChain[i+1,0]==0 ):
			openduration 	= int_open * DELTAT
			int_open 		= 0
			allopenT.append(openduration)

	tau_o 			= np.mean(allopenT)
	tau_c 			= np.mean(allclosedT) 
	OTN = len(allopenT)
	CTN = len(allclosedT)

	allopenTNP 		= np.zeros([OTN, 1])
	allclosedTNP 	= np.zeros([CTN, 1])

	for i in range(0, OTN):
		allopenTNP[i,0] = allopenT[i]

	for i in range(0, CTN):
		allclosedTNP[i,0] = allclosedT[i]
 
	return tau_o, tau_c, allopenTNP, allclosedTNP

#################################################################################
def coupledMatrices(zeta, rho, numChannels):
# define P(C) as an aggregate states matrix. This matrix does not do xi and eta as Rutford does, but takes zeta/rho given by Chung 1996b.
	ACp 							= np.zeros([numChannels+1, numChannels+1])
# derivatives.
	dACpdzeta 	= np.zeros([numChannels+1, numChannels+1])
	dACpdrho 	= np.zeros([numChannels+1, numChannels+1])
#	dACpdk 		= np.zeros([numChannels+1, numChannels+1]) # not needed. dAcpdk = Acp.
				
	for L in range(0, numChannels+1):
		ACp[L,0] 					= 0.5 # xi. This is 1-delta.
		ACp[L,numChannels] 			= 0.5 # delta. This is delta.

	ACp[0,0] 						= zeta # this does not have to be zeta/rho, can be another estimatable paramter.
	ACp[0,numChannels] 				= 1.0 - zeta
	ACp[numChannels,0] 				= 1.0 - rho
	ACp[numChannels, numChannels] 	= rho # P, coupled in aggregated state.

	dACpdzeta[0,0] 						=  1.0 # derivative of coupled aggregate matrix w.r.t. zeta.
	dACpdzeta[0,numChannels] 				= -1.0

	dACpdrho[numChannels,0] 				= -1.0
	dACpdrho[numChannels,numChannels] 	=  1.0 # derivative of coupled aggregate matrix w.r.t. rho.

	return ACp, dACpdzeta, dACpdrho

#################################################################################
#################################################################################
def independentMatrices2(zeta, rho, numChannels):
	AInd 		= np.zeros([numChannels+1, numChannels+1])
	dAInddrho 	= np.zeros([numChannels+1, numChannels+1])	
	dAInddzeta 	= np.zeros([numChannels+1, numChannels+1])	
	
	s = numChannels # for notation consistency with Biometrics 253*.pdf equation 6.
	for i in range(0, s+1): # rows.
		for j in range(0, s+1):
			min_k 	= max(i-j, 0)
			max_k 	= min(s-j, i)
#			print(i, j, min_k, max_k)
			aij 		= 0.0
			daijdrho 	= 0.0
			daijdzeta	= 0.0
			for k in range(min_k, max_k+1): # this is from Biometrics paper.
				alpha_sijk = float( math.comb(i, k)*math.comb(s-i, k+j-i) )
				asijkrho	 = math.pow( 1.0 - rho, k)	*math.pow(rho, i-k)
				asijkzeta	 = math.pow(1.0 - zeta, k+j-i)	*math.pow(zeta, s-j-k)
				
				aij = aij + alpha_sijk*asijkrho*asijkzeta
				
				dasijkdzeta = -(k+j-i)*math.pow(1.0 - zeta, k+j-i-1.0)*math.pow(zeta, s-j-k)+(s-j-k)*math.pow(1.0 - zeta, k+j-i)	*math.pow(zeta, s-j-k-1.0)
				daijdzeta = daijdzeta + alpha_sijk*asijkrho*dasijkdzeta
				
				dasijkdrho = -k*math.pow(1.0-rho, k-1.0)*math.pow(rho, i-k) + math.pow(1.0-rho, k)*(i-k)*math.pow(rho, i-k-1.0)
				daijdrho = daijdrho + alpha_sijk*asijkzeta*dasijkdrho

			AInd[i,j] 			= aij
			dAInddrho[i,j] 		= daijdrho
			dAInddzeta[i,j] 	= daijdzeta
	return AInd, dAInddzeta, dAInddrho

#################################################################################	
def transition_matrix(transitions):
    n = 1+ max(transitions) #number of states

    M = [[0]*n for _ in range(n)]

    for (i,j) in zip(transitions,transitions[1:]):
        M[i][j] += 1

    #now convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return M

#################################################################################
def srkHMM(myDataRaw, nSweeps, sweepNum):
	global DELTAT
	DELTAT 	= myDataRaw[2,0] - myDataRaw[1,0] # seconds.
	T0 		= 0.0 
	T1 		= 0.2 
	T2 		= 0.35
	T3 		= 5.10 
	T4 		= 5.3
	T5 		= 6.0 
	epsT 	= 0.01 
	
	intT0 		= int(T0 	/DELTAT)
	intT1 		= int(T1 	/DELTAT) 
	intT2 		= int(T2 	/DELTAT) 
	intT3 		= int(T3 	/DELTAT) 
	intT4 		= int(T4	/DELTAT) 
	intT5 		= int(T5	/DELTAT)
	intepsT 	= int(epsT 	/DELTAT) 
	quantal 	= 0.5 # god given number.

	intT 	= intT3 - intT2
	X 		= np.zeros([intT, 1])
	orig 	= np.zeros([intT, 1])
	for i in range(0, intT):
		X[i,0] 		= - myDataRaw[intT2+i,2]
		orig[i, 0] 	= X[i,0]


	preclamp = np.zeros([intT1, 1])
	for i in range(0, intT1):
		preclamp[i,0] = myDataRaw[i,2]

	from BaselineRemoval import BaselineRemoval
	baseObj=BaselineRemoval(X[:,0])
	Zhangfit_output=baseObj.ZhangFit()
		
	for i in range(0, intT):
		X[i,0] = Zhangfit_output[i]
	SR 		= 1.0/DELTAT # sampling rate (frequency). Hz.
	fco 	= 140.0 # cut off frequency given by Galina. Hz.
	a 		= 3.011*fco # see the matlab script in this directory.		
	N 		= int(np.ceil(0.398*SR/fco))
	
	L 		= int(2*N + 1)
	b 		= np.zeros([L, 1])
	for k in range(-N, N+1):
		b[k+N,0] = 3.011*(fco/SR)*np.exp( -np.pi*(a*k/SR)**2 );
	b[:,0] = b[:,0] / np.sum(b[:,0])

	if np.sum(b[:,0]) < 0.985 or np.sum(b[:,0]) > 1.0:
		print('the Gaussian window is off, exiting.')
		sys.exit()

	y_lfilter 	= scipy.ndimage.convolve(X[:,0] , b[:,0] , output=None, mode='reflect', cval=0.0, origin=0)
	peaks, _ 	= scipy.signal.find_peaks(-y_lfilter, distance=N, )
	
	modArr 		= scipy.stats.mode(y_lfilter[peaks], axis=0) # most often occuring.
	removeThis 	= modArr[0]
	removeThis 	= np.median(y_lfilter[peaks])
	for i in range(0, intT):
		y_lfilter[i] = y_lfilter[i] - 0.9*removeThis

	for i in range(0, intT):
		X[i,0] = y_lfilter[i]

	myStates		= (-1)*np.ones([intT, 1])	
	currentState 	= 0
	oldState 		= 0
	myFrac 			= 0.1
	for i in range(0, intT):
		for level in range(0, 11):
			if( (quantal*level - quantal*myFrac) < X[i,0] and (quantal*level + quantal*myFrac) >= X[i,0] ):
				myStates[i, 0] = level
				if level!=oldState:
					oldState = level
		if(myStates[i,0]<0):
			myStates[i,0] = oldState

	NC = icCountNumberOfChannels(myStates, 0)

	transitions = []
	for i in range(0, len(myStates[:,0])):
		transitions.append(int(myStates[i,0]))
	AaggD 		= transition_matrix(transitions) # the array transitions has to be 1D and int.
	AaggData 	= np.array(AaggD)

# number of transitions. Popen from page 176.
	nT 			= 0
	Popen 		= 0.0
	Topen 		= 0.0
	Tclosed 	= 1.0
	for i in range(1, intT):
		if myStates[i-1,0]!=myStates[i,0]:
			nT+=1
		if myStates[i,0]>0.0:
			Popen = Popen + myStates[i,0]
			Topen = Topen + 1
		if myStates[i,0]==0:
			Tclosed = Tclosed + 1
	if NC>0.0:
		Popen = Popen/(NC*intT)
	else:
		Popen = 0.0
	NPopen = Topen / (Topen + Tclosed)
			
# number of states, number of transitions, Popen, NPopen.
	print(NC, nT, Popen, NPopen)

	tau_o_unitary 	= -1.0
	tau_c_unitary 	= -1.0
	kco_unitary 	= -1.0
	koc_unitary 	= -1.0
	Pom_unitary 	= -1.0
	if NC==1:
		tau_o_unitary, tau_c_unitary, openTimes_unitary, closedTimes_unitary = icUnitaryChannelTimeConstants(myStates, DELTAT)
		if tau_o_unitary > 0.0 and tau_c_unitary > 0.0:
			kco_unitary = 1.0/tau_c_unitary
			koc_unitary = 1.0/tau_o_unitary
			Pom_unitary = 1.0 - koc_unitary / (kco_unitary + koc_unitary)
		opfile = f"opentimes_unitary.dat"
		with open(opfile,'a') as fo:
			np.savetxt(fo, openTimes_unitary.reshape(1, -1), fmt='%2.2f', delimiter="\n") # remove fmt.
		clfile = f"closedtimes_unitary.dat"
		with open(clfile,'a') as fc:
			np.savetxt(fc, closedTimes_unitary.reshape(1, -1), fmt='%2.2f', delimiter="\n") # remove fmt.

	tau_o_all 	= -1.0
	tau_c_all 	= -1.0
	kco_all 	= -1.0
	koc_all 	= -1.0
	Pom_all 	= -1.0

	tau_o_all, tau_c_all, openTimes_all, closedTimes_all = icAllChannelTimeConstants(myStates, DELTAT)
	if tau_o_all > 0.0 and tau_c_all > 0.0:
		kco_all = 1.0/tau_c_all
		koc_all = 1.0/tau_o_all
		Pom_all = 1.0 - koc_all / (kco_all + koc_all)
	opallfile = f"opentimes_all.dat"
	with open(opallfile,'a') as foall:
		np.savetxt(foall, openTimes_all.reshape(1, -1), fmt='%20.20f', delimiter="\n") # remove fmt.
	clallfile = f"closedtimes_all.dat"
	with open(clallfile,'a') as fcall:
		np.savetxt(fcall, closedTimes_all.reshape(1, -1), fmt='%20.20f', delimiter="\n") # remove fmt.


	maxCurrent 		= quantal * max(myStates[:,0])

###################### iterations. #########xxxxxxxxxxxxxx#############
# The estimation iterations. 
	zeta 	= 0.1
	rho 	= 0.1
	kappa 	= 0.01

	if NC==0:
		zeta 	= 1.0
		rho 	= 0.0
		kappa 	= 0.0
		return np.array([0.0, 0.0, 0.0, zeta, rho, kappa, 0, Popen, NPopen, nSweeps, -1.0, -1.0, -1.0,-1.0, -1.0, -1.0]) # iterations, NC, cost, zeta, rho, kappa, number of transitions, nSweeps.
	if NC==1:
		zeta 	= AaggData[0,0]
		rho 	= AaggData[1,1]
		kappa 	= 0.0
		return np.array([0.0, 1.0, 0.0, zeta, rho, kappa, nT, Popen, NPopen, nSweeps, tau_o_unitary, tau_c_unitary, Pom_unitary, tau_o_all, tau_c_all, Pom_all])
			
	for iterations in range(0, 5): # When it works, it takes less than 50 iterations.
		ACp,dACpdzeta, dACpdrho 	= coupledMatrices(zeta, rho, NC)
		AInd, dAInddzeta, dAInddrho = independentMatrices2(zeta, rho, NC) # derivative w.r.t. k is negative of the ind. matrix.
		Aagg 		= (1.0 - kappa)*  AInd 		+ kappa*ACp 			# the aggregate (L+1)x(L+1) elements.
		dAaggdzeta 	= (1.0 - kappa)*dAInddzeta 	+ kappa*dACpdzeta 
		dAaggdrho 	= (1.0 - kappa)*dAInddrho 	+ kappa*dACpdrho 	# the aggregate (L+1)x(L+1) elements.
		dAaggdk 	= 			-  AInd 			+              ACp 		# the aggregate (L+1)x(L+1) elements.		

		smallmu 	= 0.000001 # this is the gradient.
		costfunction 	= 0.0
		dFdzeta 	= 0.0
		dFdrho 		= 0.0
		dFdk 		= 0.0

		for i in range(0, NC+1):
			for j in range(0, NC+1):
				costfunction 	+= (Aagg[i,j] - AaggData[i,j])*(Aagg[i,j] - AaggData[i,j])
				dFdzeta 	+= (Aagg[i,j] - AaggData[i,j])*dAaggdzeta[i,j]
				dFdrho 		+= (Aagg[i,j] - AaggData[i,j])*dAaggdrho[i,j]
				dFdk 		+= (Aagg[i,j] - AaggData[i,j])*dAaggdk[i,j]

		if iterations>1:
			deltazeta 		= abs(zeta_revised - zeta)
			deltarho		= abs(rho_revised - rho)
			deltakappa 		= abs(kappa_revised - kappa)
		
# do the estimate of zeta, rho, k.
		zeta_revised 		= zeta 	- smallmu * dFdzeta
		rho_revised 		= rho 	- smallmu * dFdrho
		
# some simple hack. More clever use of kappa becoming negative may help.
		if kappa 	- smallmu * dFdk > 0.0:
			kappa_revised 	= kappa 	- smallmu * dFdk
		else:
			kappa_revised = 0.0

# error trapping.
# You have to allow for machine precision.
		goodBad = 1
		if zeta_revised<=0.0 or zeta_revised>1.0:
			print('revised zeta not a probablity. exiting.')
			print(zeta_revised)
			goodBad = 0
#			sys.exit()
			break
			
		if rho_revised<=0.0 or rho_revised>1.0:
			print('revised rho not a probablity. exiting.')
			print(rho_revised)
			goodBad = 0			
#			sys.exit()
			break
			
#		if kappa_revised<=0.0 or kappa_revised>1.0:
#			print('revised kappa not a probablity. exiting.')

		if kappa_revised <0.0:
			kappa_revised = 0.0
		
		if kappa_revised>1.0:
			print('revised kappa more than 1, not a probablity. exiting.')
			goodBad = 0			
#			sys.exit()
			break

		diffz = abs(zeta - zeta_revised)
		diffr = abs(rho - rho_revised)
		diffk = abs(kappa - kappa_revised)
	
	# put new into old.
		if zeta_revised>0.0 and zeta_revised<=1.0:
			zeta 	= zeta_revised
		if rho_revised>0.0 and rho_revised<=1.0:			
			rho 		= rho_revised
		if kappa_revised>0.0 and kappa_revised<=1.0:		
			kappa 	= kappa_revised
		
		if iterations==-1:
			print('iter, NC, costfunction, zeta, rho, kappa:')	
			print(iterations, NC, costfunction, zeta, rho, kappa)
#		myEstimates.append([iterations, NC, costfunction, zeta, rho, kappa])
# the write is an overhead at each iteration, but lets me see the plot at each iterate.
#		np.savetxt("myEstimates.txt", myEstimates)
		someEps = 0.000001
		if abs(dFdk)<someEps and abs(dFdzeta) < someEps and abs(dFdrho) < someEps:
			break
		if diffz < someEps and diffr < someEps and diffk < someEps:
			break

	return np.array([iterations, NC, zeta, rho, kappa, nT, Popen, NPopen, nSweeps, tau_o_unitary, tau_c_unitary, Pom_unitary, tau_o_all, tau_c_all, Pom_all])

# 
def main():
	if len(sys.argv)<4:
		print('syntax: prog name, ical or drug, control or pressure, cell number.')
		sys.exit()
	print(sys.argv[0], sys.argv[1], sys.argv[2], 'cell number: ',  sys.argv[3]) #
	abf 	= pyabf.ABF("input.abf")
	nSweeps = abf.sweepCount
	print(sys.argv[1], sys.argv[2], sys.argv[3], nSweeps)
#	sys.exit()

	cellList=['1', '2']

	for i in range(0, nSweeps):
		abf.setSweep(sweepNumber=i, channel=0)
		voltage 	= abf.sweepY
		abf.setSweep(sweepNumber=i, channel=1)
		current 	= abf.sweepY
		myData 		= np.array(np.transpose([abf.sweepX, voltage, current]))
		print("Sweep number %d" % i)
		myEst 		= srkHMM(myData, nSweeps, i)
		np.set_printoptions(precision=10) # remove.
		if myEst[0]>-1.0:
			with open('myEstimates.txt','a') as f:
				np.savetxt(f, myEst.reshape(1, -1), fmt='%20.20f', delimiter=" ") # make this %2.2f, or remove fmt.

if __name__ == "__main__":
	main()
