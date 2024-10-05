#!/bin/bash
#
#
mle:
	./srkMLEDriver.sh

single:
	./srkMLEDriverLocalSingleRecord.sh

mc:
	python3 srkICaLMarkovChain.py
clp:
	python3 srkpClamp2ascii.py

# this will convert an abf file to plain text. there are 3 columns: time, voltage, current.
abf:
	python3 srkPyABF.py

clean:
	rm -rf someOutput.dat sample.txt solution.dat myEstimates.txt input.abf sim* *.log jobfile NC_probs.dat dataHistogram.dat finalEstimates.txt AaggMatrix.dat
	
veryclean:
	rm -rf pics*
	make clean
	rm *.abf myEstimates* NmyEstimates* *myHistog* *times*.dat psd*.dat stateshistogram*.dat acFiltered*.dat *.png
	rm -rf *.png icalData2023Sim
