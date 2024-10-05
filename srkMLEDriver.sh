#!/bin/bash
#
# Driver for the Mironova et al. cell attached patch clamp recordings.
# Licence. 
# This script is part of research codes developed in the Welsh Laboratory, University of Western Ontario, Canada.
# The author of this research code is: Dr. Sanjay R Kharche. email: skharche@uwo.ca .
# Users may use these codes provided that the original authors are acknowledged in the form of citations &
# the authors are informed of the work that uses the codes.
# Forward all communications to the Welsh Lab. email: galina.mironova@uwo.ca
# We do not guarantee that the code is free from errors. Users use the code at their own risk. We do not guarantee external dependencies will work either.
# The code is provided 'as is' without any guarantees.
#
rm *.abf
rm *myEstimates*.txt
#
# Choice of drug.
icaldrug="Nif"
# Control or Pressure cases.
cases="Control Pressure"
for cas in  ${cases}
do
# Recordings from the Mironova et al. data.
	for i in 1 2
	do
		cp DATA/Cell$i/${cas}Nifcell$i.abf input.abf && python3 ./srkMLE.py ${icaldrug} ${cas} $i && mv myEstimates.txt myEstimates${cas}cell$i.txt
	done
done
#

