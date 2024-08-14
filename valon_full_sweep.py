#!/usr/bin/env python

"""
valon_full_sweep.py - Valon sweep for a given frequency range.
Define range and step in freq variable.
Command: python valon_full_sweep output_file_name
Data output in ascii.
"""

import numpy as np
import n9020b
import time
import sys
import v5019


valon_id = "0403"
v5019_id = "6001"
command =''
powerout = ''
sp = None

TERMINATOR = '\r'

WAIT_TIMEOUT = 5
MAX_ATTEMPTS = 5
MULT_FACTOR = 12.

fstart = int(input("Enter starting Frequency in MHz: "))
fstop = int(input("Enter stop Frequency in MHz: "))
step = int(input("Enter steps amount: "))
freq = np.arange(fstart, fstop, step) #Define sweep frequency range and step amount

# Use this variable in case of using a heterodyne mixer
Lo = 49000    #Default Gunn frequency in MHz
# Lo = float(input("Enter Lo frequency [MHz]"))


if __name__ == '__main__':
    
    # Set Valon
    valon = v5019.V5019()
    
    valon.open_serial_port(port_name="/dev/ttyS0")
    time.sleep(1)
    
    # Set initial frequency
    valon.set_frequency(freq[0])
    
    powerout = float(input("Enter v5019 output power [dBm]: "))
    valon.set_power(powerout)
    valon.wave_on()

    attenuator = int(input("Enter attenuator value [dB]: "))
    
    print("file ", sys.argv[1] + "\n")
    f = open(sys.argv[1], 'a')
    
    pout = float(powerout) - attenuator
    print("Power out Valon - Attenuator \n", pout)
    f.write("Data: fValon fMix dBm   Valon power %3.1f \n" % (pout)) 

	#Set Spectrum Analyzer
    mxa = n9020b.N9020B()
    fcentering = mxa.set_center_frequency(freq[0])
    freqs,pows = mxa.get_trace()
    idn = mxa.command('*IDN?')
    
    try:
        for cfreq in freq:
            cfreq2 = MULT_FACTOR * cfreq
            valon.set_frequency(cfreq2)
            # print("Current frequency ", cfreq, "MHz")
            
            fcentering = mxa.set_center_frequency(cfreq2)
            span = mxa.set_span('10000')
            bw = mxa.set_rbw('1')
            freqs,pows = mxa.get_trace()
            time.sleep(1)
            center_to_marker = mxa.seek_max_and_center(0,0)
            print("Center to marker \n")
            time.sleep(1)
            print("Coarse centering ", center_to_marker , mxa.freq_units)
            span = mxa.set_span('500')
            bw = mxa.set_rbw('0.05')
            time.sleep(1)
            
            mark1 = mxa.mark_to_max(1)
            mark1b = mxa.get_marker_pos()
            freq, dbm = mark1b
       	    
            print("qsyn freq - marker freq - power ", mark1b)
            line = "%s "*3 % (cfreq2, freq, dbm) + "\n"
            print("line ", line)
            f.write(line)
            time.sleep(2)
    finally:
	    f.close()

valon.set_power(-40)
valon.wave_off()
valon.save_state()
valon.close_serial_port()


