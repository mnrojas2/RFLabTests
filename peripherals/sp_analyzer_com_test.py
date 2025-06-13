#!/usr/bin/env python

"""
spectrum_analyzer_test.py - Code to test the communication with the N9020b SA
"""

import n9020b
import time


#Spectrum Analyzer Parameters
# Use this variable in case of using a heterodyne mixer
Lo = 49000 # Default Gunn frequency in MHz
#Lo = float(input("Enter Lo frequency [MHz]"))

TERMINATOR = '\r'

WAIT_TIMEOUT = 5
MAX_ATTEMPTS = 5
MULT_FACTOR = 12  # 1 when checking base band, 12 when using AMCi
REF_LEVEL = 10    # MXA reference level 


if __name__ == '__main__':
    mxa = n9020b.N9020B()
    fcentering = mxa.set_center_frequency(150000)
    input()
    freqs,pows = mxa.get_trace()
    input()
    idn = mxa.command('*IDN?')
    input()
    
    f = open('notas/sa_outputs.txt', 'a')
    freq = 12500 * MULT_FACTOR
    
    try:
        print("Current frequency ", freq, "MHz")
        fcentering = mxa.set_center_frequency(freq)
        span = mxa.set_span('14000')
        bw = mxa.set_rbw('3')
        
        freqs, pows = mxa.get_trace()
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
        
        cfreq = freq
        print("qsyn freq - marker freq - power ", mark1b)
        line = f"{cfreq}, {freq}, {dbm}\n"
        print("line ", line)
        f.write(line)
        time.sleep(2)
        
    finally:
	    f.close()