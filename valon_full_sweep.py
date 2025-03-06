#!/usr/bin/env python

"""
valon_full_sweep.py - Valon sweep for a given frequency range.
Define range and step in freq variable.
Command: python valon_full_sweep output_file_name
Data output in ascii.
"""

import os
import argparse
import numpy as np
import n9020b
import time
import sys
import v5019


# Initialize parser
parser = argparse.ArgumentParser(description='Reads data from txt files and plots ADC histogram and shows average and std.')
parser.add_argument('output_file', type=str, metavar='file', help='Name of the output file to save data.')
parser.add_argument('-fstart', '--freq_start', type=int, default=10400, help='Start frequency Valon (Mhz).')
parser.add_argument('-fstop', '--freq_stop', type=int, default=13600, help='Stop frequency Valon (Mhz).')
parser.add_argument('-stp', '--freq_step', type=int, default=100, help='Steps difference (Mhz).')
parser.add_argument('-pwr', '--power', type=float, default=6.0, help='Set Valon output power (dBm).')
parser.add_argument('-att', '--attenuation', type=float, default=10.0, help='Set Valon-coupled attenuator value (dBm).')


# Main
args = parser.parse_args()

valon_id = "0403"
v5019_id = "6001"
command =''
powerout = ''
sp = None

TERMINATOR = '\r'

WAIT_TIMEOUT = 5
MAX_ATTEMPTS = 5
MULT_FACTOR = 12.

#Define sweep frequency range and step amount
freq = np.arange(args.freq_start, args.freq_stop, args.freq_step) 

# Use this variable in case of using a heterodyne mixer
Lo = 49000    #Default Gunn frequency in MHz
# Lo = float(input("Enter Lo frequency [MHz]"))
    
# Set Valon
valon = v5019.V5019()

valon.open_serial_port()
time.sleep(1)

# Set initial frequency
valon.set_frequency(freq[0])

# Enter v5019 output power [dBm]
valon.set_power(args.power)
valon.wave_on()

# Create folder and filename to save log data
if not os.path.exists('./notas'):
    os.mkdir('./notas')

# Create or open outputfile
print(f"file {args.outputfile}\n")
f = open(args.outputfile, 'a')

# Get output power by substracting the attenuation (attenuator coupled directly to the output of the Valon).
pout = float(powerout) - args.attenuation
print("Power out Valon - Attenuator \n", pout)
f.write(f"Data: fValon fMix dBm. Valon output power: {pout}\n") 

# Initialize Spectrum Analyzer communication
mxa = n9020b.N9020B()
fcentering = mxa.set_center_frequency(freq[0])
freqs,pows = mxa.get_trace()
idn = mxa.command('*IDN?')

try:
    for cfreq in freq:
        # Define frequency measured by the spectrum analyzer (mult_factor)
        cfreq12 = MULT_FACTOR * cfreq
        
        # Set the first frequency value in Valon
        valon.set_frequency(cfreq)
        print(f"Current frequency {cfreq} MHz")
        
        # Set behavior of the Spectrum Analyzer, finding and measuring the signal received
        fcentering = mxa.set_center_frequency(cfreq12)
        span = mxa.set_span('10000')
        bw = mxa.set_rbw('1')
        freqs, pows = mxa.get_trace()
        time.sleep(1)
        
        # Find the signal line and center it on the screen
        center_to_marker = mxa.seek_max_and_center(0,0)
        print("Center to marker \n")
        time.sleep(1)
        print("Coarse centering ", center_to_marker , mxa.freq_units)
        span = mxa.set_span('500')
        bw = mxa.set_rbw('0.05')
        time.sleep(1)
        
        # Position the marker in the max value found (line frequency)
        mark1 = mxa.mark_to_max(1)
        
        # Get the position of the marker in frequency (MHz/GHz) and power (dBm)
        mark1b = mxa.get_marker_pos()
        freq, dbm = mark1b
        
        # Save the measured values in the output file
        print("qsyn freq - marker freq - power ", mark1b)
        line = f"{cfreq12} {freq} {dbm}\n"
        print("line ", line)
        f.write(line)
        time.sleep(2)
        
finally:
    f.close()

valon.set_power(-40)
valon.wave_off()
valon.save_state()
valon.close_serial_port()


