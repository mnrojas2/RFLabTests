#!/usr/bin/env python

"""
valon_150ghz_characterize.py - Controls the Valon 5019 via USB, generating a series of frequencies. Reads the N9020B abd stores data.
When used with the AMC-I, and extender, N9020 fcent is fpll * MULT_FACTOR
Original author (Python 2 version): Juan C. Fluxa 20220819
Adaptation to Python 3: mnrojas2
""" 

import os
import sys
import numpy as np
import n9020b
import time
import v5019


# Spectrum Analyzer Parameters
# Use this variable in case of using a heterodyne mixer
Lo = 49000 # Default Gunn frequency in MHz
# Lo = float(input("Enter Lo frequency [MHz]"))

MULT_FACTOR = 12.
NUM_MEAS=3
data_out_fName = 'auto_measurement_data.txt'


if __name__ == '__main__':
    measure_list_fNames = sys.argv[1:]
    
    # Initialize Valon
    valon = v5019.V5019()
    valon.list_available_ports()
    valon.open_serial_port()
    valon.set_frequency()
    valon.set_power()
    valon.wave_on()
    
    # Create folder to save log data
    if not os.path.exists('./notas'):
        os.mkdir('./notas')
    
    # Initialize Spectrum Analyzer serial communication
    f = open('./notas/sa_outputs.txt', 'a')
    freq = 12500 * MULT_FACTOR
    
    mxa = n9020b.N9020B()
    mxa.freq_units = 'GHz'
    # mxa.set_continuous_off()
    mxa.set_span(2)
    mxa.set_continuous_peak_search(True)
    bw = mxa.set_rbw('0.001')
    # mxa.set_ref_level(10) 
    # time.sleep(20)
    mxa.set_ref_level(20)
    
    for measure_list_fName in measure_list_fNames:
        print(f"Starting list {measure_list_fName}")
        with open(measure_list_fName) as f:
            measure_list = f.readlines()
        # measure_list = map(int,measure_list) # py2
        measure_list = [int(x) for x in measure_list]
		
        # mxa = n9020b.N9020B()
        # mxa.freq_units = 'GHz'
        mxa.set_continuous_off()
        mxa.set_span(40)
        mxa.set_continuous_peak_search(True)
        # mxa.set_display_off()
        mxa.set_center_frequency(measure_list[0] / 1e9 * MULT_FACTOR)
        print('First point GHz ', measure_list[0] / 1e9 * MULT_FACTOR)
		
        for i,p in enumerate(measure_list):
            print(f"Point {i} - Command: {p / 1e9 * MULT_FACTOR} GHz")
            
            # integer, fract1, fract2, mod2, fpfd, rf_div_sel, prescaler, cp_bleed = dev.set_freq(p,1)
            valon.set_frequency()
            time.sleep(.01)
            
            # Needed when used with 150 GHz extender
            mxa.set_center_frequency(p / 1e9 * MULT_FACTOR)
            centr_to_marker = mxa.seek_max_and_center(0,0)
            fs = []
            pows = []
			
            for i in range(NUM_MEAS):
                mxa.update_spec()
                power =  mxa.get_marker_y()
                power = 10**(power/10)
                pows.append(power)
        
            print(pows, 10 * np.log10(np.mean(pows)), 10 * np.log10(np.std(pows)))
			
            with open(data_out_fName, 'a') as f:
                f.write(f"{p / 1e9 * MULT_FACTOR} {10 * np.log10(np.mean(pows))} {10 * np.log10(np.std(pows))}\n")
                mxa.set_continuous_on()
                # mxa.set_continuous_peak_search(False)

    valon.wave_off()