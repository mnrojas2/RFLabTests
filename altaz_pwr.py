#!/usr/bin/env python

import os
import argparse
import numpy as np
import datetime as dt
import re
from astropy.io import ascii
from astropy.table import MaskedColumn
from matplotlib import pyplot as plt

# Load altaz_FLY ecsv file
# Load matching log_rfmeasure file
# Get time and ADC data from log_rfmeasure, filter data with ema()
# Time can be from Raspberry Pi or ctime from arduino

# Calculate Power (in dBm) in the waveguide and interpolate to altaz ctime and add it as a new column
# Save the altaz file with the new column as a file with the same name, plus "_pwr"

def ema(npdata, window):
    # Calculates the exponential moving average
    # Get the factor based on number of elements that will contain the moving window
    alpha = 2 / (window + 1)
    
    # Initialize the new vector and the first EMA value
    ema_values = np.zeros_like(npdata)
    ema_values[0] = npdata[0]  
    
    for i in range(1, npdata.shape[0]):
        ema_values[i] = alpha * npdata[i] + (1 - alpha) * ema_values[i - 1]
    
    return ema_values

def convert2dBm(adcval, volt_input, old_measure=False):
    # Calculates the value of 
    # adc/volt parameters
    adc_bits = 2**10-1 # Resolution ADC
    adc_maxVolt = 3.3  # Maximum Voltage ADC
    
    # polynomial fit parameters
    # y = a3*x**3 + a2*x**2 + a1*x + a0
    a0 = -78.851
    a1 = 18.08
    a2 = -8.1663
    a3 = 1.5023
    
    # RF Loss (Distance: 80 cm, Frequency: 150 GHz, Transmitter Gain: 0 dB, Receiver Gain: 25.2 dB)
    rfl = 48.82
    
    # Proportional change to convert old range data (dec 8-13) to new range data (dec 14-15)
    old2new_prop = 22.754 / 33.731 # amplifier gain: old / new
    
    # Get the voltage from all adc readings
    adcVolt = adc_maxVolt * (adcval / adc_bits)
    
    # Fix saturation issue with old measurements (dec 6 to dec 13)
    if old_measure:
        # Scale old data to new data range
        adcVolt = old2new_prop * adcVolt
        
        # Correct saturation values by adding an estimated value from the curve (defined in excel, for values equal or under 1.2, value is more or less the same)
        if volt_input > 1.2:
            adcVolt = adcVolt + 0.7217 * volt_input - 0.921
    
    # Convert voltage to power dBm based on the equation calculated in Excel
    adc_dBm = a3 * np.power(adcVolt, 3) + a2 * np.power(adcVolt, 2) + a1 * adcVolt + a0
    
    return adc_dBm + rfl

def load_logfile(file):
    # Load rfmeasure file
    filetxt = open(file, 'r')
    cols = filetxt.read()
    all_lines = cols.split('\n')

    data_rows = []
    init_ct_time = 0
    datetime_pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}"

    # Read txt data and save in list
    for line in all_lines:
        if line != '' and line[0] != '#':
            # DateTimeRPI,ArduinoCounter,ArduinoMicros,AttOutputVal,DiodeSignal,PPStimer,Dronetimer
            rpi_time, ard_timer, ard_micros, att_volt, adc_status, pps_timer, drone_timer = line.split(',')
            # Convert datetime to timestamp
            try:
                time_dt = dt.datetime.strptime(rpi_time, "%Y:%m:%d:%H:%M:%S.%f").timestamp()
            except ValueError:
                time_dt = dt.datetime.strptime(rpi_time, "%Y:%m:%d:%H:%M:%S").timestamp()

            data_rows.append([time_dt, ard_timer, float(int(ard_micros)/1e6), float(att_volt), int(adc_status), int(pps_timer), int(drone_timer)])
            
        elif line != '' and line[0] == '#' and 'Arduino time sync' in line:
            # Search for the datetime string in the line
            matches = re.findall(datetime_pattern, line)

            if matches:
                # Extract the matched datetime string
                first_datetime_str = matches[0]
                
                # Convert the string to a datetime object
                init_ct_time = dt.datetime.strptime(first_datetime_str, "%Y-%m-%d %H:%M:%S.%f")

    # Convert data list into numpy array
    data_rows = np.array(data_rows, dtype=np.float64)
    return init_ct_time, data_rows



# Main

# Load argparse arguments
# args = parse.parser()

# Load altaz file
data_altaz = ascii.read('./altaz_files/altaz_FLY771_20241214_satp1_v20250114.ecsv', fast_reader=False)

# Get time row from altaz
t_altaz = data_altaz['ctime'].data

# Load rf logfile
t_ct1, data_logfile = load_logfile('./logs_campaign/logfile_1214_181215_rfmeasure.txt')

# Get input voltage attenuation setting
volt_input = data_logfile[:,3].mean()

# Get rpi time (in UTC) and adc data
t_rpi = data_logfile[:,0]
adc_data = data_logfile[:,4]

# Filter all points above average to get only the upper side of the chopper
t_rpi_top = t_rpi[adc_data >= adc_data.mean()]
adc_top = adc_data[adc_data >= adc_data.mean()]

# Get the exponential moving average of the output power
adc_averaged = ema(adc_top, 1480)
print(adc_averaged.mean())

# Convert the averaged list to output power
adc_dB = convert2dBm(adc_averaged, volt_input=volt_input, old_measure=False)

# Interpolate adc values to ctime of altaz
adc_dB_interpol = np.interp(t_altaz, t_rpi_top, adc_dB)
print(t_altaz[0], t_rpi_top[0])

plt.figure()
plt.plot(t_rpi_top, adc_top, '.')
plt.plot(t_rpi_top, adc_averaged, '-.')

plt.figure()
plt.plot(t_rpi_top, adc_dB, '-.')

plt.figure()
plt.plot(t_altaz, adc_dB_interpol, '-.')
plt.show()