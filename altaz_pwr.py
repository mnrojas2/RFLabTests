#!/usr/bin/env python

import os
import argparse
import numpy as np
import datetime as dt
import re
import astropy.units as u
from astropy.io import ascii
from matplotlib import pyplot as plt



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
    # Calculates the value of output power based on ADC readings from the Arduino
    # adc/volt parameters
    adc_bits = 2**10-1 # Resolution ADC
    adc_maxVolt = 3.3  # Maximum Voltage ADC
    
    # polynomial fit parameters to convert from voltage measured to values in dBm
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
        print("old measure adc fix enabled!")
        # Scale old data to new data range
        adcVolt = old2new_prop * adcVolt
        
        # Correct saturation values by adding an estimated value from the curve (defined in excel, for values equal or under 1.2, value is more or less the same)
        # if volt_input > 1.2:
        #    adcVolt = adcVolt + 0.7217 * volt_input - 0.921
    
    # Convert voltage to power dBm based on the equation calculated in Excel
    adc_dBm = a3 * np.power(adcVolt, 3) + a2 * np.power(adcVolt, 2) + a1 * adcVolt + a0
    
    return adc_dBm + rfl

def load_rflogfile(file):
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
def merge_pwr2altaz():
    # Normalize paths to prevent issues with generating strings
    altaz_path = os.path.normpath(args.dir_altaz) # './altaz_files/'
    rflog_path = os.path.normpath(args.dir_rflog) # './logs_campaign/'

    # Load file with table match for Flight and logfile
    fdir = open(args.filetable, 'r') # './altaz_files/FLY-log-table.txt'
    cols = fdir.read()
    all_lines = cols.split('\n')

    # Get all file names by searching the corresponding folders
    filenames = []
    for line in all_lines:
        if line != '' and line[0] != '#':
            fal, frf = line.replace(' ', '').split(',')
            fal_match, frf_match = None, None
            
            # Find altaz file(s)
            fal_matches = [] # List to save all altaz files based on a single flight (multiple telescopes observing the drone at the same time)
            for fname in os.listdir(altaz_path):
                if fal in fname and 'pwr' not in fname:
                    fal_matches.append(fname)
            
            # Find log file
            for fname in os.listdir(rflog_path):
                if frf in fname and 'rfmeasure' in fname:
                    frf_match = fname
            
            # If both files were found, add them to the filenames list
            if len(fal_matches) != 0 and frf_match != None:
                for fal_match in fal_matches:              
                    filenames.append([fal_match, frf_match])



    # For each pair of filenames, calculate its output power and merge it with the rest of altaz data.
    for fpair in filenames:
        f_altaz, f_log = fpair[0], fpair[1]
        
        # Load altaz file
        data_altaz = ascii.read(f'{altaz_path}/{f_altaz}', fast_reader=False)

        # Get time row from altaz
        t_altaz = data_altaz['ctime'].data

        # Load rf logfile
        t_ct1, data_logfile = load_rflogfile(f'{rflog_path}/{f_log}')

        # Get input voltage attenuation setting
        volt_input = data_logfile[:,3].mean()

        # Get rpi time (in UTC) and adc data
        t_rpi = data_logfile[:,0] #+ 3 * 3600
        adc_data = data_logfile[:,4]

        # Filter all points above average to get only the upper side of the chopper
        t_rpi_top = t_rpi[adc_data >= adc_data.mean()]
        adc_top = adc_data[adc_data >= adc_data.mean()]

        # Get the exponential moving average of the output power
        # adc_averaged = ema(adc_top, 1480)
        
        # Fit a projection to adjust slopes
        coefficients = np.polyfit(t_rpi_top-t_rpi_top[0], adc_top, deg=10)    # Fit a 5th-degree polynomial
        adc_averaged = np.polyval(coefficients, t_rpi_top-t_rpi_top[0])      # Adjusted y-values based on projection

        # Convert the averaged list to output power
        adc_dB = convert2dBm(adc_averaged, volt_input=volt_input, old_measure=args.old_measure)
        
        # Print some results
        print(f"Altaz: {f_altaz}, logRF: {f_log}, mean: {adc_dB.mean()}, std: {adc_dB.std()}, min: {adc_dB.min()}, max: {adc_dB.max()}, difference: {adc_dB.max()-adc_dB.min()}.")

        # Interpolate adc values to ctime of altaz
        offskip = 1850
        adc_dB_interpol = np.interp(t_altaz, t_rpi_top[offskip:], adc_dB[offskip:])

        # Add calculated power column to the altaz data
        data_altaz['power'] = adc_dB_interpol * u.dB # FBI, OPEN UP!

        # Save the new altaz file
        print(f"Saving data in '{altaz_path}/{os.path.splitext(os.path.basename(f_altaz))[0]}_pwr2{os.path.splitext(f_altaz)[1]}'")
        data_altaz.write(f'{altaz_path}/{os.path.splitext(os.path.basename(f_altaz))[0]}_pwr2{os.path.splitext(f_altaz)[1]}', overwrite=True)

        if args.plot:
            # Plot ADC signals raw and averaged
            plt.figure()
            plt.plot(t_rpi_top, adc_top, '.')
            plt.plot(t_rpi_top, adc_averaged, '-.')
            plt.plot(t_rpi_top[offskip:], adc_averaged[offskip:], '-.')
            plt.xlabel('time [s]')
            plt.ylabel('Quantized Signal [0-1023]')
            plt.title('ADC signal vs RaspberryPi time')
            plt.tight_layout()

            # Plot Output power vs measured time (Raspberry Pi)
            plt.figure()
            plt.plot(t_rpi_top, adc_dB, '-.')
            plt.plot(t_rpi_top[offskip:], adc_dB[offskip:], '-.')
            plt.xlabel('time [s]')
            plt.ylabel('Power [dBm]')
            plt.title('Calculated Output power vs RaspberryPi time')

            # Plot Output power vs altaz time (after interpolation)
            plt.figure()
            plt.plot(t_altaz, adc_dB_interpol, '-.')
            plt.xlabel('ctime [s]')
            plt.ylabel('Power [dBm]')
            plt.title('Output power vs (Altaz) ctime')

            plt.show()
            

if __name__ == '__main__':
    # Initialize parser
    parser = argparse.ArgumentParser(description='Reads ADC data from logfile, converts it to output power (dBm) and adds it to a new column in the corresponding Altaz file.')
    parser.add_argument('dir_altaz', type=str, help='Directory of the folder containing the Altaz files.')
    parser.add_argument('dir_rflog', type=str, help='Directory of the folder containing the RF source log files.')
    parser.add_argument('filetable', type=str, help='Name of the txt file that has all the matching pairs of altaz and RF source log files.')
    parser.add_argument('-p', '--plot', action='store_true', default=False, help='Enable plots of all matches.')
    parser.add_argument('-om', '--old_measure', action='store_true', default=False, help='Enables fix for measures that happened before fixing the amplifier range.')
    
    # Load argparse arguments
    args = parser.parse_args()
    
    # Run main
    merge_pwr2altaz()