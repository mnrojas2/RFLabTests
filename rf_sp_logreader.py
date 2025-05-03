#!/usr/bin/env python

import os
import argparse
import numpy as np
import datetime as dt
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


def main():
    filetxt = open(args.file, 'r')
    cols = filetxt.read()
    dtimes = cols.split('\n')
    
    time_cols = []
    data_cols = []
    
    # Read txt data and save in list
    for line in dtimes:
        line = line[:-1]
        if line != '' and line[0] != '#':
            # DateTimeRPI,ArduinoCounter,AttOutputVal,DiodeSignal,PPStimer,Dronetimer
            try:
                rpi_time, ard_timer, att_volt, adc_read = line.split(',')
            except:
                print(f"Line is {line}")
                input("ctrl + c")
            # Convert datetime to timestamp
            try:
                time_dt = dt.datetime.strptime(rpi_time, "%Y:%m:%d:%H:%M:%S.%f").timestamp()
            except ValueError:
                time_dt = dt.datetime.strptime(rpi_time, "%Y:%m:%d:%H:%M:%S").timestamp()
            
            time_cols.append(rpi_time)
            data_cols.append([time_dt, ard_timer, float(att_volt), int(adc_read)])

    # Convert data list into numpy array
    data_cols = np.array(data_cols, dtype=np.float64)
    
    print(f"Voltage attenuation: {np.round(data_cols[:,2].mean(), 1)} [V]. Mean: {data_cols[:,3].mean()}. Standard deviation: {data_cols[:,3].std()}")
    
    # Exponential moving average
    adc_read_ema = ema(data_cols[:-1,3], 100)
    
    # Fit a projection to adjust slopes
    coefficients = np.polyfit(data_cols[:-1,0]-data_cols[0,0], data_cols[:-1,3], deg=10)    # Fit a 5th-degree polynomial
    adc_fit = np.polyval(coefficients, data_cols[:-1,0]-data_cols[0,0])       # Adjusted y-values based on projection  # """
    
    plt.figure()
    plt.hist(data_cols[:,3], bins=20)
    
    plt.figure()
    plt.scatter(data_cols[:,1], data_cols[:,3])
    plt.plot(data_cols[:-1,1], adc_read_ema, color='red', label='adc with exponential moving average')
    plt.plot(data_cols[:-1,1], adc_fit, color='green', label='adc with polynomial fit')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    # Initialize parser
    parser = argparse.ArgumentParser(description='Reads data from txt files and plots ADC histogram and shows average and std.')
    parser.add_argument('file', type=str, help='Name of the txt file to read.')
    
    # Load argparse arguments
    args = parser.parse_args()
    
    # Main
    main()