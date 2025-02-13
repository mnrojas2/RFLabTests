#!/usr/bin/env python

import os
import argparse
import numpy as np
import datetime as dt
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Reads data from txt files and plots ADC histogram and shows average and std.')
parser.add_argument('file', type=str, help='Name of the txt file to read.')
# parser.add_argument('-o', '--output', type=str, metavar='file', default=None, help='Name of the file that will contain the received data (Optional).')

def main():
    args = parser.parse_args()
    
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
    
    plt.figure()
    plt.hist(data_cols[:,3], bins=20)
    
    plt.figure()
    plt.scatter(data_cols[:,1], data_cols[:,3])
    plt.show()

if __name__ == '__main__':
    main()