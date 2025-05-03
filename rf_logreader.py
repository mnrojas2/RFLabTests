#!/usr/bin/env python

import os
import argparse
import numpy as np
import datetime as dt
import re
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
    # Calculates the value of 
    # adc/volt parameters
    adc_bits = 2**10-1 # Resolution ADC
    adc_maxVolt = 3.3  # Maximum Voltage ADC
    
    # polynomial fit parameters
    # y = a3*x**3 + a2*x**2 + a1*x + a0
    a0 = -83.083 # -78.851
    a1 = 29.364 #18.08
    a2 = -15.533 #-8.1663
    a3 = 3.1506 #1.5023
    
    # RF Loss (Distance: 80.5 cm, Frequency: 150 GHz, Transmitter Gain: 0 dB, Receiver Gain: 25.2 dB)
    rfl = 48.88
    
    # Proportional change to convert old range data (dec 8-13) to new range data (dec 14-15)
    old2new_prop = 22.754 / 33.731 # amplifier gain: old / new
    
    # Get the voltage from all adc readings
    adcVolt = adc_maxVolt * (adcval / adc_bits)
    
    # Fix saturation issue with old measurements (dec 6 to dec 13)
    if old_measure:
        # Scale old data to new data range
        adcVolt = old2new_prop * adcVolt
        
        # Correct saturation values by adding an estimated value from the curve (defined in excel, for values equal or under 1.2, value is more or less the same)
        #if volt_input > 1.2:
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

def load_wtlogfile(file):
    # Load rfmeasure file
    filetxt = open(file, 'r')
    cols = filetxt.read()
    all_lines = cols.split('\n')

    data_rows = []

    # Read txt data and save in list
    for line in all_lines:
        if line != '' and line[0] != '#':
            # DateTimeRPI,ArduinoCounter,ArduinoMicros,AttOutputVal,DiodeSignal,PPStimer,Dronetimer
            rpi_time, temp, pres, alti, humi = line.split(',')
            # Convert datetime to timestamp
            try:
                time_dt = dt.datetime.strptime(rpi_time, "%Y:%m:%d:%H:%M:%S.%f").timestamp()
            except ValueError:
                time_dt = dt.datetime.strptime(rpi_time, "%Y:%m:%d:%H:%M:%S").timestamp()

            data_rows.append([time_dt, temp, pres, alti, humi])

    # Convert data list into numpy array
    data_rows = np.array(data_rows, dtype=np.float64)
    return data_rows



def main():
    # Load file and extract all lines
    # file = 'log_output_0902_154116' # old logfile format
    init_ctime, data_cols = load_rflogfile(args.file)
    weather_cols = load_wtlogfile('rf_measures250501/log144_12_weather.txt')
    temp = weather_cols[:,:2]

    # Time sampling analysis
    t_rpi = data_cols[:,0]    # Time from Raspberry Pi
    t_ard = data_cols[:,2]    # Time from Arduino micros
    t_timer = data_cols[:,1]  # Samples taken at timer frequency

    dt_ard = t_ard[1:] - t_ard[:-1]                         # Difference in time of t_ard
    dt_timer = t_timer[1:] - t_timer[:-1]                   # Difference in time of t_timer
    ddt_timer = (dt_timer[1:] - dt_timer[:-1]) / dt_ard[1:] # Derivative of dt_timer

    # if args.plot:
    #     # Plot timer vector vs RPi time
    #     plt.figure()
    #     plt.plot(t_rpi[1:], dt_timer, '-o')
    #     plt.title('Sampling timer vs RPi Time')

    #     # Plot timer vector vs Arduino micros
    #     plt.figure()
    #     plt.plot(t_ard[1:], dt_timer, '-o')
    #     plt.title('Sampling timer vs Arduino Micros')

    #     # Plot derivative of timer vector vs Arduino Micros
    #     plt.figure()
    #     plt.plot(t_ard[2:], ddt_timer, '-o')
    #     plt.title('Sampling timer derivative vs Arduino Micros')
    #     plt.show()




    # Signal analysis

    # Get adc signal column
    volt_input = data_cols[:,3].mean()
    adc_signal = data_cols[:,4]

    # Sampling multiplier
    mult = 6

    adc_signal_bin = np.where(adc_signal > 512, 1, 0)

    # Print number of HIGH values, number of LOW values and shape of the vector (HIGH+LOW)
    print(np.sum(adc_signal_bin), np.sum(np.where(adc_signal_bin == 0, 1, 0)), adc_signal_bin.shape)

    # Get only high values (output power)
    t_timer_top = t_timer[adc_signal >= adc_signal.mean()]
    top_adc = adc_signal[adc_signal >= adc_signal.mean()]

    # Get the exponential moving average of the output power
    top_adc_ema = ema(top_adc, 1480) # 10 seconds (sampling frequency = 37*4 Hz)
    
    # Reemplazar promedio con un polinomio de grado 10?
    # Usar rango original de las mediciones pre-14dec sin ajuste de medición 
    # Determinar promedio, desviación estándar, mínimo y máximo y agregarlos al spreadsheet.
    print(top_adc_ema.mean())

    # Convert the averaged list to output power
    adc_dB = convert2dBm(top_adc_ema, volt_input=volt_input, old_measure=args.old_measure)
    
    # Fit a projection to adjust slopes
    coefficients = np.polyfit(t_timer_top, top_adc, deg=10)    # Fit a 5th-degree polynomial
    top_adc_fit = np.polyval(coefficients, t_timer_top)       # Adjusted y-values based on projection  # """
    
    # Convert the averaged list to output power
    adc_dB_fit = convert2dBm(top_adc_fit, volt_input=volt_input, old_measure=args.old_measure)
    
    print(f"logRF: {os.path.basename(args.file)}, mean: {adc_dB.mean()}, std: {adc_dB.std()}, range (max-min): {adc_dB.max()-adc_dB.min()}.")

    offskip = 1850
    if args.plot:
        c=0
        if c == 1:
            # Plot adc signal vector vs RPi time
            plt.figure()
            plt.plot(t_rpi[:adc_signal.shape[0]], adc_signal, 'o') # '-o'
            plt.title('Diode signal vs RPi Time')

            # Plot adc signal vector vs Arduino micros
            plt.figure()
            plt.plot(t_ard[:adc_signal.shape[0]], adc_signal, 'o') # '-o'
            plt.title('Diode signal vs Arduino Micros')

            # Plot adc signal vector vs Arduino timer (VALID)
            plt.figure()
            plt.plot(t_timer[:adc_signal.shape[0]], adc_signal, 'o') # '-o'
            plt.title('Diode signal vs Arduino Timer')
            # plt.show()

        # Plot the output power (based on ADC readings)
        plt.figure()
        plt.scatter(t_timer_top, top_adc)
        plt.plot(t_timer_top, top_adc_ema, color='r', label='Exponential moving average')
        plt.plot(t_timer_top, top_adc_fit, '-.', label='Polynomial fit')
        plt.plot(t_timer_top[offskip:], top_adc_fit[offskip:], '-.', label='Polynomial fit with starting offset')
        plt.title('Output power in ADC values')
        plt.legend()
        
        # Plot the output power (in dB)
        # plt.figure()
        fig, ax1 = plt.subplots()
        ax1.plot(t_timer_top, adc_dB, color='r', label='Exponential moving average')
        ax1.plot(t_timer_top, adc_dB_fit, '-.', label='Polynomial fit')
        ax1.plot(t_timer_top[offskip:], adc_dB_fit[offskip:], '-.', label='Polynomial fit with starting offset')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        plt.legend()
        
        ax2 = ax1.twinx()
        ax2.plot(64*37*(temp[:,0]-temp[0,0]), temp[:,1], color='green', label='Temperature')
        ax2.tick_params(axis='y', labelcolor='tab:green')
        plt.legend()
        
        plt.title('Output power in dBm')
        fig.tight_layout()
        plt.show()



    if args.fourier: 
        # Calculate the Fourier transform of adc_signal
        x = adc_signal
        x_mean = np.mean(x)
        x = x - x_mean
        # x = x[:80000]

        ffreq = 37
        freq = np.fft.fftfreq(x.shape[-1], 1/(ffreq*2*mult))
        x_freq = (np.fft.fft(x))**2

        plt.figure()
        plt.plot(freq, np.abs(x_freq))
        plt.title('FFT of Diode signal')
        plt.show()


if __name__ == '__main__':
    # Initialize parser
    parser = argparse.ArgumentParser(description='Reads data from txt files and plots ADC histogram and shows average and std.')
    parser.add_argument('file', type=str, help='Name of the txt file to read.')
    parser.add_argument('-p', '--plot', action='store_true', default=False, help='Shows time based plots.')
    parser.add_argument('-ft', '--fourier', action='store_true', default=False, help='Shows fourier transform plot.')
    parser.add_argument('-om', '--old_measure', action='store_true', default=False, help='Enables fix for measures that happened before fixing the amplifier range.')
    # parser.add_argument('-o', '--output', type=str, metavar='file', default=None, help='Name of the file that will contain the received data (Optional).')

    # Load argparse arguments
    args = parser.parse_args()
    
    # Main
    main()