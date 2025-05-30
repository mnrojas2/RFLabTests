#!/usr/bin/env python

import os
from droneData import *


filename = './rf_measures250501/logfile_0502_132918_rfmeasure.txt'

print(os.path.exists(filename))

data, configFile_data = parse_source_logfile(filename)

ctime = data['ArduinoCounter']
diode_signal = data['DiodeSignal']

print(configFile_data['cam_start'])

# plt.figure()
# plt.plot(data['ArduinoCounter'], 'o-')

# plt.figure()
# plt.plot(np.diff(data['ArduinoCounter']), 'o-')

plt.figure()
plt.plot(data['DiodeSignal'], 'o-')

# Example array with periodic high and low values

# Identify indices of high values (assumption: high values are above a threshold)
diode_high_idx = np.where(diode_signal > diode_signal.mean())[0]
diode_high_val = diode_signal[diode_high_idx]

# Interpolate missing values
diode_signal_high = np.interp(np.arange(len(diode_signal)), diode_high_idx, diode_high_val)

# Fit a polynomial to adjust slopes
coefficients = np.polyfit(ctime-ctime[0], diode_signal_high, deg=10)    # Fit a 5th-degree polynomial
adc_fit = np.polyval(coefficients, ctime-ctime[0])                      # Adjusted y-values based on projection


diode_dBm, diode_adcfit = adc2dBm(data['DiodeSignal'], ctime)

print(adc_fit-diode_adcfit)

# # Get only high values (output power)
# ctime_top = ctime[diode_signal >= diode_signal.mean()]
# top_adc = diode_signal[diode_signal >= diode_signal.mean()]

# # Fit a projection to adjust slopes
# coefficients_b = np.polyfit(ctime_top-ctime_top[0], top_adc, deg=10)    # Fit a 5th-degree polynomial
# adc_fit_b = np.polyval(coefficients_b, ctime_top-ctime_top[0])          # Adjusted y-values based on projection

# a: with interpolation of low values
# b: no interpolation at all, just shorter array

plt.figure()
plt.plot(ctime,diode_signal, 'o-', label='raw_data')
plt.plot(ctime,diode_signal_high, 'o-', label='interpolated_data')
plt.plot(ctime, adc_fit, label='interpolated_fit')
plt.legend()

plt.figure()
plt.plot(ctime, diode_dBm)
plt.show()