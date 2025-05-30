#!/usr/bin/env python

import os
from droneData import *


filename = './rf_measures250501/logfile_0502_132918_rfmeasure.txt'

print(os.path.exists(filename))

data, configFile_data = parse_source_logfile(filename)

ctime = data['ArduinoCounter']
diode_signal = data['DiodeSignal']

print(configFile_data['cam_start'])

plt.figure()
plt.plot(data['DiodeSignal'], 'o-')

# Calculate power in dBm
diode_dBm, diode_adcfit = adc2dBm(data['DiodeSignal'], ctime)

plt.figure()
plt.plot(ctime,diode_signal, 'o-', label='raw_data')
plt.plot(ctime,diode_adcfit, label='interpolated_fit')
plt.legend()

plt.figure()
plt.plot(ctime, diode_dBm)
plt.show()