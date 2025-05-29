#!/usr/bin/env python

import os
from droneData import *


filename = './rf_measures250501/log144_12_rfmeasure.txt'

print(os.path.exists(filename))

data, configFile_data = parse_source_logfile(filename)

print(configFile_data['cam_start'])

plt.figure()
plt.plot(data['ArduinoCounter'], 'o-')

plt.figure()
plt.plot(np.diff(data['ArduinoCounter']), 'o-')
plt.show()