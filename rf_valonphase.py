#!/usr/bin/env python

import os
import argparse
import numpy as np
import glob
import re
import datetime as dt
from matplotlib import pyplot as plt

# Initialize parser
parser = argparse.ArgumentParser(description='Reads data from txt files and plots ADC histogram and shows average and std.')
parser.add_argument('file', type=str, help='Name of the txt file to read.')


def statistics(data):
    # Calculate key statistics
    summary = {
        "Count": data.size,
        "Mean": np.mean(data),
        "Std Dev": np.std(data),
        "Min": np.min(data),
        "25%": np.percentile(data, 25),
        "50% (Median)": np.median(data),
        "75%": np.percentile(data, 75),
        "Max": np.max(data)
    }

    # Print the summary
    for stat, value in summary.items():
        print(f"{stat}: {value}")


# Main
args = parser.parse_args()

# Check if argument is a single file, or a folder
if os.path.isdir(args.file):
    print(f'Loading all rfmeasure.txt files in {args.file}')
    file_list = sorted(glob.glob(f'{os.path.normpath(args.file)}/*rfmeasure.txt'), key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
elif 'rfmeasure' in args.file:
    print(f'Loading {args.file}')
    file_list = args.file
else:
    raise NameError("No files were found")

total_differences = np.array([])

for nfile in file_list:
    # Load txt file
    filetxt = open(nfile, 'r')

    # Get all rows
    cols = filetxt.read()
    dtimes = cols.split('\n')

    # Get a list of the phases of the valon in string format (4th row of the logfile)
    valon_phases_str = (dtimes[3].split('[')[-1][:-2]).replace("'", "").split(', ')

    # Convert all strings to datetime format
    valon_phases = [dt.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S.%f') for date_str in valon_phases_str]

    # Get the difference between all datetimes to know the consistency of phase measuring
    differences = np.array([(valon_phases[i+1] - valon_phases[i]).total_seconds() for i in range(len(valon_phases) - 1)])
    total_differences = np.append(total_differences, differences)

statistics(total_differences)

plt.figure()
plt.hist(total_differences, bins=50)
plt.show()

