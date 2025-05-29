from droneData import *

def load_ledlog(filename, timezone=0):
    """
    Load data from blink camera logfile
    * Check if timezone is correctly set comparing one of the samples in <https://esqsoft.com/javascript_examples/date-to-epoch.htm>
    """
    data = np.loadtxt(filename, delimiter=",",dtype=str)
    ctime = YMDHMS2ctime(data[:,0], timezone=timezone)
    return ctime, data[:,1]

import numpy as np

def load_wtrlog(filename, timezone=0):
    """
    Load data from weather sensor logfile
    """
    # Define all columns and their value types
    column_types = [('ctime', 'U26'), ('Temperature', 'f8'), ('Humidity', 'f8'), ('Pressure', 'f8'), ('Altitude', 'f8')]
    
    # Get data from the file
    data = np.loadtxt(filename, delimiter=',', dtype=column_types)
    
    # Convert the time data to ctime form (seconds since 1970-01-01)
    ctime = YMDHMS2ctime(data['ctime'], timezone=timezone)
    
    # Reconvert the array to non-structured form
    weather_data = np.column_stack([data[name] for name in data.dtype.names if name != 'ctime'])
    
    # Return both lists
    return ctime, weather_data


if __name__ == '__main__':
    filename = './rf_measures250501/log144_12_weather.txt'
    ctime, w_data = load_wtrlog(filename, timezone=-3)
    
    fig, ax1 = plt.subplots()
    line1, = ax1.plot(ctime, w_data[:,0], 'r-', label='temperature')
    ax1.set_ylabel('°C', color='red')
    ax2 = ax1.twinx()
    ax2.spines['right'].set_position(('outward', 120))  # Offset fourth axis
    line2, = ax2.plot(ctime, w_data[:,1], 'm-', label='humidity')
    ax2.set_ylabel('%', color='purple')
    ax3 = ax2.twinx()
    line3, = ax3.plot(ctime, w_data[:,2], 'b-', label='pressure')
    ax3.set_ylabel('kPa', color='blue')
    ax4 = ax3.twinx()
    ax4.spines['right'].set_position(('outward', 60))  # Offset third axis
    line4, = ax4.plot(ctime, w_data[:,3], 'g-', label='altitude')
    ax4.set_ylabel('m', color='green')
    
    handles = [line1, line2, line3, line4]
    labels = [handle.get_label() for handle in handles]
    ax1.legend(handles, labels, loc='upper left')
    plt.show()