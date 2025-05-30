import numpy as np
import time, os, csv, re
from matplotlib import pyplot as plt
#from hovercal.library_rep.other import *

#plt.ion()

os.environ['TZ'] = 'UTC'



'''
Website with best description of columns I can find:
https://datfile.net/DatCon/fieldsV3.html

fname: which file without the extension

skip: Some of the files have blank data in the first few lines which crashes the loader, skip resolves this without missing useful data as this is well before launch
'''

time_fields = ['Clock:offsetTime', 'GPS:Time', 'GPS:Date']
gps_fields = [['RTKdata:Lon_P', 'RTKdata:Lat_P', 'RTKdata:Hmsl_P'],
              ['RTKdata:Lon_S', 'RTKdata:Lat_S', 'RTKdata:Hmsl_S'],
              ['GPS:Long', 'GPS:Lat', 'GPS:heightMSL'],
              ['IMU_ATTI(0):Longitude', 'IMU_ATTI(0):Latitude', 'IMU_ATTI(0):alti:D'],
              ['IMU_ATTI(1):Longitude', 'IMU_ATTI(1):Latitude', 'IMU_ATTI(1):alti:D'],
              ['IMU_ATTI(2):Longitude', 'IMU_ATTI(2):Latitude', 'IMU_ATTI(2):alti:D'],
              ['IMUCalcs(0):Long:C', 'IMUCalcs(0):Lat:C', 'IMUCalcs(0):height:C'],
              ['IMUCalcs(1):Long:C', 'IMUCalcs(1):Lat:C', 'IMUCalcs(1):height:C'],
              ['IMUCalcs(2):Long:C', 'IMUCalcs(2):Lat:C', 'IMUCalcs(2):height:C'],
              ]



def HHMMSS2hms(hms):
    """
    Separete the hms int into three components
    """
    HH = (hms - hms % 10000) / 10000
    mm = hms - HH * 10000
    MM = (mm - mm % 100) / 100
    SS = mm % 100
    return HH, MM, SS




def GPSDateTime2ctime(ymd, hms, timezone=0):
    """
    Convert from HHMMSS to absolute seconds
    """
    YYYY = (ymd - ymd % 10000) / 10000
    mm = ymd - YYYY * 10000
    MM = (mm - mm % 100) / 100
    DD = mm % 100
    hh, mm, ss = HHMMSS2hms(hms)
    ctime = []
    if isinstance(ymd, (int, float)):
        date = (int(YYYY), int(MM), int(DD),
                int(hh), int(mm), int(ss), 0, 0, 0)
        return time.mktime(date) - timezone*3600
    else:
        for i in range(len(YYYY)):
            date = (int(YYYY[i]), int(MM[i]), int(DD[i]),
                    int(hh[i]), int(mm[i]), int(ss[i]), 0, 0, 0)
            ctime.append(time.mktime(date) - timezone*3600)
        return np.array(ctime)


def reconstruct_drone_ctime(ymd, hms, clock, timezone=0):
    """
    Reconstruct ctime using the GPS time data (resolution of a second) and the internal clock (relative time)

    Args:
        ymd: YYYYMMDD
        hms: HHMMSS
        clack: float. Time from the internal IMU clock
        timezone: in hours
    """
    gps_secs = GPSDateTime2ctime(ymd, hms, timezone=timezone)

    # Find instants in which the seconds are updated
    tags = np.where(np.diff(gps_secs) > 0.5)[0] + 1

    # Correct the GPS time adding the fractions of a second from the IMU clock
    ctime = clock + np.mean(gps_secs[tags] - clock[tags])
    return ctime


def flatten(lists):
    fl = []
    for sublist in lists:
        for item in sublist:
            fl.append(item)
    return fl


def print_csv_fields(log_file):
    csvfile = open(log_file, encoding="utf-8")
    reader = csv.DictReader(csvfile)
    print(reader.fieldnames)


class droneData(object):
    def __init__(self, log_file, fields=None, timezone=0, **kwargs):
        if fields == None: fields = gps_fields[0]
        self.load(log_file, fields=fields, timezone=timezone, **kwargs)
        self.test = 1

    def load(self, log_file, skip=100, skip2=0, fields=gps_fields[0], timezone=0, **kwargs):
        if np.ndim(fields) > 1: fields = flatten(fields)
        self.fields = fields
        fields = time_fields + fields

        csvfile = open(log_file, encoding="utf-8")
        reader = csv.DictReader(csvfile)
        data = []

        for i, row in enumerate(reader):
            if i < skip:
                continue
            # Current loading, have to alter this to change which data is loaded form teh giant csv files
            one_row = [row[fld] for fld in fields]
            data.append(one_row)

        # Loads as strings, this converts to floats
        flight_arr = np.array(data)
        flight_arr[flight_arr == ''] = 'NaN'
        flight_arr = flight_arr.astype(float)
        flight_arr = flight_arr[np.isnan(flight_arr).sum(axis=1) == 0]
        flight_arr = flight_arr[(flight_arr == 0).prod(axis=1) == 0]
        flight_arr = flight_arr[skip2:]

        # Add ctime information
        ctime = reconstruct_drone_ctime(flight_arr[:,2], flight_arr[:, 1], flight_arr[:, 0], timezone=timezone)

        # Find refresh data frames
        dtags = np.where(np.diff(flight_arr[:, 3]) != 0)[0]+1

        # Store ctime
        self.ct = ctime[dtags]

        # Store data in array
        self.data = flight_arr[dtags, 3:]

        # Store auxiliary time data
        self.timedata = flight_arr[dtags, :3]

    def plot(self, index, ylabel=None, title=None, new=True):
        if new: plt.figure()
        plt.plot(self.ct, self.data[:,index])
        plt.xlabel("ctime [s]")
        plt.ylabel(ylabel)
        plt.title(title)


def YMDHMS2ctime(YMDHMS, timezone=0):
    valid = YMDHMS != "-1"
    ctimes = -np.ones(len(YMDHMS))
    ymds = []
    hmss = []
    fracs = []
    for ymdhms in YMDHMS[valid]:
        YY, MM, DD, HH, mm, SS = ymdhms.split(":")
        if "." in SS:
            ss, frac = SS.split(".")
        else: ss = SS, frac = "0"
        ymds.append(int(YY+MM+DD))
        hmss.append(int(HH+mm+ss))
        fracs.append(float("0." + frac))
    ctimes[valid] = GPSDateTime2ctime(np.array(ymds), np.array(hmss), timezone=timezone)
    ctimes[valid] += np.array(fracs)
    return ctimes


def LED(status):
    led = []
    for st in status:
        if st[0] == "r":
            led.append(1)
        elif st[0] == "g":
            led.append(2)
        else:
            led.append(0)
    return np.array(led)


def INCL(incl_data):
    idata = []
    for incl in incl_data:
        if len(incl) == 28:
            pitch = decode_incl_word(incl[8:14])
            roll = decode_incl_word(incl[14:20])
            yaw = decode_incl_word(incl[20:26])
            idata.append([pitch, roll, yaw])
        else:
            idata.append([-1, -1, -1])
    return np.array(idata)


def decode_incl_word(incl_word):
    """

    """
    #print(incl_word)
    if incl_word[0] == "0": sign = 1
    else: sign = -1
    angle = float(incl_word[1:4]) + float(incl_word[4:6])/100
    return sign * angle


def parse_source_logfile(filename, skip=5, show_labels=True, timezone=0):
    """
    Reads and parse data from RF measurements logfile
    """
    # Get the header data (saved in the first 5 rows)
    with open(filename, "r") as logfile:
        # Get the labels of the columns
        labels = logfile.readline()
        labels = labels.split(",")
        
        # Raise an error if the length of the labels list is not equal to the list described in log_file_labels
        assert len(labels) >= len(log_file_labels)
        
        # Get configFile data used for the creation of that logfile
        header = logfile.readline().replace(" ","")
        configstr = re.findall(r'\{(.*?)\}', header)[0]
        configstr = configstr.split(',')

        # Get every item on the configFile list saved in a dictionary
        config = {}
        for h in configstr:
            name, value = h.replace("'", '').split(":")
            try:
                config[name] = eval(value)
            except:
                config[name] = value
        
        # Get Arduino and Drone interrupt flag times
        arduino_drone = logfile.readline()
        arduino_drone = re.findall(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+", arduino_drone)
        arduino_drone = [x.replace(" ", ":").replace("-", ":") for x in arduino_drone]
        
        # Get ctime when Arduino started timer
        arduino_ctime = YMDHMS2ctime(np.array([arduino_drone[0]]), timezone=timezone)
        mult = 64 # Hardcoded value from RPi_serial.ino
        period = 1/(config['arduino_timer_basefreq'] * mult)
        
        # If a measurement of the drone sync flag exists, get the ctime of it
        if len(arduino_drone) > 1:
            drone_sync = YMDHMS2ctime(np.array([arduino_drone[1]]), timezone=timezone)
        else: drone_sync = 0
        
        # Get Valon interrupts (phase)
        valon_times = logfile.readline()
        valon_times = re.findall(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+", valon_times)
        valon_times = np.array([x.replace(" ", ":").replace("-", ":") for x in valon_times])
        
        # Get ctime of those interrupts
        valon_ctimes = YMDHMS2ctime(valon_times, timezone=timezone)
        
        # Get average period and std of valon flag ctimes
        valon_periods = np.diff(valon_ctimes)
        valon_av_periods = np.mean(valon_periods)
        valon_std_periods = np.std(valon_periods)
        
        # Get Camera initial time
        cam_start = logfile.readline()
        cam_start = float(cam_start.split(":")[1][:-2])
        config['cam_start'] = cam_start
        
    logfile.close()

    # Get columns data
    data = {}
    for key in log_file_labels.keys():
        if show_labels:
            print(key)
        par = log_file_labels[key]
        raw_data = np.loadtxt(filename, skiprows=skip, delimiter=",", usecols=(par["col"]), dtype=par["dtype"])
        
        # If the column corresponds to ArduinoCounter, convert the value to ctime by using 'arduino_ctime' and 'period'
        if key == 'ArduinoCounter': raw_data = arduino_ctime + (raw_data-1)*period
        
        data[key] = par["eval"](raw_data)
    return data, config


def adc2dBm(data, ctime, adc_bits=1023, adc_maxVolt=3.3, polyfit=[-34.203, 29.364, -15.533, 3.1506]):
    """
    # Calculates the value of power emmitted by the waveguide, 
    based on the measurements done by the diode detector read by an ADC.
    """
    # polynomial fit parameters 
    # * Based on measurements made with the spectrum analyzer and diode detector then fitting a 3th order polynomial. *
    # y = a3*x**3 + a2*x**2 + a1*x + a0
    a0, a1, a2, a3 = polyfit
    
    # Identify indices of high values (assumption: high values are above a threshold)
    data_high_idx = np.where(data > data.mean())[0]
    data_high_val = data[data_high_idx]

    # Interpolate missing values
    data_high = np.interp(np.arange(len(data)), data_high_idx, data_high_val)

    # Fit a polynomial to adjust slopes
    coefficients = np.polyfit(ctime-ctime[0], data_high, deg=10)    # Fit a 5th-degree polynomial
    adc_fit = np.polyval(coefficients, ctime-ctime[0])                      # Adjusted y-values based on projection
    
    # Get the voltage from all adc readings
    adcVolt = adc_maxVolt * (adc_fit / adc_bits)
    
    # Convert voltage to power dBm
    adc_dBm = a3 * np.power(adcVolt, 3) + a2 * np.power(adcVolt, 2) + a1 * adcVolt + a0
    
    # Return the power in dBm and the interpolated array of bits that produces it
    return adc_dBm, adc_fit


def same(data): return data

log_file_labels = {
    "DateTimeRPI": {"dtype": str, "eval": YMDHMS2ctime, "col": 0},
    "ArduinoCounter": {"dtype": int, "eval": same, "col": 1},
    "ArduinoMicros": {"dtype": int, "eval": same, "col": 2},
    "AttOutputVal": {"dtype": float, "eval": same, "col": 3},
    "DiodeSignal": {"dtype": float, "eval": same, "col": 4},
    "PPStimer": {"dtype": int, "eval": same, "col": 5},
    "Dronetimer": {"dtype": str, "eval": same, "col": 6}
}

def read_gimbal_logfile(filename, skip=10, timezone=0):
    """
    Reads logfile recorded by the source micro-computer
    """
    YMD = int(os.path.basename(filename).split("_")[1]) + 20000000
    data = np.loadtxt(filename, delimiter=",", skiprows=skip)
    HMS = data[:,1]
    clock = data[:,0] * 0.0025
    YMD = np.repeat(YMD, len(HMS))
    gps_secs = GPSDateTime2ctime(YMD, HMS, timezone=timezone)
    ctime = reconstruct_drone_ctime(YMD, HMS, clock, timezone=timezone)
    return ctime, data[:,2:]


def interpolate(new_ctime, ctime, data):
    """
    Interpolates a dataset according to its ctime
    """
    intdata = []
    for d in data.T:
        intdata.append(np.interp(new_ctime, ctime, d, left=0, right=0))
    return np.array(intdata).T

def GWkGMS2ctime(dataframe):
    """
    Get time data from Black Square drone dataframe and convert it to ctime
    """

    # Define GPS Epoch (January 6, 1980) for GWk
    gps_epoch = np.datetime64("1980-01-06T00:00:00")

    # Get GPS time of all rows GMS & GWk form
    gps_data = dataframe.gps[['GMS','GWk']].to_numpy()

    # Convert GWk and GMS to UTC time
    utc_times = gps_epoch + np.timedelta64(7, "D") * gps_data[:, 1] + np.timedelta64(1, "ms") * gps_data[:, 0] - np.timedelta64(37, "s")

    # Convert to Unix timestamp (ctime)
    ctime_values = (utc_times - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")

    return utc_times, ctime_values


def load_wtrlog(filename, timezone=0):
    """
    Load parsed data from weather sensor logfile
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
