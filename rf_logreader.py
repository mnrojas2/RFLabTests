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


# Load file and extract all lines
# file = 'log_output_0902_154116' # old logfile format
file = 'log_measure' # new logfile format
filetxt = open(f'{file}.txt', 'r')
cols = filetxt.read()
dtimes = cols.split('\n')

time_cols = []
data_cols = []

# Read txt data and save in list
for line in dtimes:
    if line != '' and line[0] != '#':
        # DateTimeRPI,ArduinoCounter,ArduinoMicros,AttOutputVal,DiodeSignal,PPStimer,Dronetimer
        rpi_time, ard_timer, ard_micros, att_volt, adc_status, pps_timer, drone_timer = line.split(',')
        # Convert datetime to timestamp
        try:
            time_dt = dt.datetime.strptime(rpi_time, "%Y:%m:%d:%H:%M:%S.%f").timestamp()
        except ValueError:
            time_dt = dt.datetime.strptime(rpi_time, "%Y:%m:%d:%H:%M:%S").timestamp()
        
        time_cols.append(rpi_time)
        data_cols.append([time_dt, ard_timer, float(int(ard_micros)/1e6), float(att_volt), int(adc_status), int(pps_timer), int(drone_timer)])

# Convert data list into numpy array
data_cols = np.array(data_cols, dtype=np.float64)

# Time sampling analysis
t_rpi = data_cols[:,0]    # Time from Raspberry Pi
t_ard = data_cols[:,2]    # Time from Arduino micros
t_timer = data_cols[:,1]  # Samples taken at timer frequency

dt_ard = t_ard[1:] - t_ard[:-1]                         # Difference in time of t_ard
dt_timer = t_timer[1:] - t_timer[:-1]                   # Difference in time of t_timer
ddt_timer = (dt_timer[1:] - dt_timer[:-1]) / dt_ard[1:] # Derivative of dt_timer

# Plot timer vector vs RPi time
plt.figure()
plt.plot(t_rpi[1:], dt_timer, '-o')
plt.title('Sampling timer vs RPi Time')

# Plot timer vector vs Arduino micros
plt.figure()
plt.plot(t_ard[1:], dt_timer, '-o')
plt.title('Sampling timer vs Arduino Micros')

# Plot derivative of timer vector vs Arduino Micros
plt.figure()
plt.plot(t_ard[2:], ddt_timer, '-o')
plt.title('Sampling timer derivative vs Arduino Micros')
# plt.show()




# Signal analysis

# Get adc signal column
adc_signal = data_cols[:,4]

# Sampling multiplier
mult = 6

adc_signal_bin = np.where(adc_signal > 512, 1, 0)

# Print number of HIGH values, number of LOW values and shape of the vector (HIGH+LOW)
print(np.sum(adc_signal_bin), np.sum(np.where(adc_signal_bin == 0, 1, 0)), adc_signal_bin.shape)

# Plot adc signal vector vs RPi time
plt.figure()
plt.plot(t_rpi[:adc_signal.shape[0]], adc_signal, 'o') # '-o'
plt.title('Diode signal vs RPi Time')

# Plot adc signal vector vs Arduino micros
plt.figure()
plt.plot(t_ard[:adc_signal.shape[0]], adc_signal, 'o') # '-o'
plt.title('Diode signal vs Arduino Micros')

# Plot adc signal vector vs Arduino timer
plt.figure()
plt.plot(t_timer[:adc_signal.shape[0]], adc_signal, 'o') # '-o'
plt.title('Diode signal vs Arduino Timer')
# plt.show()

top_adc = adc_signal[adc_signal >= adc_signal.mean()]
top_adc_ema = ema(top_adc, 100)
print(top_adc_ema.mean())

plt.figure()
plt.scatter(t_timer[:len(top_adc)], top_adc)
plt.plot(t_timer[:len(top_adc)], top_adc_ema, color='r')
plt.show()


# Fourier transform of adc_signal
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

# col_us = data_cols[1:,1] - data_cols[:-1,1]
# uq, idx = np.unique(data_cols[:,1], return_index=True)

# data_cols_edges = data_cols[idx, :]
# # data_cols_edges = data_cols_edges[::2]

# diff_uq = data_cols_edges[1:,1] - data_cols_edges[:-1,1]
# wave = np.zeros(data_cols_edges.shape[0])
# wave[1::2] = 1

# plt.figure()
# plt.plot(data_cols_edges[1:,0], diff_uq)
# plt.title("Periods of time vs RPi timestamp")

# plt.figure()
# plt.plot(diff_uq, '-o')
# plt.title("Periods of time between edges (Arduino).")

# plt.figure()
# plt.hist(diff_uq)
# plt.title("Histogram of the periods of time between edges (Arduino)")

# plt.figure()
# plt.hist((data_cols_edges[1:,0]-data_cols_edges[:-1,0]))
# plt.title("Time periods of the RPi receiving serial data after an edge change.")

# plt.figure()
# plt.hist((data_cols[1:,0]-data_cols[:-1,0]), bins=100)
# plt.title("Time periods of the RPi receiving serial data complete.")
# plt.show()



#######
# nuevos
# extraer la columna de ADC de todos los vuelos nuevos -> ajustar los valores a potencia de salida
#
# viejos
# extraer la columna de ADC de todos los vuelos viejos -> corregir problemas de saturación utilizando la curva estimada escalada a formato curva nueva -> ajustar los valores a potencia de salida
#######