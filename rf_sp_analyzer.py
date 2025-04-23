#!/usr/bin/env python

import os
import argparse
import time
import serial
import datetime
import RPi.GPIO as GPIO
import subprocess
import keyword
import board
import v5019
import signal


# Time sync interrupt with drone and Arduino class
class timeSync:
    def __init__(self, signal, max_count=1):
        self.signal = signal        # GPIO port that will receive the interrupt.
        self.timed = False          # Boolean to indicate the process has finished to stop.
        self.time_bck = []          # List that saves the time of all instances when the interrupt reached the device.
        self.max_count = max_count  # Maximum number of interrupts to receive.
        
        # Initialize pin and interrupt
        GPIO.setup(self.signal, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        GPIO.add_event_detect(self.signal, GPIO.RISING, callback=self.signal_callback, bouncetime=2)
        
    def signal_callback(self, signal):
        # Save the system time in a list every time an interrupts gets triggered.
        self.time_bck.append(str(datetime.datetime.now()))
        if (len(self.time_bck) >= self.max_count):
            self.timed = True
        
    def end_interrupt(self):
        # Disable interrupt
        GPIO.remove_event_detect(self.signal)
        time.sleep(0.1)
        
    def wait_for_callback(self):
        # Function that enables a while loop waiting for the first interrupt signal to arrive to continue the script.
        while True:
            if self.timed:
                self.end_interrupt()
                break

# Class to manage the behavior of the while loop to also stop it with "Ctrl+C"
class LoopController:
    def __init__(self):
        self.running = True

    def signal_handler(self, sig, frame):
        # What to do after pressing 'Ctrl' and 'C'
        print("Ctrl+C pressed. Exiting loop...")
        self.running = False

# Button interrupt for shutdown (button has a physical debouncer)
class ButtonOff:
    def __init__(self, signal):
        self.signal = signal        # Pin to where the button is connected
        self.counter = 0            # Pressed button counts
        
        # Setup and activate interruption
        GPIO.setup(self.signal, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        GPIO.add_event_detect(self.signal, GPIO.RISING, callback=self.signal_callback, bouncetime=50)
        
    def signal_callback(self, signal):
        # Interrupt function, just increases the counter and add a small delay
        self.counter += 1
    
    def off_system(self):
        # Function used to shutdown the Raspberry Pi
        if self.counter >= 2:
            return True
        return False
    
    def end_interrupt(self):
        # Deactivates the interrupt
        GPIO.remove_event_detect(self.signal)
        time.sleep(0.1)
    

# Normal Functions
def get_checksum(preframe):
    # Function that obtains the checksum (last 2 digits of the hexadecimal sum) for a given input frame in bytes
    # the output is the original frame with the checksum appended, e.g., in: b'773f21a2' -> out: b'773f21a202'
    prefix = preframe[0:2]
    preframe = preframe[2:]
    temp = 0
    for i in range(len(preframe)//2):
        nib = int(preframe[2*i:2+2*i], 16)
        temp += nib
    if len(hex(temp)) < 5:
        var = b'%02x' % (temp)
    else:
        var = b'%02x' % (int(hex(temp)[-2:], 16))        
        
    frame = prefix + preframe + var.upper()
    return frame

def att_power2volt(power):
    # Converts voltage float value to int in mV
    # FUTURE WORK: This must has the power as input (i.e: -18 dBm and convert it into a integer voltage value in mV)
    volt = int(1000 * power) #int(power) # voltage must be an integer in mV
    return str(volt).zfill(4)



# Main
def main():
    ## GPIO variables
    arduino_reset = 19          # Arduino reset digital port
    rpi_off = 26                # Button digital port
    ardig0 = 23                 # Arduino digital port (IN interrupt)

    # Logfile LED ports
    Led8R = 21
    Led8G = 16

    # Use the Broadcom SOC channel
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    # Set up reset button for RaspberryPi
    GPIO.setup(rpi_off, GPIO.IN)

    # Set up Arduino reset pin 
    GPIO.setup(arduino_reset, GPIO.OUT)
    GPIO.output(arduino_reset, GPIO.LOW)

    # Arduino digital communication setup (RPI GPIO 23 to Arduino Digital 5).
    GPIO.setup(ardig0, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)      # Interrupt ard_sync from Arduino

    # RPI controlled LEDs setup
    GPIO.setup(Led8R, GPIO.OUT)     # Logfile LED red
    GPIO.setup(Led8G, GPIO.OUT)     # Logfile LED green

    # All LEDs start as red:
    GPIO.output(Led8R, GPIO.HIGH)
    GPIO.output(Led8G, GPIO.LOW)
    

    # Set pathfile where this file is and not from where is getting executed.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Create folder and filename to save log data
    if not os.path.exists('./sa_logs'):
        os.mkdir('./sa_logs')
    
    # Define filename for the log file
    if not args.output:
        date_now = str(datetime.datetime.now()).split('.')[0].replace('-', '').replace(':', '').replace(' ', '_')
        filename = f'./sa_logs/sa_test_{date_now}.txt'
    else:
        filename = f'./sa_logs/{args.output}.txt'
    print(f"Logs will be saved in {filename}.")
    
    # Set header to receive in Arduino
    header = "640F"

    # Parse the information and convert it to string
    serial_port    = "/dev/ttyAMA3"
    debug_en       = True            # To read some initialization data from Serial0 in Arduino GIGA.
    en_vln_chopper = args.chopper    # Enable or disable Valon controlled chopper.
    en_ard_chopper = False           # Enable or disable Arduino controlled chopper (deprecated).
    chopper_freq   = 37              # Frequency the signal will be chopped, and as base frequency to initiate the counter in the Arduino.
    
    # Define attenuation voltage for multiplier
    if args.powervoltage >= 3.3:
        att_pwr = 3.3
    elif args.powervoltage <= 0.0:
        att_pwr = 0.0
    else:
        att_pwr = args.powervoltage # 2.0
    att_power_volt = att_power2volt(att_pwr)  # dBm requires convertion to voltage

    # Define Valon parameters
    vfreq = args.frequency
    vpwr  = 6
    vamd  = 30
    vamf  = chopper_freq

    # Make the string with the necessary variables for the arduino and add the checksum value at the end
    arduino_init = (header + f'{int(debug_en)}{int(en_vln_chopper)}{int(en_ard_chopper)}{str(int(chopper_freq)).zfill(4)}{att_power_volt}').encode()
    arduino_init = get_checksum(arduino_init)

    # Reset Arduino
    GPIO.output(arduino_reset, GPIO.HIGH)
    time.sleep(0.1)
    GPIO.output(arduino_reset, GPIO.LOW)
    time.sleep(0.1)

    # Initialize serial port
    ser = serial.Serial(serial_port, 
        baudrate=460800, 
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=1)

    # Reset Serial buffer
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    print("Serial initialization Ready!")

    # Send configInit.txt variables and wait until Arduino acknowledges it
    print(f'Sending {arduino_init}')
    while True:
        ser.write(arduino_init)
        time.sleep(1)
        data = ser.readline()
        if data.decode('utf-8') == "init_out\r\n":
            print("ConfigInit variables set.")
            break
    
    # Start Valon
    valon = v5019.V5019()
    valon.open_serial_port()    # Open Valon serial port
    valon.set_frequency(vfreq)  # Sends the frequency command [Mhz]
    valon.set_power(vpwr)       # Sends the power command [dBm]

    # If Arduino doesn't act as chopper, Valon will
    if en_vln_chopper: 
        valon.set_amd(vamd)     # Sends the AM signal amplitude command [dB]
        valon.set_amf(vamf)     # Sends the AM frequency command [Hz] (47 Hz)
    else:
        valon.set_amd(0)     # Sends the AM signal amplitude command [dB]
        valon.set_amf(1)     # Sends the AM frequency command [Hz] (47 Hz)
    valon.wave_on()             # Enable Valon
    
    # Initiate logging data from Arduino
    while True:
        data = ser.readline()
        if data.decode('utf-8') == "enlog\r\n":
            print("Connection has established")
            break
        else:
            ser.write("request".encode())

    # Enable interrupt to wait for digital signal from Arduino
    ard_sync = timeSync(ardig0, 1) # Pin ardig0, number of interrupts to receive: 2 (Arduino Timer (1) and Diode Detector Rising Edge (10))
    ard_sync.wait_for_callback()
    print(ard_sync.time_bck[0])
    
    # Enable interrupt to wait for digital signal from button
    off_btn = ButtonOff(rpi_off)
    
    # Change status of LED
    GPIO.output(Led8R, GPIO.LOW)
    GPIO.output(Led8G, GPIO.HIGH)

    # Initialize object loopController
    controller = LoopController()

    # Register the signal handler
    signal.signal(signal.SIGINT, controller.signal_handler)

    with open(filename, 'a') as f:
        # Write header first
        f.write(f"# DateTimeRPI,ArduinoCounter,AttOutputVal,DiodeSignal.\r\n")
        f.write(f"# Valon output: {vfreq} MHz, Chopper: {vamf} Hz.\r\n")
        f.write(f"# Arduino time sync: {ard_sync.time_bck[0]} (counter = 1). \r\n")
        
        # Get power measured by user input
        power_measured = input("Introduce Power measured in Spectrum Analyzer [dBm]: ")
        f.write(f"# Spectrum analyzer measured power: {power_measured} dBm.\r\n")
        
        # Get Tx-Rx distance by user input
        fpl_distance = input("Introduce Distance between Rx and Tx [cm]: ")
        f.write(f"# Distance between transmissor and receptor: {fpl_distance} cm.\r\n")
        
        # Reset serial input to ease data logging
        ser.reset_input_buffer()
        
        ctr = 0
        time_init = time.time()
        time_start = time.time()
        while controller.running:
            # Get RaspberryPi Time
            timedata = str(datetime.datetime.now()).replace(" ", ":").replace("-",":")
            
            # Get string from Arduino
            stringArduino = ser.readline().decode('utf-8')
            ArduinoCounter, ArduinoMicros, AttOutputVal, DiodeSignal, PPStimer, Dronetimer = stringArduino.split(',')
            
            # Write string in txt file
            f.write(f"{timedata},{ArduinoCounter},{AttOutputVal},{DiodeSignal}.\r\n")
            
            # Print string in screen
            if time.time() - time_init >= 1.0 and ctr <= 5:
                print(f"Ct: {ArduinoCounter}. Attenuation Voltage: {AttOutputVal}, ADC reading: {DiodeSignal}.")
                time_init = time.time()
                ctr += 1
            
            # If button has been pressed twice, stop loop.
            if off_btn.off_system() or ((args.stop_time != 0) and (time.time() - time_start >= args.stop_time)):
                print("\nExiting loop...")
                controller.running = False
    
        # Finish and close logfile
        f.close()
        print(f"Logs in{filename} have been saved!")
    
    # Clean and close serial port
    ser.reset_input_buffer()
    ser.close()
    
    # Change status of LED
    GPIO.output(Led8R, GPIO.HIGH)
    GPIO.output(Led8G, GPIO.LOW)

    # Turn off Valon chopper and signal power
    valon.set_amd(0)
    valon.set_amf(1)
    valon.wave_off()

    # Turn off interruots
    off_btn.end_interrupt()
    try:
        ard_sync.end_interrupt()
    except:
        pass
    
    # Reset GPIO pins
    GPIO.cleanup()
    time.sleep(0.5)



if __name__ == '__main__':
    # Initialize parser
    parser = argparse.ArgumentParser(description='Through UART (ttyAMA3) it connects to Serial1 in the Arduino GIGA to receive and save attenuation voltage and ADC readings data in a txt file.')
    parser.add_argument('-o', '--output', type=str, metavar='file', default=None, help='Name of the file that will contain the received data (Optional).')
    parser.add_argument('-fq', '--frequency', type=int, default=12500, help='Value of input frequency to Valon (in Mhz).')
    parser.add_argument('-pv', '--powervoltage', type=float, default=2.0, help='Value of voltage for attenuation in the RF Multiplier (Range: 0-3.3V).')
    parser.add_argument('-st', '--stop_time', type=float, default=0.0, help='Enables stopping time after a certain period of time defined by this variable.')
    parser.add_argument('-ch', '--chopper', action='store_true', default=False, help='Enable RF chopper @ 37 Hz.')

    # Load argparse arguments
    args = parser.parse_args()
    
    # Main
    main()