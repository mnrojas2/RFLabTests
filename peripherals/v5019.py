#!/usr/bin/env python

'''
v5019.py - Library to control the Valon Frequency Synthesizer 5019 - compatible with Python3
Author: mnrojas2 (based in the code made by J.C Fluxa)
'''

from serial.tools import list_ports
import serial
import time


class V5019():
    
    def __init__(self, freq=12500, pwr=6, amd=30, amf=37):
        self.header = "640F"
        self.freq = freq
        self.pwr = pwr
        self.ser = None
        self.amd = amd
        self.amf = amf
        
    def load_from_file(self, filename):
        # Reads the contents of the text file.
        try:
            with open(filename) as file:
                for line in file:
                    if "=" in line and not line.startswith("#") and 'valon' in line:
                        name, value = line.replace('\n','').split("=")
                        if 'freq' in name:
                            self.freq = float(value)
                        elif 'pwr' in name:
                            self.pwr = float(value)
                        elif 'amd' in name:
                            self.amd = float(value)
                        elif 'amf' in name:
                            self.amf = int(value)
            
                print(f"CURRENT VALUES: freq: {self.freq}, pwr: {self.pwr}, amd: {self.amd}, amf: {self.amf}")
            file.close()
        except FileNotFoundError:
            print("THE FILE WAS NOT FOUND.")
            
    def load_manual(self, new_freq, new_pwr, new_amd, new_amf):
        # Updates the values from direct inputs.
        self.freq = float(new_freq)
        self.pwr = float(new_pwr)
        self.amd = float(new_amd)
        self.amf = int(new_amf)
            
        print(f"CURRENT VALUES: freq: {self.freq}, pwr: {self.pwr}, amd: {self.amd}, amf: {self.amf}")
    
    def list_available_ports(self):
        # Lists all the ports when called.
        ports = list_ports.comports()
        for port in ports:
            print(port)

    def open_serial_port(self, port_name='/dev/ttyAMA5', baud_rate=115200, timeout=3.0):
        # This function will open the serial port specified.  
        # The port_name parameter needs to be set to whatever port the synthesizer is connected at which the list_available_ports function will tell you.
        # For USB connector: port_name="/dev/cu.usbserial-12203142", baud_rate=9600
        # For USER PORT connector: port_name="/dev/ttyAMA5", baud_rate=115200
        try:
            self.ser = serial.Serial(port_name, baud_rate, timeout=timeout)
            self.ser.setDTR(False)
            self.ser.flushInput()
            self.ser.setDTR(True)
            print(f"Serial port {port_name} opened successfully.")
            time.sleep(1.0)
        except serial.SerialException as e:
            print(f"Error opening serial port: {e}")

    def close_serial_port(self):
        # Closes serial port.
        if self.ser and self.ser.is_open:
            self.ser.close()
            
    def get_help(self):
        # Gets a list of commands available for V5019
        self.send_command("H\r")
            
    def get_device_info(self):
        # Returns valon data, such as manufacturer's name, model, serial number, firmware version and more.
        self.send_command("STAT\r")
            
    def save_state(self):
        # Saves the current synthesizer state to flash memory.
        self.send_command("SAV\r")
        
    def recall_state(self):
        # Recalls the synthesizer state from flash memory.
        self.send_command("RCL\r")
        
    def clean_saved_state(self):
        # Clean all user data in flash memory. Unlike RST, it does not remove user synthesizer names.
        self.send_command("CLE\r")
        
    def reset_saved_state(self):
        # Resets both synthesizers to default factory settings. Does not change the Reference Trim value.
        self.send_command("RST\r")
    
    def wave_off(self):
        # Turns off the Valon.
        self.send_command("OEN OFF\r")
        print("****** VALON OFF ******")
        
    def wave_on(self):
        # Turns on the Valon (updates power level set previously).
        self.send_command("OEN ON\r")
    
    def set_frequency(self, new_freq=None):
        # Sets wave frequency.
        if new_freq != None:
            self.freq = new_freq
        self.send_command("MODE CW; f "+str(self.freq)+" MHz\r") # In Mhz
    
    def set_power(self, new_pwr=None):
        # Sets wave power.
        if new_pwr != None:
            self.pwr = new_pwr
        self.send_command("PWR "+str(self.pwr)+"\r") # In dBm

    def set_amd(self, new_amd=None): # AM modulation
        # Sets wave modulation low level.
        if new_amd != None:
            self.amd = new_amd
        self.send_command("AMD "+str(self.amd)+"\r") # In dB
    
    def set_amf(self, new_amf=None): # AM frequency
        # Sets wave modulation frequency.
        if new_amf != None:
            self.amf = new_amf
        self.send_command("AMF "+str(self.amf)+"\r") # In Hz (Valon takes an integer, floats gets truncated).

    def send_command(self, command, delay=0.1):
        # This function sends a command via the serial port specified as an argument in the function call.
        try:
            # Clear the input buffer
            self.ser.reset_input_buffer()

            # Send the command with a carriage return
            command_with_cr = f"{command}\r"
            self.ser.write(command_with_cr.encode())

            # Introduce a delay before reading the response
            time.sleep(delay)

            # Read the response with a timeout
            response_bytes = self.ser.read(1024)  # Adjust the buffer size as needed

            # Decode and print the response
            response = response_bytes.decode().strip()
            print(f"Response from device: {response}")
            
        except serial.SerialException as e:
            print(f"Error communicating with the device: {e}")
