from serial.tools import list_ports
import sys
import serial
import time


# This function lists all the ports when called
def list_available_ports():
    # Identify the correct port
    ports = list_ports.comports()
    for port in ports:
        print(port)

# This function will open the serial port specified.  
# The port_name parameter needs to be set to whatever port the synthesizer is connected at which the list_available_ports function will tell you
# def open_serial_port(port_name='/dev/cu.usbserial-12203142', baud_rate=9600, timeout=3.0):
def open_serial_port(port_name='COM11', baud_rate=9600, timeout=3.0):
    global ser
    try:
        ser = serial.Serial(port_name, baud_rate, timeout=timeout)
        ser.setDTR(False)
        ser.flushInput()
        ser.setDTR(True)
        print(f"Serial port {port_name} opened successfully.")
        time.sleep(1.0)
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")

def close_serial_port():
    global ser
    if ser and ser.is_open:
        ser.close()
        
def valon_off():
    send_command("OEN OFF\r")
    
def valon_on():
    send_command("OEN ON\r")

# This function sends a command via the serial port specified as an argument in the function call 
def send_command(command, delay=0.1):
    global ser
    try:
        # Clear the input buffer
        ser.reset_input_buffer()

        # Send the command with a carriage return
        command_with_cr = f"{command}\r"
        ser.write(command_with_cr.encode())

        # Introduce a delay before reading the response
        time.sleep(delay)

        # Read the response with a timeout
        response_bytes = ser.read(1024)  # Adjust the buffer size as needed

        # Decode and print the response
        response = response_bytes.decode().strip()
        print(f"Response from device: {response}")
    except serial.SerialException as e:
        print(f"Error communicating with the device: {e}")


# Declare ser as a global variable
ser = None

# Open configInit.txt file
file = '/home/drone/drone_smartina/configInitArduino.txt'
header = "640F"

# with open(file) as json_file:
#    configInit_data = json.load(json_file)

# # Parse the information and convert it to string
# valon_freq = int(configInit_data['valon_freq'])
# valon_pwr  = float(configInit_data['valon_pwr'])

# Set valon frequency based on terminal commands or by default
if len(sys.argv) > 1:
    valon_freq = sys.argv[1]
else:
    valon_freq = 12500
    
if len(sys.argv) > 2:
    valon_pwr = sys.argv[2]
else:
    valon_pwr = 6 # (-0.5 dBm en vuelos abril 2024)
    
# This line actually calls the list ports function to see which ports are there
list_available_ports()

# Open serial port
open_serial_port()

# Sends the ID command
send_command("MODE CW; f "+str(valon_freq)+" MHz\r")

# Sends the ID command
send_command("PWR "+str(valon_pwr)+"\r")

# Sends the ID command
valon_on()

# Close serial port
input("Press Enter to finish")
close_serial_port()

