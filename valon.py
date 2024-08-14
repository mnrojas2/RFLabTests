import v5019
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Valon 5019 control class test")
    parser.add_argument('-ini', '--init_file', type=str, help='Name of the configfile with saved parameters to load.')
    parser.add_argument('-frq', '--frequency', type=float, metavar='f', help='Sets the frequency of the signal (MHz).')
    parser.add_argument('-pwr', '--power', type=float, metavar='pwr', help='Sets the power level of the signal (dBm).')
    parser.add_argument('-amd', '--am_modulation', metavar='amd', type=float, help='Sets the modulation level of the chopper (dBm).')
    parser.add_argument('-amf', '--am_frequency', metavar='amf', type=int, help='Sets the frequency of the chopper (Hz) with a resolution of 1 Hz.')
    parser.add_argument('-off', '--off', action='store_true', default=False, help='Disables the RF output buffer amplifiers (turns off power value in Valon).')
    parser.add_argument('-chp', '--chopper', action='store_true', default=False, help='Enables the chopper control with the Valon.')

    args = parser.parse_args()
    
    # init_file = '/home/drone/drone_smartina/configInit.txt'
    # init_file = 'configFile.txt'
        
    valon = v5019.V5019()
    if args.init_file:
        valon.load_from_file(args.init_file)
        
    # This line actually calls the list ports function to see which ports are there
    valon.list_available_ports()

    # Open serial port
    valon.open_serial_port(port_name="COM11")

    if args.off:
        # Turn off chopper
        valon.set_amd(0)
        valon.set_amf(1)
        
        # Turn off signal power
        valon.wave_off()
        
    else:
        # Sends the frequency command [Mhz]
        valon.set_frequency(args.frequency)

        # Sends the power command [dBm]
        valon.set_power(args.power)

        if args.chopper:
            # Sends the AMsignal amplitude command [dB]
            valon.set_amd(args.am_modulation)

            # Sends the AM frequency command [Hz]
            valon.set_amf(args.am_frequency)

        # Sends the ID command
        valon.wave_on()

    # Close serial port
    valon.close_serial_port()