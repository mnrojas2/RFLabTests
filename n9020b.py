#!/usr/bin/env python

"""
n9020b.py - Library to control the N9020B MXA Signal Analyzer - compatible with Python3
*REQUIRES telnet (sudo apt-get telnet)*
"""

import pexpect
import numpy as np
import time
DEVICE_IP = '192.168.0.5'
DEVICE_PORT = 5023

TERMINATOR = '\r'
PROMPT = 'SCPI> '

WAIT_TIMEOUT = 5
MAX_ATTEMPTS = 20

class N9020B(object):
	
	def __init__(self, ip=DEVICE_IP, port=DEVICE_PORT):
		
		self.ip = ip
		self.port = port
		
		self.terminator = TERMINATOR
		self.prompt = PROMPT
		self.wait_timeout = WAIT_TIMEOUT
		self.max_attempts = MAX_ATTEMPTS
		
		self._connection_command = f'telnet {self.ip} {self.port}'
		
		self.connection = pexpect.spawn(self._connection_command)
		self.freq_units = 'MHz'
		self.verbose = False
		self.connect()
		print('connected')
	
	def connect(self):
		self.connection = pexpect.spawn(self._connection_command)
		self.get_prompt()
		
	def hz_to_freq_units(self, freq):
		freq = float(freq)
		if self.freq_units == 'MHz':
			freq = freq / 10.**6
		elif self.freq_units == 'GHz':
			freq = freq / 10.**9
		elif self.freq_units == 'Hz':
			pass #default returned by analyzer is Hz
		else:
			print('Warning, unknown frequency unit. Value will be left in Hz')
		return freq
	
	def get_prompt(self, depth=0):
		if depth >= self.max_attempts:
			raise pexpect.exceptions.TIMEOUT('Failed to get system prompt')
		
		try:
			self.connection.expect(self.prompt, timeout = self.wait_timeout)
		except pexpect.exceptions.TIMEOUT:
			self.get_prompt(depth=depth+1)
		
		return depth
			
	def command(self, cmd):
		this_command = cmd+self.terminator
		if self.verbose:
			print(f'Sending: {cmd}')
		self.connection.send(cmd+self.terminator)
		
		self.get_prompt()
		data = self.connection.before.decode()
		data = data.split(this_command)[-1].strip()
		return data
	
	def rlist_to_vectors(self, rlist):
		# fparray = np.array(map(float,rlist.strip().split(','))) # py2
		fparray = np.array([float(x) for x in rlist.strip().split(',')])
		return fparray[0:len(fparray):2], fparray[1:len(fparray):2]

	def get_trace(self):
		data = self.command('FETC:SAN?')
		return self.rlist_to_vectors(data)
	
	def get_correction(self, cnum=1):
		data = self.command(f':CORR:CSET{cnum}:DATA?')
		return self.rlist_to_vectors(data)
		
	def wait_for_measure(self):
		data = self.command('*OPC?') #This should block and only return '1' when the operation is complete. 
		if data == '1':
			return True
		else:
			#raise ValueError('Apparently operation did not complete? Unclear what this condition means.')
			return False
	
	def update_spec(self, block=True):
		self.command(':INIT:REST')
		#self.command(':INIT:SAN')
		if block:
			#if not self.wait_for_measure():
			#	self.update_spec()
			self.command('*WAI')
	
	def trigger(self, block=True):
		self.command('*TRG')
		if block:
			self.command('*WAI')
	
	def mark_to_max(self, marknum=1):
		self.command(f':CALC:MARK{marknum}:MAX') 
  
	def mark_to_center(self,position):
		self.command(':CALC:MARK1:X:CENTER'+' '+str(position)+' MHZ')

	def set_ref_level(self,ref):
		self.command(':DISPLAY:WINDOW:TRACE:Y:RLEVEL'+' '+str(ref)+' DBM')
		
	def get_marker_y(self, marknum=1):
		return float(self.command(f':CALC:MARK{marknum}:Y?'))
	
	def get_marker_x(self, marknum=1):
		return self.hz_to_freq_units(float(self.command(f':CALC:MARK{marknum}:X?')))

	def get_marker_pos(self, marknum=1):
		x = self.hz_to_freq_units(float(self.command(f':CALC:MARK{marknum}:X?')))
		y = float(self.command(f':CALC:MARK{marknum}:Y?'))
		return x, y 

	def set_noise_figure(self):
		self.command(':SENS:INST:SEL NFIGURE')
		self.command(':SENS:INST:NSEL 219')
  
	def set_noise_source_on(self):
		self.command(':SOURCE:NOISE ON')

	def set_noise_source_off(self):
		self.command(':SOURCE:NOISE OFF')
		
	def set_ac_coupling(self):
		self.command(':INPUT:COUPLING AC')
		print("ac_coupling")
		
	def configure_noise_figure(self):
		self.command(':CONF:NFIGURE')

	def set_center_frequency(self, freq):
		self.command(f':SENS:FREQ:CENT {freq} {self.freq_units}')
	
	def set_start_frequency(self, freq):
		self.command(f':SENS:FREQ:STAR {freq} {self.freq_units}')
		
	def set_stop_frequency(self, freq):
		self.command(f':SENS:FREQ:STOP {freq} {self.freq_units}')
		
	def get_center_frequency(self):
		return self.hz_to_freq_units(self.command(':SENS:FREQ:CENT?'))
	
	def set_span(self, span):
		self.command(f':SENS:FREQ:SPAN {span} {self.freq_units}')

	def get_span(self):
		return self.hz_to_freq_units(self.command(':SENS:FREQ:SPAN?'))
		
	def set_display_off(self):
		self.command('DISP:ENAB OFF')
		
	def set_display_on(self):
		self.command('DISP:ENAB ON')
		
	def set_rbw(self, bw):
		self.command(f'SENS:BAND:RES {bw} {self.freq_units}')

	def set_rbw_auto(self):
		self.command('SENS:BAND:RES:AUTO ON')
		
	def set_rbw_filt_gaussian(self):
		self.command('SENS:BAND:SHAP GAUS')
		
	def set_rbw_filt_flat(self):
		self.command('SENS:BAND:SHAP FLAT')

	def seek_max_and_center(self, marker_off=False, return_amp=False):
		'''Uses a marker to find the maximum of the current span and sets the center frequency to the maximum'''
		self.mark_to_max(1)
		maxx, maxy = self.get_marker_pos()
		self.set_center_frequency(maxx)
		if marker_off:
			self.all_markers_off()
		if not return_amp:
			return maxx
		else:
			return maxx, maxy
			
	def all_markers_off(self):
		self.command(':CALC:MARK:AOFF')
	
	def set_sweep_time(self, time):
		'''Set the sweep time in seconds. May result in MEAS UNCAL for swept analyzer measurements but is useful for zero span measurements'''
		self.command(f'SENS:SWE:TIME {time}')
	
	def get_sweep_time(self):
		return float(self.command('SENS:SWE:TIME?'))
	
	def set_sweep_time_auto(self):
		self.command('SENS:SWE:TIME:AUTO ON')

	def set_sweep_points(self, points):
		self.command(f'SENS:SWE:POIN {points}')
	
	def get_sweep_points(self):
		return self.command('SENS:SWE:POIN?')
		
	def set_continuous_off(self):
		self.command(':INIT:CONT OFF')

	def set_continuous_on(self):
		self.command(':INIT:CONT ON')
		
	def get_continuous(self):
		val = self.command(':INIT:CONT?')
		if val == '1':
			return True
		elif val == '0':
			return False
	
	def set_trace_type(self, ttype='writ', trace_num=1):
		'''See page 833'''
		ttype_dict = {}
		ttype_dict['WRIT'] = ['writ', 'write', 'clear', 'clearwrite']
		ttype_dict['AVER'] = ['aver', 'average']
		ttype_dict['MAXH'] = ['pos', 'peak', 'max', 'maxhold']
		ttype_dict['MINH'] = ['min', 'minimum', 'minhold']
		
		for t in ttype_dict:
			if ttype.lower() in ttype_dict[t]:
				ttype = t
				break
	
		self.command(f'TRAC{trace_num}:TYPE {ttype.upper()}')
		
	def get_trace_type(self, trace_num=1):
		'''See page 833'''
		return self.command(f'TRAC{trace_num}:TYPE?')

	def set_detector_type(self, ttype='norm', trace_num=1):
		'''See page 840'''
		ttype_dict = {}
		ttype_dict['NORM'] = ['norm', 'normal']
		ttype_dict['AVER'] = ['aver', 'average']
		ttype_dict['POS'] = ['pos', 'peak', 'max']
		
		for t in ttype_dict:
			if ttype.lower() in ttype_dict[t]:
				ttype = t
				break
	
		self.command(f'DET:TRAC{trace_num} {ttype.upper()}')
	
	def get_detector_type(self, trace_num=1):
		'''See page 840'''
		return self.command(f'DET:TRAC{trace_num}?')
		
	def set_average_hold_count(self, num):
		self.command(f':AVER:COUN {int(num)}')
		
	def get_average_hold_count(self):
		return int(self.command(':AVER:COUN?'))
	
	def get_sig_id(self):
		return self.command(':SID?')
		
	def set_sig_id(self, val):
		if val:
			self.command(':SID ON')
		else:
			self.command(':SID OFF')	

	def _split_table(self, table_data):
		data = table_data.split(',')
		freqs = data[::2]
		powers = data[1::2]
		peaks = []
		for f,p in zip(freqs,powers):
			peaks.append((float(f),float(p)))
		return peaks
		
	def delete_user_view(self):
		self.command('DISP:VIEW:ADV:DEL')
	
	def set_marker_table(self, val=True):
		if val:
			self.command('CALC:MARK:TABL ON') #turn marker table on
		else:
			self.command('CALC:MARK:TABL OFF')
			
	def get_marker_table_status(self):
		return int(self.command('CALC:MARK:TABL?'))
	
	def activate_marker_table(self):
		if not self.get_marker_table_status():
			self.set_marker_table(True)
			if not self.get_marker_table_status():
				self.delete_user_view()
				self.set_marker_table(True)
				if not self.get_marker_table_status():
					raise ValueError('Unable to activate marker table')
		
	def get_marker_table(self, activate=False):
		'''This only works in spectrum analyzer mode'''
		
		if activate:
			self.activate_marker_table()
		return self.command('FETC:SAN8?')
	
	def set_peak_table(self, val=True):
		if val:
			self.command('CALC:MARK:PEAK:TABL:STAT ON') #turn peak table on
		else:
			self.command('CALC:MARK:PEAK:TABL:STAT OFF')
		
	def get_peak_table_status(self):
		return int(self.command('CALC:MARK:PEAK:TABL:STAT?'))
	
	def activate_peak_table(self):
		if not self.get_peak_table_status():
			self.set_peak_table(True)
			if not self.get_peak_table_status():
				self.delete_user_view()
				self.set_marker_table(False)
				self.set_peak_table(True)
				if not self.get_peak_table_status():
					raise ValueError('Unable to activate peak table')

	def get_peak_table(self, activate=False):
		if activate:
			self.activate_peak_table()
			
		return self._split_table(self.command('TRAC:MATH:PEAK?'))
		
	def set_continuous_peak_search(self, val=True):
		'''See page 653'''
		if val:
			self.command('CALC:MARK:CPS ON')
		else:
			self.command('CALC:MARK:CPS OFF')
	
	def get_continuous_peak_search(self):
		'''See page 653'''
		return int(self.command('CALC:MARK:CPS?'))
		

def get_device(device=None):
	if device is None:
		device = N9020B()
	return device

def set_for_max_chop(device=None, update=True, num_scans=5):	
	device = get_device(device)
	device.set_trace_type('max')
	device.set_average_hold_count(num_scans)
	
	if update:
		device.update_spec()

def set_for_min_chop(device=None, update=True, num_scans=3):
	device = get_device(device)
	device.set_trace_type('min')
	device.set_average_hold_count(num_scans)
	
	if update:
		device.update_spec()

def find_gunn(device=None, show=False, approx_freq = 145000, max_span=5000):
	device = get_device(device)
	device.set_center_frequency(approx_freq)
	device.set_span(max_span)
	device.update_spec()
	maxx = device.seek_max_and_center()
	if show:
		print('Set center frequency to maximum at: ', maxx, device.freq_units)
	return maxx

def center_gunn(device=None, span=100., show=False, chopper=True):
	'''Centers assuming no knowledge of peak location'''
	device = get_device(device)
	device.set_rbw_auto()
	device.set_sweep_points(1001)
	
	if chopper:
		set_for_max_chop(device)
	#find_gunn(device) 
	if span < 100:
		recenter_gunn(device, span=100.) #recenter first at span=100MHz
	maxx = recenter_gunn(device, span=span)

	if show:
		print('Set center frequency to maximum at: ', maxx, device.freq_units)
	if chopper:
		device.set_trace_type('clear')
	return maxx
	
def recenter_gunn(device=None, span=100., return_amp=False, chopper=True):
	'''Assumes Gunn peak is within specified span from of center. If this is not the case use center_gunn
	With return_amp True, maxx will be a tuple: (frequency, amplitude)'''

	device=get_device(device)
	
	if chopper:
		set_for_max_chop(device=device, update=False)

	device.set_span(span)
	device.set_rbw_auto()
	device.update_spec() #This blocks until the measurement is done.
	maxx = device.seek_max_and_center(return_amp=return_amp)

	if chopper:
		device.set_trace_type('clear')

	return maxx

def recenter_with_time(dev=None, span=100., return_amp=True, chopper=True):
	dev = get_device(dev)
	if chopper:
		set_for_max_chop(device=dev, update=False)

	t0 = time.time()
	maxx = recenter_gunn(dev, span=span, return_amp=return_amp)
	t1 = time.time()
	tavg = (t0+t1) / 2.

	if chopper:
		dev.set_trace_type('clear')

	return tavg, maxx	

def freq_slope_measurement(dev=None, span=100., chopper=True, dt=5):
	dev = get_device(dev)
	center_gunn(device=dev, span=span, chopper=chopper)
		
	t0, f0 = recenter_with_time(dev=dev, span=span, return_amp = False, chopper=chopper)
	time.sleep(dt)
	t1, f1 = recenter_with_time(dev=dev, span=span, return_amp = False, chopper=chopper)

	return (f1-f0) / (t1-t0)

def recenter_for_amp_meas(dev=None):
	dev = get_device(dev)
	dev.set_sweep_points(1001)
	dev.set_rbw_auto()
	dev.set_sweep_time(.200)
	dev.update_spec()
	dev.seek_max_and_center()

def set_normal(dev=None):
	dev = get_device(dev)
	dev.set_sweep_points(1001)
	dev.set_rbw_auto()
	dev.set_sweep_time_auto()
	dev.set_rbw_filt_gaussian()
	
def measure_with_time(dev=None, return_amp=True):
	t0 = time.time()
	dev.update_spec() #This blocks until the measurement is done.
	dev.mark_to_max()
	maxes = dev.get_marker_pos()
	t1 = time.time()
	tavg = (t0+t1) / 2.
	return tavg, maxes	

def set_full_range(device=None):
	device = get_device(device)
	device.set_span(50000)
	device.set_center_frequency(145000)
	device.update_spec()
	
def set_zero_span(device=None, rbw=8., time=3., num_samples=1001):
	device = get_device(device)
	
	device.freq_units = 'MHz'
	device.set_sweep_points(num_samples)
	device.set_span(0)
	device.set_rbw(rbw)
	device.set_rbw_filt_flat()
	device.set_sweep_time(time)
	
def setup_gunn_power_meas(device=None):
	'''Peak must already be within 10 MHz of CF'''
	device = get_device(device)
	device.freq_units = 'MHz'
	device.set_rbw_filt_flat()
	device.set_span(20)
	recenter_for_amp_meas(device)
	#recenter_gunn(device, span=20.)
	device.set_rbw(3.)
	device.set_sweep_points(30)
	device.set_sweep_time(.200)
	
def take_zerospan_meas(device=None, time=1., num_samples=1001, approx_freq=145000, max_span=5000):
	device = get_device(device)
	#center gunn
	device.set_continuous_off()
	device.set_detector_type('POS')
	set_full_range(device)
	find_gunn(device, approx_freq=approx_freq, max_span=max_span)
	center_gunn(device)
	
	set_zero_span(device, time=time, num_samples=num_samples)
	device.set_sig_id(False)
	device.update_spec()
	trace = device.get_trace()
	device.set_sig_id(True)
	set_full_range(device)
	find_gunn(device, approx_freq=approx_freq)
	center_gunn(device)
	device.set_continuous_on()
	return trace

def fast_peak_meas(device=None, center_freq = 145000, span = 5000, time_to_measure = 0, saveFile = None):	
	device = get_device(device)		
	device.set_center_frequency(center_freq)
	device.set_span(span)
	device.set_rbw(8.)
	device.set_sweep_time_auto()
	device.set_sweep_points(1001)
	
	tinit = time.time()
	t0 = tinit
	freqs = []
	amps = []
	times = []
	running = True
	device.set_continuous_peak_search(True)
	device.activate_marker_table()

	try:
		while running:
			try:
				# f, a = map(float,device.get_marker_table().split(',')[2:4]) # py2
				marker_table = device.get_marker_table()
				f, a = [float(x) for x in marker_table.split(',')[2:4]]
			except ValueError:
				raise ValueError('Bad table format. Is marker table enabled?')
			t1 = time.time()
			freqs.append(f)
			amps.append(a)
			times.append(t1)
			if saveFile is not None:
				saveFile.write(f'{t1} {f} {a}\n')
			if time_to_measure and t1 - tinit > time_to_measure:
				running = False
			t0 = t1
	except:
		pass

	device.set_continuous_peak_search(False)
	device.set_marker_table(False)
	return times, freqs, amps
		
def dbm_to_mw(dbm):
	dbm = np.array(dbm)
	return 10**(dbm/10.)
	
def tod_rms(data):
	return np.sqrt(np.mean(dbm_to_mw(data)**2))
	
def tod_first_harmonic_power(t, data, bw=2):
	'''Sum the power in "bw" Hz around the highest peak in the spectrum'''
	data = dbm_to_mw(data)
	fftdata = np.fft.fft(data)
	fftfreqs = np.fft.fftfreq(len(t), np.median(np.diff(t)))
	
	maxfbin = np.argmax(fftdata)
	centerfreq = fftfreqs[maxfbin]
	fftmask = np.abs(fftfreqs-centerfreq) < bw/2.
	return np.sum(np.abs(fftdata[fftmask])) / np.sum(fftmask)

def tod_harmonic_power(t, data, highpass=2):
	'''Sum the power in "bw" Hz around the highest peak in the spectrum'''
	data = dbm_to_mw(data)
	fftdata = np.fft.fft(data)
	fftfreqs = np.fft.fftfreq(len(t), np.median(np.diff(t)))
	fftmask = fftfreqs>highpass
	return np.sum(np.abs(fftdata[fftmask])) / np.sum(fftmask)

def get_tod_amplitudes(t,data):
	rms =  tod_rms(data)
	firstharm = tod_first_harmonic_power(t,data)
	allharm = tod_harmonic_power(t,data)
	return rms, firstharm, allharm

def get_amps(device=None, time = 5, NQF = 300):
	device = get_device(device)
	samples = 2*NQF*time
	t, data = take_zerospan_meas(device, time, samples)
	return get_tod_amplitudes(t,data)
