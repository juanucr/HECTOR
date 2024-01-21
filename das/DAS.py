#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 17:21:23 2022

@author: juan
"""

import numpy as np
import matplotlib.pyplot as plt

def readerh5(file):
    import h5py
    data = h5py.File(file,'r')
    ids = data['receiver_ids_ELASTIC_point']
    rec = ids[:]
    if 'strain' in data['point'].keys():
        ezz = data['point']['strain'][:,2,:]
        str_ezz = np.zeros_like(ezz)
        for i in range(len(ezz)):
            tmp = np.where(rec == i)
            str_ezz[i,:] = ezz[tmp,:]
            strain = str_ezz.T 
    return strain.T

#%%
class DAS:
    
    def __init__(self, file, dx, gl, fname, file_format):
       
        print('Reading : ' + file)
        
        if file_format =='tdms':
            from nptdms import TdmsFile
            tdms_file = TdmsFile(file)
            self.sampling_rate = tdms_file.properties['SamplingFrequency[Hz]']
            self.starttime = tdms_file.properties['CPUTimeStamp']
            self.gl=tdms_file.properties['GaugeLength']
            traces = (tdms_file.as_dataframe().to_numpy()).T
            traces=traces

            self.ntrs, self.npts = traces.shape
            self.dt=1./self.sampling_rate
            self.fname = fname
            self.sampling_rate = tdms_file.properties['SamplingFrequency[Hz]']
            del tdms_file

        elif file_format =='segy':
            from obspy.io.segy.core import  _read_segy
            das_data = _read_segy(file,format='segy',unpack_trace_headers=True)
            traces = np.stack([trace.data for trace in das_data])
                                                
            self.ntrs, self.npts = traces.shape
            self.dt = das_data[0].stats.delta
            self.fname = fname
            self.starttime = das_data[0].stats.starttime
            self.sampling_rate = das_data[0].stats.sampling_rate
            del das_data
                        
        elif file_format =='h5':
            from obspy import UTCDateTime
            traces = readerh5(file)
            self.ntrs, self.npts = traces.shape
            
            print('Insert the time duration of your h5 file (in seconds): ')
            duration = float(input())
            self.dt = duration / self.npts
            self.sampling_rate = 1/self.dt
            self.fname = fname
            self.starttime = UTCDateTime.now()
        
        elif file_format =='npy':
            from obspy import UTCDateTime
            traces = np.load(file)
            self.ntrs, self.npts = traces.shape
            print('Insert the time duration of your numpy file (in seconds): ')
            duration = float(input())
            self.dt = duration / self.npts
            self.sampling_rate = 1/self.dt
            self.tax = np.arange(0, traces.shape[0], self.dt)
            
            self.fname = fname
            self.traces = traces
            self.starttime = UTCDateTime.now()
                        
        else:                
            print("Only tdms, sgy, npy and h5 file formats are supported")
                    
        self.traces = traces
        self.dx = dx
        self.gl = gl
        self.format = file_format

        
    def __downsample(self, data, sampling_rate):
        from scipy.signal import resample
        
        sampling_rate = int(sampling_rate)        
        new = int(sampling_rate*self.npts/self.sampling_rate)
        data2 = resample(data, new, window='hann', axis= 1) 
        self.traces = data2
        self.ntrs, self.npts = data2.shape
        self.sampling_rate = sampling_rate
        self.dt = 1/self.sampling_rate
        return data2
    
  
    def __filter(self, data, ftype, fmin, fmax, order=4):
        from scipy.signal import butter, sosfilt
               
        if ftype =='bandpass':
            sos = butter(order, [fmin,fmax], 'bandpass', fs=self.sampling_rate, output='sos')
        
        elif ftype =='highpass':
            sos = butter(order, fmin, 'highpass', fs=self.sampling_rate, output='sos')
                
        elif ftype =='lowpass':
            sos = butter(order, fmax, 'lowpass', fs=self.sampling_rate, output='sos')
        
        data2 = sosfilt(sos, data, axis=1)
        
        self.traces = data2
        self.ntrs, self.npts = data2.shape
        return data2
        

    def denoise(self, data, ftype=None, fmin=None, fmax=None, sampling_rate_new=None, k0=False, low_vel_events=False, order=4):
        
        if sampling_rate_new is not None:
            
            traces = self.__filter(data, 'lowpass', fmin, fmax) # lowpass before downsampling
            traces = self.__downsample(traces, sampling_rate_new) # downsampling of the data
        
        else:
            
            traces = data
        
        if ftype is not None:
            
            traces = traces - traces.mean() # detrend the data
            traces = self.__filter(traces, ftype, fmin, fmax) # Bandpass filtering the data
        
        else:
            
            traces = data
        
        traces = self.__trace_normalization(traces) # normalize the data
        
        if k0 or low_vel_events:
            traces = self.__fk_filt(traces, k0, low_vel_events) #FK filtering the data
        
        self.traces = traces
        self.ntrs,self.npts = traces.shape



        
    def data_select(self, starttime=None, endtime=None, startlength=None, endlength=None):
        
        if starttime is not None:
            i_starttime = int(starttime/self.dt)
        else:
            i_starttime = int(0)
        
        if endtime is None or endtime == -1:
            i_endtime = int(self.traces.shape[-1] - self.traces.shape[-1]%self.dt)
        else:
            i_endtime = int(endtime/self.dt)
        
        
        if startlength is None:
            i_startlength = int(0)
        else:
            i_startlength = int(startlength/self.dx)
        
        if endlength is None or endlength == -1:
            i_endlength = int(self.traces.shape[0])
        else:
            i_endlength = int(endlength/self.dx)
        
        traces = self.traces[i_startlength:i_endlength,i_starttime:i_endtime]
        self.traces = traces
        self.ntrs, self.npts = traces.shape
        
     
    def __trace_normalization(self, data):
        from scipy.signal  import detrend
        data = detrend(data, type='constant')
        nf = np.abs(data).max(axis=1)
        data = data / nf[:, np.newaxis]
        return data
    
        
    def visualization(self, data, filename, path_results='./report/', savefig=False):
        time = np.arange(self.npts)*self.dt
        depth = np.arange(self.ntrs)*self.dx
        plt.figure(figsize=[20,5])
        plt.imshow(data,extent=[min(time),max(time),max(depth),min(depth)],cmap='seismic',aspect='auto')
        plt.ylabel('Distance along the fiber [m]')
        plt.xlabel('Relative time [s]')
        plt.tight_layout()
        
        if savefig is not False:
            plt.savefig(path_results + self.fname + '_imshow' + '.eps', dpi=300)
            plt.savefig(path_results + self.fname + '_imshow' + '.png', dpi=300)
        
   
    def __fk_filt(self, data, k0=False, low_vel_events=False): # Adapted to the 2019 FORGE dataset
        from scipy.signal import windows

        fk = np.fft.rfft2(data)
        n, m = fk.shape
        filt = np.ones([n,m])

        ## here define the shape of the triangular window (inner and outer) and scale it by the number of considered traces
        max_value_outer_trian = int(m/2.5)
        outer_window = (windows.triang(n) * max_value_outer_trian)
        
        signal_len = (self.npts/self.sampling_rate)
        delta_filt = int(10*signal_len)
       
        if k0:
            filt[0:3,:] = 0.5
            filt[n-3:,:] = 0.5
        
        if low_vel_events:
            for i in range(filt.shape[0]):
                filt[i,int(outer_window[i])-int(delta_filt):int(outer_window[i])] = 0.5
                filt[i,:int(outer_window[i])-3] = 0.
            
        if k0:
            filt[0:2,:] = 0.
            filt[n-2:,:] = 0.
                    
        fkfilt = np.abs(fk)*filt*np.exp(1j*np.angle(fk))        
        data_filt = np.fft.irfft2(fkfilt)

        self.traces = data_filt
        return data_filt
        
            
    def plotfk(self, path_results='./report/', savefig=False):
        from matplotlib import colors
        f = np.fft.rfftfreq(self.npts, d=self.dt)
        k = np.fft.fftfreq(self.ntrs, d=self.dx)
        fk= np.fft.rfft2(self.traces) + 1
        fk = np.abs(fk) / np.max(np.abs(fk))
        
        plt.figure()
        plt.imshow(np.abs(np.fft.fftshift(fk, axes=(0,))).T,
                   extent=[min(k), max(k), min(f), max(f)],
                   aspect='auto',
                   cmap='plasma',
                   interpolation=None,
                   origin='lower',
                   norm=colors.LogNorm())
        
        h = plt.colorbar()
        h.set_label('Amplitude Spectra  (rel. 1 $(\epsilon/s)^2$)')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Wavenumber [1/m]')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        
        if savefig is not False:
            plt.savefig(path_results + self.fname + '_FK' + '.eps', dpi=300)
            plt.savefig(path_results + self.fname + '_FK' + '.png', dpi=300)

                
    def analytic_signal(self,traces):
        tracef=np.fft.fft(traces)
        nsta,nfreq=np.shape(tracef)
        freqs=np.fft.fftfreq(nfreq,self.dt)
        traceh=tracef+(np.sign(freqs).T*tracef)
        traces=traces+1j*np.fft.ifft(traceh).real
        self.envelope=np.abs(traces)

    def stalta(self,traces):
        tsta = .01
        tlta = tsta*10
        nsta=int(tsta/self.dt)
        nlta=int(tlta/self.dt)
        ks=self.dt/tsta
        kl=self.dt/tlta
        sta0=np.mean(traces[:,nlta:nlta+nsta]**2, axis=1)
        lta0=np.mean(traces[:,0:nlta]**2, axis=1)
        stalta=np.zeros(np.shape(traces))
        for i in range(nlta+nsta,self.npts):
            sta0=ks*traces[:,i]**2+((1.-ks)*sta0)
            lta0=kl*traces[:,i-nsta]**2+((1.-kl)*lta0)
            stalta[:,i]=sta0/lta0
        stalta[:,0:nlta+nsta]=stalta[:,nlta+nsta:2*(nlta+nsta)]
        stalta=self.__trace_normalization(stalta)
        self.traces=stalta
        
        self.ntrs,self.npts = stalta.shape
        return stalta

