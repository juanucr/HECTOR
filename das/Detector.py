#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 17:21:23 2022

@author: juan
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import DAS as DAS
from numba import jit


## COMPILED FUNCTIONS ##

@jit(nopython=True)
def sample_select(ns_window, data, dat_win, y, a, b, c, d_static):
    
    x = np.sqrt( ( 1 + ( (y-d_static) /c ) **2 ) *a**2 ) + b
    
    # selection of samples
    for k in range(data.shape[0]):
        i_hyp = x[k]
        i_hyp = int(i_hyp)

        if i_hyp < ns_window//2:
            dat_win[k,:] = 0.
                        
        elif i_hyp + ns_window//2 < data.shape[1]:
            dat_win[k,:] = data[k,i_hyp-ns_window//2:i_hyp+ns_window//2]

        else:
            dat_win[k,:] = 0.

    return dat_win


@jit(nopython=True)
def semblance_func(arr):
    ntr = arr.shape[0]
    num, den = np.sum(np.square(np.sum(arr, axis=0))), np.sum(arr**2) * ntr
    return 1e-16 if den==0. else num/den


#############################################################################

class Detector(DAS.DAS):
    def __init__(self,file, dx, gl, fname, file_format):
        DAS.DAS.__init__(self,file, dx, gl, fname, file_format)
    
    
    
    def __svd(self, arr): # Tried to compile this function but doesn't get faster
        '''
        SVD decomposition of data covariance matrix 
        
        '''
        if arr.all()==False:
            return 0.
        else:
            u, s, v = np.linalg.svd(np.cov(arr))
            m = s.shape[0] # number of traces
        
            op = np.sum(s[1:]/(m-1))
            s_n = (s[0] - op) / op
        
            return 0. if op==0. else s_n
    
   
    def hyperbolae_tuning(self, a, b, d, c_min, c_max, c_step, path_results='./report/', savefig=False):
        '''
        

        Parameters
        ----------
        a : Int or float
            smaller this number --> the flatter the hyperbola vertex is.
        b : Int or float
            position along the sample axis.
        d : Int or float
            position along the trace axis.
        c_min : Int or float
            Range of curvatures. The smaller the number --> the higher the curvature is.
        c_max : Int or float
            Range of curvatures. The bigger the number --> the smaller the curvature is.
        c_step : Int or float
            Step between cmin and cmax. Can be a float to have more density of hyperbolas.
        path_results : TYPE, optional
            DESCRIPTION. The default is './report/'.
        savefig : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        A group of hyperbolas on your data. This way you can check if with the parameters you cover the seismic event of your interest.

        '''
                
        fontsize = 10
        plt.rcParams['axes.linewidth'] = 0.3
        plt.rc('xtick', labelsize=8) 
        plt.rc('ytick', labelsize=8)
        
        fig, ax = plt.subplots(figsize=(10,5))
        
        ax.imshow(self.traces, cmap='seismic', aspect='auto')
        ax.set_ylabel('Number of traces', fontsize=fontsize)
        ax.set_xlabel('Samples', fontsize=fontsize)
       
        y = np.arange(self.ntrs)
        c_vector = np.arange(c_min, c_max, c_step) # range of curvatures. The smaller the number --> the higher the curvature is.
        b = b-a
        
                
        for c in c_vector:
            x = np.sqrt( ( 1 + ( (y-d) /c ) **2 ) *a**2 ) + b
            ax.plot(x, y, c='gray', alpha=.5)

        fig.tight_layout()
        
        if savefig is not False:
            plt.savefig(path_results + self.fname + '_hyperbola_tuning.eps', dpi=300)
            plt.savefig(path_results + self.fname + '_hyperbola_tuning.png', dpi=300)

            

    def detector(self,
                  ns_window,
                  a,
                  b_step,
                  c_min, c_max, c_step,
                  d_static=None, d_min=None, d_max=None, d_step=None,
                  shift=0, svd=False, lat_search=False):
        
        
        
        int(ns_window)+1 if int(ns_window)%2 != 0 else int(ns_window)
        
        y = np.arange(self.ntrs)
        
        c_vector = np.arange(c_min, c_max+c_step, c_step)
        self.curv = c_vector
        b_vector = np.arange(0, self.npts, b_step) - a
    
        dat_win = np.zeros((self.ntrs, ns_window))
       
        sem = np.zeros((len(c_vector), len(b_vector)))
        sem2 = sem.copy()
        sem_matrix = np.zeros((len(c_vector), len(b_vector)))
        
        ##############----- PERFORM SCANNING -------------#########
        
        if lat_search == False and d_static is not None:
            
            for b in np.ndenumerate(b_vector):
                print(int(np.round(100*(b[1]*self.dt)/(self.npts*self.dt), decimals=0)), '%')
                
                # selection of samples
                for c in np.ndenumerate(c_vector):
                    
                    # selection of samples
                    dat_win = sample_select(ns_window, self.traces, dat_win, y, a, b[1], c[1], d_static)
                    sem[c[0],b[0]] = semblance_func(dat_win)
                    sem2[c[0],b[0]] = self.__svd(dat_win) if svd==True else None
                    
            
            sem2 = sem2/sem2.max() # svd weight normalized between 0 and 1 to scale sem matrix
            sem = sem*sem2 if svd==True else sem
            
        elif lat_search == False and d_static is None:
            raise Exception('Must enter a value for d_static')
        
        
        elif lat_search == True:
            
            if d_min == None and d_max == None and d_step == None:
                d_min, d_max, d_step = 0, int(self.ntrs * self.dx), int(self.ntrs/10)
            
            d_vector = np.arange(d_min, d_max+d_step, d_step)
            
            for d in np.ndenumerate(d_vector):
                print(int(np.round(100*d[1]/(d_vector.max()), decimals=1)), '%')
                
                for b in np.ndenumerate(b_vector):
                    
                    for c in np.ndenumerate(c_vector):
                        
                        dat_win = sample_select(ns_window, self.traces, dat_win, y, a, b[1], c[1], d[1])
                        sem[c[0],b[0]] = semblance_func(dat_win)
                        sem2[c[0],b[0]] = self.__svd(dat_win) if svd==True else None
                        
                        
                sem2 = sem2/sem2.max() # normalized between 0 and 1 to scale sem matrix

            
                if svd==False:
                    sem_matrix = np.dstack((sem_matrix, sem))
                else:
                    sem_matrix = np.dstack((sem_matrix, sem*sem2))
                    
            sem_matrix = sem_matrix[:,:,1:]
            d_position = np.where(sem_matrix == np.max(sem_matrix))[2]
            self.d_best = d_vector[d_position[0]]
            sem = sem_matrix[:,:,d_position]
                    
        self.sem_matrix = sem_matrix
        self.sem = sem
        
        #####################################################
        
    def plot(self, data, tini, tend, filename, db, path_results='./report/', savefig=False):
        
        width = 15/2.54
        fontsize = 10
        plt.rcParams['axes.linewidth'] = 0.3
        plt.rc('xtick', labelsize=8) 
        plt.rc('ytick', labelsize=8)
        
        fig, ((ax1, ax2, ax3)) = plt.subplots(3, 1,  figsize=(width,4), sharex=True, constrained_layout=False, gridspec_kw={'height_ratios': [1, 2, 3]})
        
        cut = 3 # remove the last 3 columns of the semblance which are not properly computed for every scanning

        time = np.arange(self.npts-(db*cut))*self.dt
        coh_vector, threshold = self.coh, self.threshold 
        
        time_tmp = np.linspace(time.min(), time.max(), len(coh_vector))
        
        ax1.plot(time_tmp, coh_vector, linewidth=.4)
        ax1.plot(time_tmp, threshold, 'r--', linewidth=.4)
        ax1.set_ylabel('Max. coherence', fontsize=fontsize)
        ax1.spines[['right', 'top']].set_visible(False)
                
        events_clean=self.events
        
        for ev in events_clean:
            ax1.axvline(x = ev, ls='--', color='k',lw=.6)

        divider = make_axes_locatable(ax1)
        cax2 = divider.append_axes("right", size="1.4%", pad=.05)
        cax2.axis('off')
        
        ## AX2 ##
        depth=np.arange(self.ntrs)*self.dx
        im2 = ax2.imshow(self.sem, extent=[min(time),max(time),max(self.curv),min(self.curv)],cmap='viridis',aspect='auto')
        ax2.set_ylabel('Curv. coeff.', fontsize=fontsize)
        
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='1.4%', pad=.05)
        fig.colorbar(im2, cax=cax, orientation='vertical')
        cax.xaxis.set_ticks_position("none")
                
        ## AX3 ##
        
        im3 = ax3.imshow(data[:,:-cut*db],extent=[min(time),max(time),max(depth),min(depth)],cmap='seismic',aspect='auto')
        ax3.set_ylabel('Linear Fiber Length [m]', fontsize=fontsize)
        ax3.set_xlabel('Relative time [s]', fontsize=fontsize)
        
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right', size='1.4%', pad=.05)
        fig.colorbar(im3, cax=cax, orientation='vertical')
        cax.xaxis.set_ticks_position("none")

        fig.tight_layout()
  
        
        if savefig!=False:
            plt.savefig(path_results+filename+'.eps', dpi=300)
            plt.savefig(path_results+filename+'.png', dpi=300)
            
            del fig, ax1, ax2, ax3, coh_vector, threshold, events_clean
        

############################
    def detected_events(self, min_numb_detections, max_dist_detections, path_results='./report/'):

        import csv
        from scipy import stats
        from scipy import signal
        path_results=path_results
        from obspy import UTCDateTime

        cut = 3 # remove the last 3 columns of the semblance which are not properly computed for every scanning
        self.sem = self.sem[:,:-cut]
        
        coh_vector = np.sum(np.square(self.sem), axis=0) # GIAN MARIA
        #coh_vector = np.sum(self.sem, axis=0) #JUAN
        
        
        
        #define a noise threshold above which to make detections 
        
        noise=stats.trim_mean(coh_vector, 0.05)

        threshold = np.full_like(coh_vector, noise) 

        self.coh, self.threshold = coh_vector, threshold

        duration = self.npts*self.dt

        tax = np.linspace(0, duration, len(self.coh))

        pos = np.where(self.coh > self.threshold)
        
        #y_logs=[]
        #for k, l in enumerate(coh_vector):
        #    if k < len(coh_vector)-1:
        #        y_logs.append(np.log(coh_vector[k+1]/coh_vector[k]))
        #
        #x_logs=np.linspace(1,len(y_logs),len(y_logs))
        #y_logs=np.cumsum(y_logs)
        #plt.plot(x_logs,y_logs)
        #noise1=np.sqrt(np.mean(np.square(y_logs)))
        #plt.axhline(y = noise1, color = 'r')
        #plt.savefig(fname+'.png', dpi=150)
        

        diff = []
        for i in pos:
            d = np.abs(self.coh[i] - self.threshold[i])
            diff.append(d)
        diff = np.array(diff)
        
        #print(self.coh[pos])

        #group close detections
        def grouper(iterable):
            prev = None
            group = []
            for item in iterable:
                if prev is None or item - prev <= max_dist_detections:
                    group.append(item)
                else:
                    yield group
                    group = [item]
                prev = item
            if group:
                yield group
        
        pos1=(dict(enumerate(grouper(pos[0]), 1)))
        #print(pos1)
        
        pos_ev=[]
        for k,i in pos1.items():
            
            ## check that in each "event" there are at least a certain number of detections 
            semb_coeff=(self.coh[i])
            
            if len(i)>=min_numb_detections:# and np.mean(semb_coeff)/noise>=1.8:
                detect_= np.min(i)
                #if detect_<=2: #here I want to remove continuations of events from the previous file
                #    continue
                noise_det=np.sqrt(np.mean(np.square(coh_vector[detect_-22:detect_-2])))
                if detect_ <=22:
                    noise_det=noise*2.0

                signal_det=np.sqrt(np.mean(np.square(semb_coeff[:30])))
                #signal_det=np.sqrt(np.mean(np.square(coh_vector[detect_:detect_+30])))
                SNR_det=10*np.log(signal_det/noise_det)
                #print(SNR_det,'SNR',detect_)
                if SNR_det>=6:
                    pos_ev.append(np.min(i))
                
                if len (i) > 60:
                    #a=(signal.find_peaks(semb_coeff,height=0.002))[0][1:]
                    #g=c['peak_heights']
                    sample_diff=np.diff(semb_coeff,n=10)[15:]
                    max_sample_diff=np.amax(sample_diff)
                    detection=int(np.where(sample_diff==max_sample_diff)[0])+15
                    #print(sample_diff,max_sample_diff)
                    #a=signal.argrelmax(semb_coeff,order=30)[0][1:]
                    noise_=np.sqrt(np.mean(np.square(semb_coeff[detection-17:detection-2])))
                    signal_=np.sqrt(np.mean(np.square(semb_coeff[detection:detection+20])))
                    SNR=10*np.log(signal_/noise_)
                    #print(SNR,i[int(detection)])
                    if  SNR>4:# and a[0][0]:
                        pos_ev.append((i[int(detection)]))
        
        events = tax[pos_ev]
        events = np.around(events, decimals=3)
        fname2 = [self.fname] * len(events)
        
               
        abs_time = []
        
        if self.format != 'tdms':
            for i in events:
                abs_time.append(self.starttime + i)
        else:
            for time_ev in events:
                time_ev=("%.3f" % float(time_ev))
                secs, ms = str(time_ev).split('.')
                secs, ms = int(secs), int(ms)
                tmp= self.starttime + np.timedelta64(secs,'s') + np.timedelta64(ms,'ms')
                abs_time.append(tmp)
        
               
        events2 = events.tolist()
        if np.any(events)==False:
            None
        else:
            with open('{}{}.out'.format(path_results, self.fname), 'w') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerows(zip(fname2, events2, abs_time))
                                
        # ADD HEADERS
        self.events = events
        return self.events, self.coh, self.threshold
        del fname2, events2, abs_time
    
  