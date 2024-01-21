import time
import sys
sys.path.append('./das/')
from Detector import Detector

#%%
#define output directory for results

################    DAS object and filtering parameters   ##############################
path_input = './input/'
#fname = 'FORGE_DFIT_UTC_20220421_135024.398.tdms'
fname = 'FORGE_78-32_iDASv3-P11_UTC190423213209.sgy'
#fname = 'receivers.h5'
#fname = 'synthetic_seismogram_500_4500_DAS_4000.npy'
f = path_input + fname
file_format = 'segy'

dx = 1.02
gl = 10.
downsampling_rate = 500
ftype = 'bandpass'
freqmin, freqmax = 10, 249
k0 = True
low_vel_events = True

#xini=1350
#xend=2400
xini=250
xend=1160
tini=0
tend=5


################    Clustering parameters   ##############################

min_numb_detections=10
max_dist_detections=2

################    SEMBLANCE parameters   ##############################

ns_window = 20 #width of data window for semblance
a = 20
b = 375
b_step =  10  # db sample step along time 

c_min, c_max, c_step = 70, 500, 5 # set of parameters for characterization of the hyperbole (for semblance)

d_min, d_max, d_step = 0, 1200, 50 #dd sample step along distance, dmax maximum step along distance

d_static = 892

#svd singular value decomposition for weighting for the semblance matrix 
svd = False

# apply semblance analysis also along distance
lat_search = False
savefig = False



#%%
t0=time.time()

## Read and visualize input file
arr = Detector(f, dx, gl, fname, file_format); #arr.visualization(arr.traces, f, savefig=True)

## Select a subset of the data 
arr.data_select(endtime = tend,
                startlength = xini,
                endlength = xend); #arr.visualization(arr.traces, f)

# Denoise the selected data
arr.denoise(data = arr.traces,
            sampling_rate_new = downsampling_rate,
            ftype = ftype,
            fmin = freqmin,
            fmax = freqmax,
            k0=k0,
            low_vel_events = low_vel_events); #arr.visualization(arr.traces, f)


## Apply the detector
arr.detector(ns_window = ns_window,
              a = a,
              b_step = b_step,
              c_min = c_min,
              c_max = c_max,
              c_step = c_step,
              d_static = d_static,
              d_min = d_min,
              d_max = d_max,
              d_step = d_step,
              svd = svd,
              lat_search = lat_search)

print('Total time is {} seconds'.format(time.time()-t0))

## PLOT RESULTS
events = arr.detected_events(min_numb_detections,max_dist_detections)
arr.plot(arr.traces, tini, tend, fname, b_step, savefig=savefig)
#arr.plotfk(savefig=False)

#%%
################    SEMBLANCE parameters tuning   ##############################

if lat_search == False:
    arr.hyperbolae_tuning(a, b, d_static, c_min, c_max, c_step)

else:
    arr.hyperbolae_tuning(a, b, arr.d_best, c_min, c_max, c_step)
    
    
