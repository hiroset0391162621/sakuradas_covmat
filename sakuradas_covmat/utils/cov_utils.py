import glob
import os
import sys
import datetime
from obspy.core import Stream
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")

try:
    import scienceplots
except:
    pass
plt.style.use(["science", "nature"])
plt.rcParams['xtick.direction'] = "inout"
plt.rcParams['ytick.direction'] = "inout"
plt.rcParams["text.usetex"] = False
plt.rcParams['agg.path.chunksize'] = 100000
plt.rcParams["date.converter"] = "concise"
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

sys.path.append("../cov")
sys.path.append("cov")
import covariancematrix
import array_cov

from read_hdf5 import read_hdf5

sys.path.append("../")
from Params import *



def network_covmat():
    
    
    
    N_minute = int( (hdf5_endttime_jst - hdf5_starttime_jst).total_seconds() / 60.0 )
    windL = 60.0*N_minute
    print("N_minute", N_minute)
    
    hdf5_starttime_utc = hdf5_starttime_jst + datetime.timedelta(hours=-9)
    hdf5_file_list = []
    for mm in range(N_minute):
        ts_utc = hdf5_starttime_utc + datetime.timedelta(minutes=mm)
        hdf5_dirname = "/Users/hirosetakashi/Volumes/noise_monitoring/noise_monitoring/DAS/Tohoku_15/Fiber-2_HDF5/"+ts_utc.strftime("%Y")+"/"+ts_utc.strftime("%m")+"/"+ts_utc.strftime("%d")+"/" 
        filename = glob.glob(
            hdf5_dirname+"decimator_"+ts_utc.strftime("%Y-%m-%d_%H.%M.%S")+"_UTC_"+"*.h5"
        )[0] 
        print(filename)     
        hdf5_file_list.append(filename)
        
    stream_minute = Stream()
    for i in range(len(hdf5_file_list)):
        stream_minute += read_hdf5(hdf5_file_list[i], fiber, used_channel_num_list)
    
    stream_minute.merge(method=1)
    for tr in stream_minute:
        if np.ma.is_masked(tr.data):
            tr.data = tr.data.filled(0)  
    
    stream_minute.resample(Fs, no_filter=False, window="hann")
    
    print(stream_minute)
    
    
    

    stream_cov = array_cov.ArrayStream()

    print('channels', used_channel_list)
    for tr in stream_minute:
        if tr.stats.station in used_channel_list:
            stream_cov += tr.copy()
    
    
    
    
    # frequency limits for filtering (depends on the target signal)
    
    sampling_rate = stream_cov[
        0
    ].stats.sampling_rate  # assumes all streams have the same sampling rate


    preproc_spectral_secs = (
        window_duration_sec * average * overlap
    )  # dT_main ### length of mainwindow: window_duration_sec * average * overlap
    print('length of mainwindow: window_duration_sec * average * overlap', preproc_spectral_secs)
    
    
    # stream_cov.detrend(type="demean")
    # stream_cov.detrend(type="linear")
    stream_cov.filter(
        type="bandpass",
        freqmin=low_pass,
        freqmax=high_pass,
        corners=3,
        zerophase=True,
    )
    
    stream_cov.plot()
    
    
    # stream_plot = stream_cov.copy()

    # ## Preprocess
    # # stream_cov.preprocess(
    # #     domain="spectral", method="smooth", window_duration_sec=preproc_spectral_secs
    # # )  # preproc_spectral_secs
    # stream_cov.preprocess(domain="temporal", method="onebit")


    # times, frequencies, covariances = covariancematrix.calculate(
    #     stream_cov, window_duration_sec, average, average_step=None
    # )
    
    # print(times)

    # # Spectral width
    # spectral_width = covariances.coherence(kind="spectral_width")
    
    
    # fig = plt.figure(figsize=(10, 4), constrained_layout=True)
    
    # ax0 = fig.add_subplot(211)
    # lapset = np.arange(0, windL, 1/Fs)
    # for i in range(len(stream_plot)):
    #     ax0.plot(stream_plot[i].times("matplotlib"), stream_plot[i].data/np.nanmax(np.abs(stream_plot[i].data)), lw=0.5, label=stream_plot[i].stats.network.lower()+stream_plot[i].stats.station.replace(" ", ""))
    
    # ax0.set_ylabel("norm. amp.", fontsize=12)
    # ax0.set_ylim(-1, 1)
    # #ax0.set_xlim(0, windL)
    # #ax0.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=8)
    # for spine in ax0.spines.values():
    #     spine.set_linewidth(1.5) 
    # ax0.tick_params(axis='both', which='major', length=4, width=1)  
    # ax0.tick_params(axis='both', which='minor', length=2, width=0.75)
    # ax0.tick_params(which='both', direction='out')
    
    # ax1 = fig.add_subplot(212)
    # ny, nx = spectral_width.T.shape
    # xmin, xmax = times[0], times[-1] + 1/Fs
    # ymin, ymax = frequencies[0], frequencies[-1]
    # x = np.linspace(xmin, xmax, nx + 1)
    # y = np.linspace(ymin, ymax, ny + 1)
    # print(ymin, ymax)
    # X, Y = np.meshgrid(x, y)
    # img = ax1.pcolormesh(X, Y, spectral_width.T, cmap="RdYlBu", shading='auto')
    
    # # img = ax1.imshow(
    # #     spectral_width.T,
    # #     origin="lower",
    # #     cmap="RdYlBu",
    # #     interpolation="none",
    # #     extent=[
    # #         0,
    # #         windL,
    # #         0,
    # #         sampling_rate,
    # #     ],
    # #     aspect="auto",
    # #     # vmin=0,
    # #     # vmax=4
    # # )
    # cbar = plt.colorbar(img, ax=ax1)
    # cbar.set_label(
    #     "spectral width", fontsize=10
    # )
    # ax1.set_xlim(0, windL)
    # ax1.set_ylim(
    #     [low_pass, stream_cov[0].stats.sampling_rate / 2]
    # )  # hide microseismic background noise below 0.5Hz
    # ax1.set_ylabel("Frequency [Hz]", fontsize=12)
    # ax1.set_yscale("log")
    # ax1.set_xlabel("Time [s]", fontsize=12)
    # for spine in ax1.spines.values():
    #     spine.set_linewidth(1.5) 
    # ax1.tick_params(axis='both', which='major', length=4, width=1)  
    # ax1.tick_params(axis='both', which='minor', length=2, width=0.75)
    # ax1.tick_params(which='both', direction='out')
    
    # plt.suptitle(fiber+' '+stream_cov[0].stats.starttime.strftime("%Y-%m-%d %H:%M:%S") + " - " + stream_cov[0].stats.endtime.strftime("%Y-%m-%d %H:%M:%S"), fontsize=12)
    
    
    # os.makedirs('figure', exist_ok=True)
    # plt.savefig('figure/specw_'+stream_cov[0].stats.starttime.strftime("%Y%m%d-%H%M%S")+"_"+str(Nseconds).zfill(4)+".png", dpi=300, bbox_inches='tight')
    # plt.show()
    

    

if __name__ == "__main__":
    
    network_covmat()