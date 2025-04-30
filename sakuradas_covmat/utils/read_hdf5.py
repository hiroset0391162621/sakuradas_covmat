import h5py
import numpy as np
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt
from obspy.core import Stream, Trace
import time



def phase2strain(phase, gauge_length):
    lamda = 1550.12 * 1e-9 
    n = 1.4682 
    xi = 0.78
    return (lamda*phase) / (4*np.pi*n*xi*gauge_length)


# JSTに変換する関数
def to_jst(utc_time):
    jst_time = utc_time + timedelta(hours=9)  
    return jst_time



def read_hdf5(filename, fiber, channels):
    
    with h5py.File(filename, "r") as h5file:
        raw_data = h5file['Acquisition/Raw[0]/RawData'][:, channels]
        raw_data = raw_data.T
        
        start_time_str = h5file["/Acquisition/Raw[0]/RawDataTime"].attrs["StartTime"].decode('utf-8')
        start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))

        end_time_str = h5file["/Acquisition/Raw[0]/RawDataTime"].attrs["EndTime"].decode('utf-8')
        end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))

        output_data_rate = h5file["/Acquisition/Raw[0]"].attrs["OutputDataRate"]
        
        G = h5file["/Acquisition"].attrs["GaugeLength"]
        
    # StartTimeとEndTimeをJSTに変換
    start_time_jst = to_jst(start_time)
    end_time_jst = to_jst(end_time)

    # 結果の表示
    print("Raw Data Shape:", raw_data.shape)
    print("Start Time (JST):", start_time_jst)
    print("End Time (JST):", end_time_jst)
    print("Output Data Rate (Hz):", output_data_rate)


    st_minute = Stream()
    #for i in range(raw_data.shape[0]):
    for i in range(len(channels)):
        tr = Trace(phase2strain(raw_data[i,:], G))
        tr.stats.starttime = start_time_jst
        tr.stats.sampling_rate = output_data_rate
        if fiber=='round':
            tr.stats.channel = "X"
            tr.stats.network = "SAK"
            #tr.stats.station = str(i).zfill(4)
            tr.stats.station = str(channels[i]).zfill(4)
        elif fiber=='nojiri':
            tr.stats.channel = "X"
            tr.stats.network = "NOJ"
            #tr.stats.station = str(i).zfill(4)
            tr.stats.station = str(channels[i]).zfill(4)
        st_minute += tr
    
    # print(st_minute)
    # print(st_minute[0].stats)
    
    

    return st_minute


def read_hdf5_singlechannel(filename, fiber, channel_idx):
    
    with h5py.File(filename, "r") as h5file:
        raw_data = h5file['Acquisition/Raw[0]/RawData'][:]
        raw_data = np.transpose(raw_data)
        raw_data = raw_data[channel_idx, :]
        
        start_time_str = h5file["/Acquisition/Raw[0]/RawDataTime"].attrs["StartTime"].decode('utf-8')
        start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))

        end_time_str = h5file["/Acquisition/Raw[0]/RawDataTime"].attrs["EndTime"].decode('utf-8')
        end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))

        output_data_rate = h5file["/Acquisition/Raw[0]"].attrs["OutputDataRate"]
        
        G = h5file["/Acquisition"].attrs["GaugeLength"]
        
    
    start_time_jst = to_jst(start_time)
    end_time_jst = to_jst(end_time)

    
    print("Raw Data Shape:", raw_data.shape)
    print("Start Time (JST):", start_time_jst)
    print("End Time (JST):", end_time_jst)
    print("Output Data Rate (Hz):", output_data_rate)


    st_minute = Stream()
    tr = Trace(phase2strain(raw_data, G))
    tr.stats.starttime = start_time_jst
    tr.stats.sampling_rate = output_data_rate
    if fiber=='round':
        tr.stats.channel = "X"
        tr.stats.network = "SAK"
        tr.stats.station = str(channel_idx).zfill(4)
    elif fiber=='nojiri':
        tr.stats.channel = "X"
        tr.stats.network = "NOJ"
        tr.stats.station = str(channel_idx).zfill(4)
    st_minute += tr
    
    
    
    # print(st_minute)
    # print(st_minute[0].stats)
    # print(st_minute[0].data)
    

    return st_minute

if __name__ == "__main__":
    
    fiber = 'nojiri' #'round'
    filename = '../hdf5/decimator_2024-07-14_09.19.00_UTC_058993.h5'
    st_minute = read_hdf5(filename, fiber)
    print(st_minute)
