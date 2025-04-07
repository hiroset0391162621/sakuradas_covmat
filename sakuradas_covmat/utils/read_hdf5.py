import h5py
import numpy as np
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt
from obspy.core import Stream, Trace
import time



def phase2strain(phase):
    lamda = 1550.12 * 1e-9 ### 光の波長 [m]
    n = 1.4682 ### 光ファイバーの屈折率
    xi = 0.78 ### photo-elastic scaling factor
    G = 9.9259261 ### ゲージ⻑ [m] 
    return (lamda*phase) / (4*np.pi*n*xi*G)


# JSTに変換する関数
def to_jst(utc_time):
    jst_time = utc_time + timedelta(hours=9)  
    return jst_time



def read_hdf5(filename, fiber):
    
    with h5py.File(filename, "r") as h5file:
        raw_data = h5file['Acquisition/Raw[0]/RawData'][:]
        raw_data = np.transpose(raw_data)

        # StartTimeの取得
        start_time_str = h5file["/Acquisition/Raw[0]/RawDataTime"].attrs["StartTime"].decode('utf-8')
        start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))

        # EndTimeの取得
        end_time_str = h5file["/Acquisition/Raw[0]/RawDataTime"].attrs["EndTime"].decode('utf-8')
        end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))

        # OutputDataRateの取得
        output_data_rate = h5file["/Acquisition/Raw[0]"].attrs["OutputDataRate"]
        
        
    # StartTimeとEndTimeをJSTに変換
    start_time_jst = to_jst(start_time)
    end_time_jst = to_jst(end_time)

    # 結果の表示
    print("Raw Data Shape:", raw_data.shape)
    print("Start Time (JST):", start_time_jst)
    print("End Time (JST):", end_time_jst)
    print("Output Data Rate (Hz):", output_data_rate)


    st_minute = Stream()
    for i in range(raw_data.shape[0]):
        tr = Trace(phase2strain(raw_data[i,:]))
        tr.stats.starttime = start_time_jst
        tr.stats.sampling_rate = output_data_rate
        if fiber=='round':
            tr.stats.channel = "sak"+str(i).zfill(4)
        elif fiber=='nojiri':
            tr.stats.channel = "X"
            tr.stats.network = "NOJ"
            tr.stats.station = str(i).zfill(4)
        st_minute += tr
    
    # print(st_minute)
    # print(st_minute[0].stats)
    
    

    return st_minute

if __name__ == "__main__":
    
    fiber = 'nojiri' #'round'
    filename = '../hdf5/decimator_2024-07-14_09.19.00_UTC_058993.h5'
    st_minute = read_hdf5(filename, fiber)
    print(st_minute)
