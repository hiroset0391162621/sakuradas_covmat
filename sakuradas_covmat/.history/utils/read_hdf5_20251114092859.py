import h5py
import numpy as np
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt
from obspy.core import Stream, Trace
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial



def phase2strain(phase, gauge_length):
    lamda = 1550.12 * 1e-9 
    n = 1.4682 
    xi = 0.78
    return (lamda*phase) / (4*np.pi*n*xi*gauge_length)


# JSTに変換する関数
def to_jst(utc_time):
    jst_time = utc_time + timedelta(hours=9)  
    return jst_time

# def read_hdf5(filename, fiber, channels):
    
#     with h5py.File(filename, "r") as h5file:
#         raw_data = h5file['Acquisition/Raw[0]/RawData'][:, channels]
#         raw_data = raw_data.T
        
#         start_time_str = h5file["/Acquisition/Raw[0]/RawDataTime"].attrs["StartTime"].decode('utf-8')
#         start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))

#         end_time_str = h5file["/Acquisition/Raw[0]/RawDataTime"].attrs["EndTime"].decode('utf-8')
#         end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))

#         output_data_rate = h5file["/Acquisition/Raw[0]"].attrs["OutputDataRate"]
        
#         G = h5file["/Acquisition"].attrs["GaugeLength"]
        
#     # StartTimeとEndTimeをJSTに変換
#     start_time_jst = to_jst(start_time)
#     end_time_jst = to_jst(end_time)

#     # 結果の表示
#     print("Raw Data Shape:", raw_data.shape)
#     print("Start Time (JST):", start_time_jst)
#     print("End Time (JST):", end_time_jst)
#     print("Output Data Rate (Hz):", output_data_rate)


#     st_minute = Stream()
#     #for i in range(raw_data.shape[0]):
#     for i in range(len(channels)):
#         tr = Trace(phase2strain(raw_data[i,:], G))
#         tr.stats.starttime = start_time_jst
#         tr.stats.sampling_rate = output_data_rate
#         if fiber=='round':
#             tr.stats.channel = "X"
#             tr.stats.network = "SAK"
#             #tr.stats.station = str(i).zfill(4)
#             tr.stats.station = str(channels[i]).zfill(4)
#         elif fiber=='nojiri':
#             tr.stats.channel = "X"
#             tr.stats.network = "NOJ"
#             #tr.stats.station = str(i).zfill(4)
#             tr.stats.station = str(channels[i]).zfill(4)
#         st_minute += tr
    
#     # print(st_minute)
#     # print(st_minute[0].stats)
    
    

#     return st_minute


def read_hdf5(filename, fiber, channels=None, n_jobs=1):
    """
    HDF5ファイルからDASデータを読み込む
    
    Parameters
    ----------
    filename : str
        HDF5ファイルのパス
    fiber : str
        'round' または 'nojiri'
    channels : array-like, optional
        読み込むチャネルのインデックス。Noneの場合は全チャネル
    n_jobs : int, default=1
        並列処理に使用するワーカー数。1の場合は並列化なし
        
    Returns
    -------
    obspy.Stream
        読み込んだデータを含むStreamオブジェクト
    """
    with h5py.File(filename, "r") as h5file:
        dataset = h5file['Acquisition/Raw[0]/RawData']
        n_total_channels = dataset.shape[1]
        if channels is None:
            channel_indices = np.arange(n_total_channels)
        else:
            channel_indices = np.asarray(channels, dtype=int)
            if channel_indices.ndim != 1:
                raise ValueError("channels must be a 1D sequence of indices")
        if channel_indices.size == 0:
            return Stream()
        if (channel_indices < 0).any() or (channel_indices >= n_total_channels).any():
            raise IndexError("channel index out of bounds")

        raw_data = dataset[:, channel_indices]

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

    # convert phase to strain in a vectorized fashion to avoid per-channel loops
    strain_scale = phase2strain(1.0, G)
    strain_data = np.ascontiguousarray(raw_data.T, dtype=np.float32)
    strain_data *= np.float32(strain_scale)

    network_lookup = {"round": "SAK", "nojiri": "NOJ"}
    network = network_lookup.get(fiber, fiber.upper())

    # Trace生成を並列化（チャネル数が多い場合に効果的）
    if n_jobs > 1 and len(channel_indices) > 10:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            trace_fn = partial(_create_trace, 
                             strain_data=strain_data,
                             channel_indices=channel_indices,
                             start_time_jst=start_time_jst,
                             output_data_rate=output_data_rate,
                             network=network)
            traces = list(executor.map(trace_fn, range(len(channel_indices))))
    else:
        traces = []
        for idx, channel_id in enumerate(channel_indices):
            stats = {
                "starttime": start_time_jst,
                "sampling_rate": output_data_rate,
                "channel": "X",
                "network": network,
                "station": str(int(channel_id)).zfill(4),
            }
            traces.append(Trace(data=strain_data[idx], header=stats))

    return Stream(traces=traces)


def _create_trace(idx, strain_data, channel_indices, start_time_jst, output_data_rate, network):
    """Trace生成のヘルパー関数（並列化用）"""
    channel_id = channel_indices[idx]
    stats = {
        "starttime": start_time_jst,
        "sampling_rate": output_data_rate,
        "channel": "X",
        "network": network,
        "station": str(int(channel_id)).zfill(4),
    }
    return Trace(data=strain_data[idx], header=stats)


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

def read_multiple_hdf5(filenames, fiber, channels=None, n_jobs=4):
    """
    複数のHDF5ファイルを並列で読み込む
    
    Parameters
    ----------
    filenames : list of str
        HDF5ファイルパスのリスト
    fiber : str
        'round' または 'nojiri'
    channels : array-like, optional
        読み込むチャネルのインデックス
    n_jobs : int, default=4
        並列処理に使用するプロセス数
        
    Returns
    -------
    list of obspy.Stream
        各ファイルから読み込んだStreamオブジェクトのリスト
    """
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        read_fn = partial(read_hdf5, fiber=fiber, channels=channels, n_jobs=1)
        streams = list(executor.map(read_fn, filenames))
    return streams


if __name__ == "__main__":
    
    fiber = 'nojiri' #'round'
    filename = '../hdf5/decimator_2024-07-14_09.19.00_UTC_058993.h5'
    st_minute = read_hdf5(filename, fiber)
    print(st_minute)
