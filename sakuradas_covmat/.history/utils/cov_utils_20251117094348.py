import glob
import os
import sys
import datetime
from obspy.core import Stream, Trace, UTCDateTime, read
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
import warnings
warnings.simplefilter("ignore")
from scipy.io import FortranFile
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

from read_hdf5 import read_hdf5, read_hdf5_singlechannel
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

sys.path.append("../")
from Params import *



def network_covmat(read_hdf5=True):
    
    N_minute = int( (hdf5_endttime_jst - hdf5_starttime_jst).total_seconds() / 60.0 )
    windL = 60.0*N_minute
    print("N_minute", N_minute)
    
    hdf5_starttime_utc = hdf5_starttime_jst + datetime.timedelta(hours=-9)
    hdf5_file_list = []
    missing_files = []
    for mm in range(N_minute):
        ts_utc = hdf5_starttime_utc + datetime.timedelta(minutes=mm)
        hdf5_dirname = "/Volumes/data/sakura/das-r8/"+ts_utc.strftime("%m")+"/"+ts_utc.strftime("%d")+"/"  #"/Users/hirosetakashi/Volumes/noise_monitoring/noise_monitoring/DAS/Tohoku_15/Fiber-2_HDF5/"+ts_utc.strftime("%Y")+"/"+ts_utc.strftime("%m")+"/"+ts_utc.strftime("%d")+"/" 
        
        file_pattern = hdf5_dirname+"decimator_"+ts_utc.strftime("%Y-%m-%d_%H.%M")+".00_UTC"+".h5"
        matched_files = glob.glob(file_pattern)
        
        if len(matched_files) > 0:
            filename = matched_files[0]
            print(filename)     
            hdf5_file_list.append(filename)
        else:
            print(f"警告: ファイルが見つかりません: {file_pattern}")
            missing_files.append((ts_utc, mm))
            hdf5_file_list.append(None)  # Noneを追加して位置を保持
    
    if read_hdf5:
        stream_minute = Stream()
        
        # 【高速化1】ファイルごとにチャネルをまとめて読み込む（HDF5ファイルの開閉を最小化）
        def read_file_with_channels(filename, sampling_rate_original):
            """1つのHDF5ファイルから複数チャネルを一度に読み込む（ファイルがない場合はゼロ埋め）"""
            if filename is None:
                # ファイルが存在しない場合、ゼロ埋めのStreamを作成
                file_streams = Stream()
                # 1分 = 60秒のデータを想定（元のサンプリングレートで）
                sampling_rate = sampling_rate_original  # HDF5ファイルの元のサンプリングレート（適宜調整）
                npts = int(60 * sampling_rate)  # 1分間のサンプル数
                
                for channel in used_channel_num_list:
                    zero_data = np.zeros(npts, dtype=np.float32)
                    tr = Trace(zero_data)
                    tr.stats.starttime = UTCDateTime(hdf5_starttime_jst)  # 仮の開始時刻
                    tr.stats.sampling_rate = sampling_rate
                    if fiber == 'round':
                        tr.stats.channel = "X"
                        tr.stats.network = "SAK"
                        tr.stats.station = str(channel).zfill(4)
                    elif fiber == 'nojiri':
                        tr.stats.channel = "X"
                        tr.stats.network = "NOJ"
                        tr.stats.station = str(channel).zfill(4)
                    file_streams += tr
                return file_streams
            else:
                # 通常の読み込み（エラー処理付き）
                try:
                    file_streams = Stream()
                    for channel in used_channel_num_list:
                        file_streams += read_hdf5_singlechannel(filename, fiber, channel)
                    return file_streams
                except (OSError, IOError, PermissionError, Exception) as e:
                    # ファイル読み込みエラー（破損、権限エラーなど）
                    print(f"エラー: ファイル読み込み失敗 {filename}: {type(e).__name__}: {e}")
                    print(f"  -> ゼロ埋めで代替します")
                    
                    # ゼロ埋めのStreamを返す
                    file_streams = Stream()
                    sampling_rate = sampling_rate_original
                    npts = int(60 * sampling_rate)
                    
                    for channel in used_channel_num_list:
                        zero_data = np.zeros(npts, dtype=np.float32)
                        tr = Trace(zero_data)
                        tr.stats.starttime = UTCDateTime(hdf5_starttime_jst)
                        tr.stats.sampling_rate = sampling_rate
                        if fiber == 'round':
                            tr.stats.channel = "X"
                            tr.stats.network = "SAK"
                            tr.stats.station = str(channel).zfill(4)
                        elif fiber == 'nojiri':
                            tr.stats.channel = "X"
                            tr.stats.network = "NOJ"
                            tr.stats.station = str(channel).zfill(4)
                        file_streams += tr
                    return file_streams
        
        # 【高速化2】スレッド数を増やす（外付けストレージでも16-24が効果的）
        # 【高速化3】chunksize指定で効率的にタスクを配分
        # chunksize設定方針：
        # - タスク数が少ない、処理時間にばらつき → chunksize=1（負荷分散優先）
        # - タスク数が多い、処理時間が均一 → chunksize=タスク数/(ワーカー数*4)（オーバーヘッド削減）
        max_workers = 24
        optimal_chunksize = max(1, len(hdf5_file_list) // (max_workers * 4))
        # ファイルごとに処理時間が異なる可能性があるため、小さめのchunksizeで負荷分散
        chunksize = min(optimal_chunksize, 10) if len(hdf5_file_list) > 100 else 1
        
        print(f"並列読み込み開始: {len(hdf5_file_list)}ファイル ({len(missing_files)}個欠損) x {len(used_channel_num_list)}チャネル")
        print(f"並列設定: max_workers={max_workers}, chunksize={chunksize}")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            stream_list = list(executor.map(read_file_with_channels, hdf5_file_list, [sampling_rate_original]*len(hdf5_file_list), chunksize=chunksize))
        
        # Streamに統合（効率化）
        for stream in stream_list:
            stream_minute += stream
        
        print(f"データ統合完了: {len(stream_minute)} traces")
        
        # 【高速化4】merge前にソート（処理が速くなる）
        stream_minute.sort(['network', 'station'])
        stream_minute.merge(method=1)
        
        # マスク処理
        for tr in stream_minute:
            if np.ma.is_masked(tr.data):
                tr.data = tr.data.filled(0)  
        
        # 【高速化5】リサンプリングを並列化
        print(f"リサンプリング開始: {stream_minute[0].stats.sampling_rate}Hz -> {Fs}Hz")
        stream_minute.resample(Fs, no_filter=False, window="hann")
        
        # ゼロ埋めでデータを終了時刻まで延長（+1サンプル追加で境界問題を回避）
        target_endtime = UTCDateTime(hdf5_endttime_jst)
        for tr in stream_minute:
            current_endtime = tr.stats.endtime
            if current_endtime < target_endtime:
                # 終了時刻までのサンプル数を計算（切り上げで確実に含める）
                time_diff = target_endtime - current_endtime
                npts_to_add = int(np.ceil(time_diff * tr.stats.sampling_rate))
                if npts_to_add > 0:
                    zero_padding = np.zeros(npts_to_add, dtype=tr.data.dtype)
                    tr.data = np.concatenate([tr.data, zero_padding])
                    tr.stats.npts = len(tr.data)
        
        print(stream_minute)
        print(f"データ長確認: {stream_minute[0].stats.npts}サンプル = {stream_minute[0].stats.npts / Fs}秒")
        # stream_minute.filter(type="bandpass", freqmin=1, freqmax=4, corners=3, zerophase=True)
        # stream_minute.plot()
        

        stream_cov = array_cov.ArrayStream()

        print('channels', used_channel_list)
        for tr in stream_minute:
            if tr.stats.station in used_channel_list:
                stream_cov += tr.copy()
        
        del stream_minute
        
        
        # frequency limits for filtering (depends on the target signal)
        
        sampling_rate = stream_cov[
            0
        ].stats.sampling_rate  # assumes all streams have the same sampling rate


        preproc_spectral_secs = (
            window_duration_sec * average * overlap
        )  # dT_main ### length of mainwindow: window_duration_sec * average * overlap
        print('length of mainwindow: window_duration_sec * average * overlap=', preproc_spectral_secs)
        
        
        # stream_cov.detrend(type="demean")
        # stream_cov.detrend(type="linear")
        stream_cov.filter(
            type="bandpass",
            freqmin=low_pass,
            freqmax=high_pass,
            corners=3,
            zerophase=True,
        )
        
    # # mseed形式で保存
    mseed_dir = 'mseed_data/sakura2022'
    os.makedirs(mseed_dir, exist_ok=True)
    mseed_filename = f'{mseed_dir}/filtered_{fiber}_{hdf5_starttime_jst.strftime("%Y%m%d-%H%M%S")}_{str(Nseconds).zfill(4)}.mseed'
    print(f"mseed保存中: {mseed_filename}")
    stream_cov.write(mseed_filename, format='MSEED')
    
    # mseedから読み込み
    print(f"mseed読み込み中: {mseed_filename}")
    stream_cov_temp = read(mseed_filename)
    
    # ArrayStreamに変換
    stream_cov = array_cov.ArrayStream()
    for tr in stream_cov_temp:
        stream_cov += tr
    print(f"mseed読み込み完了: {len(stream_cov)} traces")
    
    
    
    
    stream_plot = stream_cov.copy()

    
    stream_cov.preprocess(domain="temporal", method="onebit")


    times, frequencies, covariances = covariancematrix.calculate(
        stream_cov, window_duration_sec, average, step=average, average_step=None
    )
    
    print('times', times)

    # Spectral width
    spectral_width = covariances.coherence(kind="spectral_width")
    
    
    fig = plt.figure(figsize=(10, 4), constrained_layout=True)
    
    ax0 = fig.add_subplot(211)
    lapset = np.arange(0, windL, 1/Fs)
    for i in range(len(stream_plot)):
        ax0.plot(stream_plot[i].times("matplotlib"), stream_plot[i].data/np.nanmax(np.abs(stream_plot[i].data)), lw=0.5, label=stream_plot[i].stats.network.lower()+stream_plot[i].stats.station.replace(" ", ""))
    
    ax0.set_ylabel("norm. amp.", fontsize=12)
    ax0.set_ylim(-1, 1)
    ax0.set_xlim(hdf5_starttime_jst, hdf5_endttime_jst)
    #ax0.set_xlim(0, windL)
    #ax0.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=8)
    for spine in ax0.spines.values():
        spine.set_linewidth(1.5) 
    ax0.tick_params(axis='both', which='major', length=4, width=1)  
    ax0.tick_params(axis='both', which='minor', length=2, width=0.75)
    ax0.tick_params(which='both', direction='out')
    
    ax1 = fig.add_subplot(212)
    ny, nx = spectral_width.T.shape
    
    taxis = [ hdf5_starttime_jst+datetime.timedelta(seconds=_) for _ in times ]
    x = np.array([ mdates.date2num(_) for _ in taxis ])
    xmin, xmax = x[0], x[-1]
    #xmin, xmax = times[0], times[-1] + 1/Fs
    print(xmin, xmax)
    ymin, ymax = frequencies[0], frequencies[-1]
    #x = np.linspace(xmin, xmax, nx + 1)
    y = np.linspace(ymin, ymax, ny + 1)
    print(ymin, ymax)
    X, Y = np.meshgrid(x, y)
    img = ax1.pcolormesh(X, Y, spectral_width.T, cmap="RdYlBu", shading='auto', rasterized=True)
    
    # img = ax1.imshow(
    #     spectral_width.T,
    #     origin="lower",
    #     cmap="RdYlBu",
    #     interpolation="none",
    #     extent=[
    #         0,
    #         windL,
    #         0,
    #         sampling_rate,
    #     ],
    #     aspect="auto",
    #     # vmin=0,
    #     # vmax=4
    # )
    cbar = plt.colorbar(img, ax=ax1, pad=0.01)
    cbar.set_label(
        "spectral width", fontsize=10
    )
    #ax1.set_xlim(0, windL)
    ax1.set_xlim(hdf5_starttime_jst, hdf5_endttime_jst)
    ax1.set_ylim(
        [low_pass, stream_cov[0].stats.sampling_rate / 2]
    )  # hide microseismic background noise below 0.5Hz
    ax1.set_ylabel("Frequency [Hz]", fontsize=12)
    ax1.set_yscale("log")
    for spine in ax1.spines.values():
        spine.set_linewidth(1.5) 
    ax1.tick_params(axis='both', which='major', length=4, width=1)  
    ax1.tick_params(axis='both', which='minor', length=2, width=0.75)
    ax1.tick_params(which='both', direction='out')
    
    plt.suptitle(fiber+' '+hdf5_starttime_jst.strftime("%Y-%m-%d %H:%M:%S") + " - " + stream_cov[0].stats.endtime.strftime("%Y-%m-%d %H:%M:%S"), fontsize=12)
    
    
    os.makedirs('figure/specw_sakura2022', exist_ok=True)
    plt.savefig('figure/specw_sakura2022/specw_'+hdf5_starttime_jst.strftime("%Y%m%d-%H%M%S")+"_"+str(Nseconds).zfill(4)+".png", dpi=300, bbox_inches='tight')
    plt.close()

    # ### save spectral_width data
    os.makedirs('specw_data/sakura2022', exist_ok=True)
    np.savez_compressed('specw_data/sakura2022/specw_'+fiber+'_'+hdf5_starttime_jst.strftime("%Y%m%d-%H%M%S")+"_"+str(Nseconds).zfill(4)+".npz",
                        datetime_num=x,
                        frequencies=y,
                        spectral_width=spectral_width.T)
    



def network_covmat_sakbin():
    
    Fs = 50
    low_pass = 0.1
    high_pass = 2

    overlap = 0.5  ### [0,1] step for sliding main windows-> preproc_spectral_secs*overlap
    average = 20
    window_duration_sec = 60.0
        
    channels = list(range(250,7825,250))

    print('used channels', channels)
    
    N_minute = int( (hdf5_endttime_jst - hdf5_starttime_jst).total_seconds() / 60.0 )
    windL = 60.0*N_minute
    print("N_minute", N_minute)
    
    ts_jst = hdf5_starttime_jst
    bin_dirname = "/Users/hirosetakashi/Volumes/noise_monitoring/noise_monitoring/DAS/sakuData_1d_2_onebit_nospatialstacking_strainrate/"+ts_jst.strftime("%Y")+"/"+ts_jst.strftime("%m")+"/"+ts_jst.strftime("%d")+"/" 
    
    
    stream_cov = array_cov.ArrayStream()
    
    for ch_idx in channels: 
    
        f = FortranFile(bin_dirname+'sak'+str(ch_idx).zfill(4)+'.bin', 'r')
        bin_data = f.read_record(dtype=float)
        f.close()
        tr = Trace(bin_data)
        tr.stats.starttime = ts_jst
        tr.stats.sampling_rate = Fs
        tr.stats.station = str(ch_idx).zfill(4)
        tr.stats.network = 'sak'
        stream_cov += tr
        
        del tr
    
    print(stream_cov)
    
    # frequency limits for filtering (depends on the target signal)
    
    sampling_rate = stream_cov[0].stats.sampling_rate  # assumes all streams have the same sampling rate


    preproc_spectral_secs = (
        window_duration_sec * average * overlap
    )  # dT_main ### length of mainwindow: window_duration_sec * average * overlap
    print('length of mainwindow: window_duration_sec * average * overlap', preproc_spectral_secs)
    
    
    stream_plot = stream_cov.copy()
    stream_cov.preprocess(domain="temporal", method="onebit")


    times, frequencies, covariances = covariancematrix.calculate(
        stream_cov, window_duration_sec, average, step=average, average_step=None
    )
    
    print('times', times)
    print('frequencies', frequencies)

    # Spectral width
    spectral_width = covariances.coherence(kind="spectral_width")
    
    
    fig = plt.figure(figsize=(10, 4), constrained_layout=True)
    ax1 = fig.add_subplot(211)
    ny, nx = spectral_width.T.shape
    
    taxis = [ hdf5_starttime_jst+datetime.timedelta(seconds=_) for _ in times ]
    x = np.array([ mdates.date2num(_) for _ in taxis ])
    xmin, xmax = x[0], x[-1]
    #xmin, xmax = times[0], times[-1] + 1/Fs
    print(xmin, xmax)
    ymin, ymax = frequencies[0], frequencies[-1]
    #x = np.linspace(xmin, xmax, nx + 1)
    #y = np.linspace(ymin, ymax, ny + 1)
    print(ymin, ymax)
    print('spectral_width min, max', np.nanmin(spectral_width), np.nanmax(spectral_width))
    #X, Y = np.meshgrid(x, y)
    #img = ax1.pcolormesh(X, Y, spectral_width.T, cmap="RdYlBu", shading='auto', rasterized=True)
    
    img = ax1.imshow(
        spectral_width.T,
        origin="lower",
        cmap="RdYlBu",
        interpolation="none",
        extent=[
            xmin,
            xmax,
            ymin,
            ymax,
        ],
        aspect="auto",
        rasterized=True,
        vmin=0.5,
        vmax=5
    )
    
    cbar = plt.colorbar(img, ax=ax1, pad=0.01)
    cbar.set_label(
        "spectral width", fontsize=10
    )
    #ax1.set_xlim(0, windL)
    ax1.set_xlim(hdf5_starttime_jst, hdf5_endttime_jst)
    ax1.set_ylim(low_pass, high_pass)  # hide microseismic background noise below 0.5Hz
    ax1.set_ylabel("Frequency [Hz]", fontsize=12)
    ax1.set_yscale("log")
    for spine in ax1.spines.values():
        spine.set_linewidth(1.5) 
    ax1.tick_params(axis='both', which='major', length=4, width=1)  
    ax1.tick_params(axis='both', which='minor', length=2, width=0.75)
    ax1.tick_params(which='both', direction='out')
    
    plt.suptitle(hdf5_starttime_jst.strftime("%Y-%m-%d %H:%M:%S") + " - " + stream_cov[0].stats.endtime.strftime("%Y-%m-%d %H:%M:%S"), fontsize=12)
    
    
    os.makedirs('figure/round/', exist_ok=True)
    plt.savefig('figure/round/specw_'+hdf5_starttime_jst.strftime("%Y%m%d")+"_"+str(Nseconds).zfill(4)+".pdf", dpi=300, bbox_inches='tight')
    plt.show()
    
    

if __name__ == "__main__":
    
    t_starts = [
        # datetime.datetime(2022, 11, 15, 0, 0),
        # datetime.datetime(2022, 11, 16, 0, 0),
        # datetime.datetime(2022, 11, 17, 0, 0),
        # datetime.datetime(2022, 11, 18, 0, 0),
        # datetime.datetime(2022, 11, 19, 0, 0),
        # datetime.datetime(2022, 11, 20, 0, 0),
        # datetime.datetime(2022, 11, 21, 0, 0),
        ### datetime.datetime(2022, 11, 22, 0, 0), ### error
        datetime.datetime(2022, 11, 23, 0, 0), ### error
        # datetime.datetime(2022, 11, 24, 0, 0),
        # datetime.datetime(2022, 11, 25, 0, 0),
        # datetime.datetime(2022, 11, 26, 0, 0),
        # datetime.datetime(2022, 11, 27, 0, 0),
        # datetime.datetime(2022, 11, 28, 0, 0), 
        # datetime.datetime(2022, 11, 29, 0, 0),
        # datetime.datetime(2022, 11, 30, 0, 0),
        # datetime.datetime(2022, 12, 1, 0, 0),
        #datetime.datetime(2022, 12, 7, 0, 0), ### error
        # datetime.datetime(2022, 12, 8, 0, 0, 0),
        # datetime.datetime(2022, 12, 9, 0, 0, 0)
    ]
    
    for dd in range(len(t_starts)):
        hdf5_starttime_jst = t_starts[dd]
        hdf5_endttime_jst = t_starts[dd] + datetime.timedelta(days=1)
        network_covmat()
    
    
    
    