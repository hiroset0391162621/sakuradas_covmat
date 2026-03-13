import datetime
import numpy  as np

fiber = 'round' #'round', nojiri

hdf5_starttime_jst = datetime.datetime(2022, 12, 8, 0, 0, 0)
hdf5_endttime_jst = datetime.datetime(2022, 12, 9, 0, 0, 0)
Nseconds = int( (hdf5_endttime_jst-hdf5_starttime_jst).total_seconds() )
N_minute = int( (hdf5_endttime_jst - hdf5_starttime_jst).total_seconds() / 60.0 )

Fs = 20.0
low_pass = 0.1
high_pass = 10

overlap = 0.5  ### [0,1] step for sliding main windows-> preproc_spectral_secs*overlap
average = 48
window_duration_sec = 300.0 #10.0

hdf5_dirname =  "/Volumes/data/sakura/das-r8/" #"hdf5/"

used_channel_list = [str(_).zfill(4) for _ in range(100, 2500, 500)]   
used_channel_num_list = np.array(used_channel_list, dtype=np.int64)