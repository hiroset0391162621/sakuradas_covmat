import datetime
import numpy  as np

fiber = 'nojiri' #'round'

hdf5_starttime_jst = datetime.datetime(2025, 1, 9, 0, 0, 0)
hdf5_endttime_jst = datetime.datetime(2025, 1, 10, 0, 0, 0)
Nseconds = int( (hdf5_endttime_jst-hdf5_starttime_jst).total_seconds() )
N_minute = int( (hdf5_endttime_jst - hdf5_starttime_jst).total_seconds() / 60.0 )

Fs = 100
low_pass = 0.2
high_pass = 50

overlap = 0.5  ### [0,1] step for sliding main windows-> preproc_spectral_secs*overlap
average = 30
window_duration_sec = 10.0

hdf5_dirname = "hdf5/"

used_channel_list = [str(_).zfill(4) for _ in range(400, 850, 50)]   
used_channel_num_list = np.array(used_channel_list, dtype=np.int64)