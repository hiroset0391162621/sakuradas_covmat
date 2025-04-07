import datetime


fiber = 'nojiri' #'round'

hdf5_starttime_jst = datetime.datetime(2023, 12, 1, 0, 0, 0)
hdf5_endttime_jst = datetime.datetime(2023, 12, 1, 0, 10, 0)

Fs = 100
low_pass = 0.2
high_pass = 50

overlap = 0.5  ### [0,1] step for sliding main windows-> preproc_spectral_secs*overlap
average = 10
window_duration_sec = 5.0

hdf5_dirname = "hdf5/"

used_channel_list = [str(_).zfill(4) for _ in range(100, 805, 5)]   