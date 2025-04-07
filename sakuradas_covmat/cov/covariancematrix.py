import numpy as np
import obspy
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh, eigh


class CovarianceMatrix(np.ndarray):
    

    def __new__(cls, input_array):
        
        obj = np.asarray(input_array, dtype=complex).view(cls)
        return obj

    def coherence(self, kind="spectral_width", epsilon=1e-10):
        
        
        if kind == "spectral_width":
            eigenvalues = self.eigenvalues(norm=sum)
            indices = np.arange(self.shape[-1])
            return np.multiply(eigenvalues, indices).sum(axis=-1)
        
        elif kind == "entropy":
            eigenvalues = self.eigenvalues(norm=sum)
            log_eigenvalues = np.log(eigenvalues + epsilon)
            return -np.sum(eigenvalues * log_eigenvalues, axis=-1)
        
        else:
            message = "{} is not an available option for kind."
            raise ValueError(message.format(kind))
        
        
    def rectlinearlity_planarity(self):
        
        matrices = self._flat()
        eigenvalues = np.zeros((matrices.shape[0], matrices.shape[-1]))
        for i, matrix in enumerate(matrices):
            u1, u2, u3 = eigh(matrix)[1]
            print(u1, u2, u3)
            
                
        #return P
        

    def eigenvalues(self, norm=max):
        
        
        matrices = self._flat()
        eigenvalues = np.zeros((matrices.shape[0], matrices.shape[-1]))
        for i, matrix in enumerate(matrices):
            eigenvalues[i] = np.abs(eigvalsh(matrix)[::-1])
            eigenvalues[i] /= norm(eigenvalues[i])

        
        return eigenvalues.reshape(self.shape[:-1])
    
    def eigenvalues_nonnorm(self):
        
        matrices = self._flat()
        eigenvalues = np.zeros((matrices.shape[0], matrices.shape[-1]))
        for i, matrix in enumerate(matrices):
            eigenvalues[i] = np.abs(eigvalsh(matrix)[::-1])
            eigenvalues[i] /= np.nanmax(eigenvalues[i])

        Lambda_mat = eigenvalues.reshape(self.shape[:-1])
        
        Shannon_entropy = np.zeros((Lambda_mat.shape[0], Lambda_mat.shape[1]))*np.nan
        Coherency = np.zeros((Lambda_mat.shape[0], Lambda_mat.shape[1]))*np.nan
        Cov = np.zeros((Lambda_mat.shape[0], Lambda_mat.shape[1]))*np.nan
        First_eigenval = np.zeros((Lambda_mat.shape[0], Lambda_mat.shape[1]))*np.nan
        
        for tt in range(Lambda_mat.shape[0]):
            
            ### (n_freq, n_stations)
            Lambda_mat_f_st = Lambda_mat[tt].copy() 
            N_freq, N_st = Lambda_mat_f_st.shape
            lam_1 = Lambda_mat_f_st[:,0]
            eigsum = np.nansum(Lambda_mat_f_st, axis=1) 
            
            ### 0. Normalize 
            #Lambda_mat_f_st_fmean = np.mean(Lambda_mat_f_st, axis=0)
            #Lambda_mat_f_st = Lambda_mat_f_st / Lambda_mat_f_st_fmean.reshape((1,Lambda_mat_f_st.shape[1]))
            
            ### 1. Shannon_entropy
            p_i = Lambda_mat_f_st.copy() / (eigsum.copy()).reshape((N_freq,1))
            
            #print('1', (-np.nansum( p_i * np.log(p_i), axis=1 )).shape)
            Shannon_entropy[tt,:] = -np.nansum( p_i * np.log(p_i), axis=1 )
            
            ### 2. Coherency
            Coherency[tt,:] = lam_1 / eigsum
            #print('2', (lam_1 / np.nansum(Lambda_mat_f_st, axis=1)).shape)
            
            ### 3. Covariance
            mu = np.nanmean(Lambda_mat_f_st, axis=1).reshape((N_freq,1))
            Cov[tt,:] = np.nansum((Lambda_mat_f_st-mu)**2, axis=1) /  N_st
            #print('3', (np.nansum((Lambda_mat_f_st-mu)**2, axis=1) / N).shape)
            
            ### 4. First eigen value
            First_eigenval[tt,:] = lam_1
            #print('4', lam_1.shape)
        
        return Shannon_entropy, Coherency, Cov, First_eigenval

    def eigenvectors(self, rank=0, covariance=False, polarization=False):
        
        # Initialization
        matrices = self._flat()
        eigenvectors = np.zeros((matrices.shape[0], matrices.shape[-1]), dtype=complex)
        eigenvalues = np.zeros( matrices.shape[0] )
        # Calculation over submatrices
        for i, m in enumerate(matrices):
            eigh_vals = eigh(m)
            eigenvectors[i] = eigh_vals[1][:, -1 - rank]
            eigenvalues[i] = eigh_vals[0][-1 - rank]

        eigenvalues = eigenvalues.reshape(self.shape[0], self.shape[1])
            
        if covariance:
            ec = np.zeros(self.shape, dtype=complex)
            ec = ec.view(CovarianceMatrix)
            ec = ec._flat()
            for i in range(eigenvectors.shape[0]):
                ec[i] = eigenvectors[i, :, None] * np.conj(eigenvectors[i])
            ec = ec.reshape(self.shape)
            return ec.view(CovarianceMatrix)
        else:
            if polarization:
                return np.real(eigenvectors).reshape(self.shape[:-1]), eigenvalues
            else:
                return np.abs(eigenvectors).reshape(self.shape[:-1]), eigenvalues

    def _flat(self):
        return self.reshape(-1, *self.shape[-2:])

    def triu(self, **kwargs):
        trii, trij = np.triu_indices(self.shape[-1], **kwargs)
        return self[..., trii, trij]


def calculate(stream, window_duration_sec, average, average_step=None, **kwargs):

    times, frequencies, spectra = stft(stream, window_duration_sec, **kwargs)

    # Parametrization
    step = average//2  if average_step is None else int(average * average_step)
    n_traces, n_windows, n_frequencies = spectra.shape

    # Times
    t_end = times[-1]
    times = times[:-1]
    times = times[: 1 - average : step]
    n_average = len(times)
    times = np.hstack((times, t_end))
    

    # Initialization
    cov_shape = (n_average, n_traces, n_traces, n_frequencies)
    covariance = np.zeros(cov_shape, dtype=complex)

    # Compute
    for t in range(n_average):
        covariance[t] = xcov(t, spectra, step, average)
        
        
    # Create frequencies vector for plotting as matplotlib now requires coordinates of pixel edges
    fs = stream[0].stats.sampling_rate
    frequencies_plotting = np.linspace(0, fs, len(frequencies) + 1)

    return (
        times,
        frequencies_plotting,
        covariance.view(CovarianceMatrix).transpose([0, -1, 1, 2]),
    )


def stft(
    stream,
    window_duration_sec,
    bandwidth=None,
    window_step_sec=None,
    window=np.hanning,
    times_kw=dict(),
    **kwargs
):

    # Time vector
    fs = stream[0].stats.sampling_rate
    npts = int(window_duration_sec * fs)
    step = npts // 2 if window_step_sec is None else int(window_step_sec * fs)
    times_kw.setdefault("type", "relative")
    times_kw.setdefault("reftime", None)
    if type(stream) is obspy.core.stream.Stream:
        times = stream[0].times(**times_kw)[: 1 - npts : step]
    else:
        times = stream.times(**times_kw)[: 1 - npts : step]
    n_times = len(times)

    # Frequency vector
    kwargs.setdefault("n", 2 * npts - 1)
    frequencies = np.linspace(0, fs, kwargs["n"])

    if bandwidth is not None:
        fin = (frequencies >= bandwidth[0]) & (frequencies <= bandwidth[1])
    else:
        fin = np.ones_like(frequencies, dtype=bool)
    frequencies = frequencies[fin]

    # Calculate spectra
    spectra_shape = len(stream), n_times, sum(fin)
    spectra = np.zeros(spectra_shape, dtype=complex)
    for trace_id, trace in enumerate(stream):
        tr = trace.data
        
        for time_id in range(n_times):
            start = time_id * step
            end = start + npts
            segment = tr[start:end] * window(npts)
            if np.nansum(segment)!=0:
                #print('fft_windL', len(segment))
                spectra[trace_id, time_id] = np.fft.fft(segment, **kwargs)[fin]

    # Times are extended with last time of traces
    t_end = stream[0].times(**times_kw)[-1]
    times = np.hstack((times, t_end))
    
    return times, frequencies, spectra


def xcov(wid, spectra_full, overlap, average):
    
    n_traces, n_windows, n_frequencies = spectra_full.shape
    beg = overlap * wid
    end = beg + average
    spectra = spectra_full[:, beg:end, :].copy()
    x = spectra[:, None, 0, :] * np.conj(spectra[:, 0, :])
    for swid in range(1, average):
        x += spectra[:, None, swid, :] * np.conj(spectra[:, swid, :])
    return x