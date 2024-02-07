"""
This script is a key part of the following publications:
    - Herig Coimbra, Pedro Henrique and Loubet, Benjamin and Laurent, Olivier and Mauder, Matthias and Heinesch, Bernard and 
    Bitton, Jonathan and Delpierre, Nicolas and Depuydt, Jérémie and Buysse, Pauline, Improvement of Co2 Flux Quality Through 
    Wavelet-Based Eddy Covariance: A New Method for Partitioning Respiration and Photosynthesis. 
    Available at SSRN: https://ssrn.com/abstract=4642939 or http://dx.doi.org/10.2139/ssrn.4642939

The main function is:  
- run_wt
    function: (1) gets data, (2) performs wavelet transform, (3) cross calculate variables, (4) averages by 30 minutes, (5) saves 
    call: run_wt()
    Input:
        a: 
    Return:
        b: 

- conditional_sampling
    function: split an array (n dimensions) into 4 arrays based on signal (+ or -) of itself and 2nd array 
    call: conditional_sampling()
    Input:
        args: arrays to be used as filter 
    Return:
        b: 

- universal_wt
    function: call any wavelet transform
    call: universal_wt()
    Input:
        a: 
    Return:
        b: 
"""

# standard modules
import os
import re
import warnings
import copy

# 3rd party modules
import pywt
import pycwt
import fcwt
from functools import reduce
import numpy as np
import pandas as pd
import itertools

# Project modules
from coimbra2023_scripts import *

def __cwt__(input, fs, f0, f1, fn, nthreads=1, scaling="log", fast=False, norm=True, Morlet=6.0):
    """
    function: performs Continuous Wavelet Transform
    call: __cwt__()
    Input:
        a: 
    Return:
        b: 
    """

    #check if input is array and not matrix
    if input.ndim > 1:
        raise ValueError("Input must be a vector")

    #check if input is single precision and change to single precision if not
    if input.dtype != 'single':
        input = input.astype('single')

    morl = fcwt.Morlet(Morlet) #use Morlet wavelet with a wavelet-parameter

    #Generate scales

    if scaling == "lin":
        scales = fcwt.Scales(morl,fcwt.FCWT_LINFREQS,fs,f0,f1,fn)
    elif scaling == "log":
        scales = fcwt.Scales(morl,fcwt.FCWT_LOGSCALES,fs,f0,f1,fn)
    else:
        scales = fcwt.Scales(morl,fcwt.FCWT_LOGSCALES,fs,f0,f1,fn)

    _fcwt = fcwt.FCWT(morl, int(nthreads), fast, norm)

    output = np.zeros((fn,input.size), dtype='csingle')
    freqs = np.zeros((fn), dtype='single')
    
    _fcwt.cwt(input,scales,output)
    scales.getFrequencies(freqs)

    return freqs, output


def __icwt__(W, sj, dt, dj, Cd=None, psi=None, wavelet=pycwt.wavelet.Morlet(6)):
    """
    function: performs Inverse Continuous Wavelet Transform
    call: __icwt__()
    Input:
        W: (cross-)spectra
        sj: scales
        dt: sampling rate
        dj: frequency resolution
        Cd, psi: wavelet-specific coefficients
        wavelet: mother wavelet (w/ cdelta and psi(0) callables). Ignored if Cd and psi are given.
    Return:
        x: array
    """
    if Cd is None: Cd = wavelet.cdelta
    if psi is None: psi = wavelet.psi(0)
        
    a, b = W.shape
    c = sj.size
    if a == c:
        sj_ = (np.ones([b, 1]) * sj).transpose()
    elif b == c:
        sj_ = np.ones([a, 1]) * sj
    
    x = (W.real / (sj_ ** .5)) * ((dj * dt ** .5) / (Cd * psi))
    return x

   
def __dwt__(*args, level=None, wave="db6"):
    """
    function: performs Discrete Wavelet Transform
    call: __dwt__()
    Input:
        *args: arrays (1D) to be transformed
        level: maximum scale (power of 2)
        wave: mother wavelet (comprehensible to pywt)
    Return:
        Ws: list of 2D arrays
    """
    Ws = []
    for X in args:
        Ws += [pywt.wavedec(X, wave, level=level)]
    level = len(Ws[-1])-1
    return Ws


def __idwt__(*args, N, level=None, wave="db6"):
    """
    function: performs Inverse Discrete Wavelet Transform
    call: __idwt__()
    Input:
        *args: 2D arrays contianing wavelet coefficient
        N: data lenght
        level: maximum scale (power of 2)
        wave: mother wavelet (comprehensible to pywt)
    Return:
        Ws: list of 2D arrays
        level: maximum scale (power of 2)
    """
    #assert sum([s==level for s in W.shape]), "Coefficients don't have the same size as level."
    def wrcoef(N, coef_type, coeffs, wavename, level):
        a, ds = coeffs[0], list(reversed(coeffs[1:]))

        if coef_type == 'a':
            return pywt.upcoef('a', a, wavename, level=level, take=N)  # [:N]
        elif coef_type == 'd':
            return pywt.upcoef('d', ds[level-1], wavename, level=level, take=N)  # [:N]
        else:
            raise ValueError("Invalid coefficient type: {}".format(coef_type))
    
    Ys = []
    for W in args:
        A1 = wrcoef(N, 'a', W, wave, level)
        D1 = [wrcoef(N, 'd', W, wave, i) for i in range(1, level+1)]
        Ys += [np.array(D1 + [A1])]
    return Ys, level


def universal_wt(signal, method, fs=20, f0=1/(3*60*60), f1=10, fn=100, 
                 dj=1/12, inv=True, **kwargs):
    """
    function: performs Continuous Wavelet Transform
    call: universal_wt()
    Input:
        signal: 1D array
        method: 'dwt', 'cwt', 'fcwt' (cwt but uses fast algorithm)
        fs: sampling rate (Hz)
        f0: highest scale (becomes level for DWT)
        f1: lowest scale (2x sampling rate)
        fn: number of scales (only used for CWT)
        dj: frequency resolution (only used for CWT)
        inv: . Default is True
        **kwargs: keyword arguments sent to wavelet transform and inverse functions 
    Return:
        wave: 2D array
        sj: scales 
    """
    assert method in [
        'dwt', 'cwt', 'fcwt'], "Method not found. Available methods are: dwt, cwt, fcwt"
    
    if method== "dwt":
        """Run Discrete Wavelet Transform"""
        lvl = kwargs.pop('level', int(np.ceil(np.log2(fs/f0))))
        # _l if s0*2^j; fs*2**(-_l) if Hz; (1/fs)*2**_l if sec.
        sj = [_l for _l in np.arange(1, lvl+2, 1)]
        waves = __dwt__(signal, level=lvl, **kwargs)
        if inv:
            N = np.array(signal).shape[-1]
            waves = __idwt__(*waves, N=N, level=lvl, **kwargs)
        wave = waves[0][0]
    
    elif method == 'fcwt':
        """Run Continuous Wavelet Transform, using fast algorithm"""
        _l, wave = __cwt__(signal, fs, f0, f1, fn, **kwargs)
        sj = np.log2(fs/_l)
        if inv:
            wave = __icwt__(wave, sj=sj, dt=fs, dj=dj, **kwargs, 
                        mother=pycwt.wavelet.Morlet(6))
    
    elif method == 'cwt':
        """Run Continuous Wavelet Transform"""
        wave, sj, _, _, _, _ = pycwt.cwt(
            signal, dt=1/fs, s0=2/fs, dj=dj, J=fn-1)
        sj = np.log2(sj*fs)
        if inv:
            wave = __icwt__(wave, sj=sj, dt=fs**-1, dj=dj, **kwargs)
    return wave, sj


def conditional_sampling(Y12, *args, names=['xy', 'a'], false=0):
    # guarantee names are enough to name all arguments
    nargs = len(args) + 1
    if nargs < len(names): names = names[:nargs]
    if nargs > len(names): names = names + ['b']* len(names)-nargs

    YS = [Y12] + list(args)
    Ys = {}
    label = {1: "+", -1: "-"}

    # run for all combinations of + and - for groups of size n
    # (e.g., n=2: ++, +-, -+, --, n=3 : +++, ++-, ...)
    for co in set(itertools.combinations([1, -1]*nargs, nargs)):
        sign = ''.join([label[c] for c in co])
        name = names[0] + sign[:2] + names[1] + ''.join([s + names[2+i]  for i, s in enumerate(sign[2:])])
        Ys[name] = Y12
        for i, c in enumerate(co):
            xy = 1 * (c*YS[i] >= 0)
            #xy[xy==0] = false
            xy = np.where(xy == 0, false, xy)
            Ys[name] = Ys[name] * xy
    return Ys


def run_wt(ymd, varstorun, raw_kwargs, output_path, wt_kwargs={}, 
           method="dwt", Cφ=1, nan_tolerance=.3,
           averaging=[30], condsamp=[], integrating=30*60, 
           overwrite=False, saveraw=False, file_duration="1D", verbosity=1):
    """
    fs = 20, f0 = 1/(3*60*60), f1 = 10, fn = 100, agg_avg = 1, 
    suffix = "", mother = pycwt.wavelet.MexicanHat(),
    **kwargs):
    """
    assert method in [
        'dwt', 'cwt', 'fcwt'], "Method not found. Available methods are: dwt, cwt, fcwt"
    if verbosity: print(f'\nRUNNING WAVELET TRASNFORM ({method})\n')
    if method in ["cwt", "fcwt"]:
        if method == "fcwt" or "mother" not in wt_kwargs.keys() or wt_kwargs.get("mother") in ['morlet', 'Morlet', pycwt.wavelet.Morlet(6)]:
            Cφ = 5.271
        else:
            Cφ = 16.568
    
    dt = 1 / wt_kwargs.get("fs", 20)
    suffix = raw_kwargs['suffix'] if 'suffix' in raw_kwargs.keys() else ''
    
    _, _, _f = ymd
    ymd = list_time_in_period(*ymd, file_duration)
    if method in ['dwt']:
        buffer = bufferforfrequency_dwt(
            N=pd.to_timedelta(file_duration)/pd.to_timedelta("1S") * dt**-1,
            n_=_f, **wt_kwargs)/2
    else:
        buffer = bufferforfrequency(wt_kwargs.get("f0", 1/(3*60*60))) / 2


    for i, yl in enumerate(ymd):
        date = re.sub('[-: ]', '', yl.strftime('%')[0])
        if file_duration.endswith("D"): date = date[:8]
        if file_duration.endswith("H") or file_duration.endswith("Min"): date = date[:12]
        
        # recheck if files exist and overwrite option
        # doesn't save time (maybe only save 5min)
        if not overwrite:
            avg_ = []
            for a in averaging:
                if not overwrite and os.path.exists(output_path.format(suffix, date, str(a).zfill(2))):
                    avg_ += [a]
            avg_ = list(set(averaging)-set(avg_))
            if not avg_:
                if verbosity > 1: warnings.warn("exists: File already exists ({}).".format(date))
                continue
        else:
            avg_ = [a for a in averaging]
        
        curoutpath_inprog = output_path.format(suffix, str(date), "").rsplit(".", 1)[
            0] + ".inprogress"
        if checkifinprogress(curoutpath_inprog): continue
        
        # load files
        # data = get_rawdata.open_flux(lookup=yl, **raw_kwargs).data
        data = loaddatawithbuffer(
            yl, d1=None, freq=_f, buffer=buffer, **raw_kwargs)
        if data.empty:
            if verbosity>1: warnings.warn("exit1: No file was found ({}).".format(date))
            if os.path.exists(curoutpath_inprog): os.remove(curoutpath_inprog)
            continue
        
        # ensure time is time
        data.TIMESTAMP = pd.to_datetime(data.TIMESTAMP)
        
        # ensure continuity
        data = pd.merge(pd.DataFrame({"TIMESTAMP": pd.date_range(*nanminmax(data.TIMESTAMP), freq=f"{dt}S")}),
                            data, on="TIMESTAMP", how='outer').reset_index(drop=True)

        # main run
        # . collect all wavelet transforms
        # . calculate covariance
        # . conditional sampling (optional)
        # . save in dataframe and .csv
        φ = {}
        μ = {}
        dat_fullspectra = {a: [] for a in avg_}
        dat_fluxresult = {a: [] for a in avg_}

        # run by couple of variables (e.g.: co2*w -> mean(co2'w'))
        for xy, condsamp in [(v.split('*')[:2], v.split('*')[2:]) for v in varstorun]:
            # run wavelet transform
            for v in xy + condsamp:
                if v not in φ.keys():
                    signal = np.array(data[v])
                    signan = np.isnan(signal)
                    N = len(signal)
                    Nnan = np.sum(signan)
                    if Nnan:
                        if (nan_tolerance > 1 and Nnan > nan_tolerance) or (Nnan > nan_tolerance * N):
                            warnings.warn(
                                f"Too much nans ({np.sum(signan)}, {np.sum(signan)/len(signal)}%) in {date}.")
                    if Nnan and Nnan < N:
                        signal = np.interp(np.linspace(0, 1, N), 
                                  np.linspace(0, 1, N)[signan == False],
                                  signal[signan==False])
                    φ[v], sj = universal_wt(signal, method, **wt_kwargs, inv=True)
                    # apply despiking (Mauder et al.)

                    def __despike__(X):
                        N = len(X)
                        X = mauder2013(X)
                        Xna = np.isnan(X)
                        try:
                            X = np.interp(np.linspace(0, 1, N), 
                                            np.linspace(0, 1, N)[Xna == False],
                                    X[Xna==False])
                        except Exception as e:
                            warnings.warn(str(e))
                        return X 
                    φ[v] = np.apply_along_axis(__despike__, 1, φ[v])
                    μ[v] = signan *1

            # calculate covariance
            Y12 = np.array(φ[xy[0]]) * np.array(φ[xy[1]]).conjugate() * Cφ
            print(date, ''.join(xy), Y12.shape, round(Y12.shape[1] / (24*60*60*20), 2), buffer)
            φs = {''.join(xy): Y12}
            μs = {''.join(xy): np.where(np.where(
                np.array(μ[xy[0]]), 0, 1) * np.where(np.array(μ[xy[1]]), 0, 1), 0, 1)}

            # conditional sampling
            names = [''.join(xy)] + [xy[0]+c for c in condsamp]
            φc = [np.array(φ[xy[0]]) * np.array(φ[c]).conjugate() for c in condsamp]
            φc = conditional_sampling(Y12, *φc, names=names) if φc else {}
            #φs.update({k.replace("xy", ''.join(xy)).replace('a', ''.join(condsamp)): v for k, v in φc.items()})
            φs.update(φc)

            # repeats nan flag wo/ considering conditional sampling variables
            μs.update(
                {k: μs[k if k in μs.keys() else [k_ for k_ in μs.keys() if k.startswith(k_)][0]] for k in φs.keys()})

            # array to dataframe for averaging
            def __arr2dataframe__(Y, qc=np.nan, prefix=''.join(xy), 
                                  id=np.array(data.TIMESTAMP), icolnames=sj):
                colnames = ["{}_{}".format(prefix, l) for l in icolnames] if icolnames is not None else None
                __temp__ = matrixtotimetable(id, Y, columns=colnames)
                __temp__["{}_qc".format(prefix)] = qc
                __temp__ = __temp__[__temp__.TIMESTAMP > min(yl)]
                __temp__ = __temp__[__temp__.TIMESTAMP <= max(yl)]
                return __temp__

            __temp__ = reduce(lambda left, right: pd.merge(left, right[['TIMESTAMP'] + list(right.columns.difference(left.columns))], on="TIMESTAMP", how="outer"),
                              [__arr2dataframe__(Y, μs[n], prefix=n) for n, Y in φs.items()])
            
            for a in avg_:
                __tempa__ = copy.deepcopy(__temp__)
                __tempa__["TIMESTAMP"] = pd.to_datetime(np.array(__tempa__.TIMESTAMP)).ceil(
                    str(a)+'Min')
                __tempa__ = __tempa__.groupby("TIMESTAMP").agg(np.nanmean).reset_index()

                maincols = ["TIMESTAMP", ''.join(xy)]
                if φc:
                    #maincols += [names[0] + c + names[1] for c in ['++', '+-', '--', '-+']]
                    sign_label = {1: "+", -1: "-"}
                    for co in set(itertools.combinations([1, -1]*len(names), len(names))):
                        #for c in ['++', '+-', '--', '-+']:
                        sign = ''.join([sign_label[c] for c in co])
                        name = names[0] + sign[:2] + names[1] + ''.join([s + names[2+i]  for i, s in enumerate(sign[2:])])
                        __tempa__.insert(1, name, np.sum(__tempa__[[
                            f"{name}_{l}" for l in sj if dt*2**l < integrating]], axis=1))
                __tempa__.insert(1, ''.join(xy), np.sum(__tempa__[[
                    "{}_{}".format(''.join(xy), l) for l in sj if dt*2**l < integrating]], axis=1))

                dat_fullspectra[a] += [__tempa__]
                dat_fluxresult[a] += [__tempa__[maincols]]
                del __tempa__
        
        for a in avg_:
            dat_fullspectra[a] = reduce(lambda left, right: pd.merge(left, right, on="TIMESTAMP", how="outer"),
                                 dat_fullspectra[a])
            dat_fluxresult[a] = reduce(lambda left, right: pd.merge(left, right, on="TIMESTAMP", how="outer"),
                                 dat_fluxresult[a])
            
            mkdirs(output_path.format(suffix, str(date), str(a).zfill(2)))
            dat_fullspectra[a].to_csv(output_path.format(
                suffix + "_full_cospectra", str(date), str(a).zfill(2)), index=False)
            dat_fluxresult[a].to_csv(output_path.format(
                suffix, str(date), str(a).zfill(2)), index=False)
                
        if os.path.exists(curoutpath_inprog): os.remove(curoutpath_inprog)
        if verbosity:
            print(date, len(yl), f'{int(100*i/len(ymd))} %', end='\n')
    return
