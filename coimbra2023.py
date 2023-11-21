"""
This script is a key part of the following publications:
    - https://hal.science/hal-04272798
"""

##########################################
###     IMPORTS                           
##########################################

# standard modules
import copy
import os
import re
import warnings
import time
import datetime
# 3rd party modules
import holidays
import yaml
import numpy as np
from functools import reduce
from sklearn.linear_model import LinearRegression
import zipfile
from io import StringIO
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype
pd.DataFrame.columnstartswith = lambda self, x: [c for c in self.columns if c.startswith(x)]
pd.DataFrame.columnsmatch = lambda self, x: [c for c in self.columns if re.findall(x, c)]
import pywt
import pycwt
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage as RPackage
# project modules
r2gapfill = RPackage(''.join(open('coimbra2024_gapfilling.R', 'r').readlines()), 'MDS_gapfill_and_partition')
routine_partition = RPackage(''.join(open('coimbra2024_gapfilling.R', 'r').readlines()), 'routine_partition')

##########################################
###     PROJECT CHOICES                           
##########################################

SITES_TO_STUDY = ['FR-Fon', 'FR-Gri']

##########################################
###     FUNCTIONS                           
##########################################

class structuredData:
    def __init__(self, **kwargs):
        for k, v in kwargs.items(): self.__dict__[k]=v
        pass

month2season = lambda month: {1:1, 2:1, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3, 9:4, 10:4, 11:4, 12:1}[month]
isweekday = lambda date: (pd.to_datetime(date).weekday()<5) * (pd.to_datetime(date) not in holidays.FR())

def partitionDWCS(data, GPP='DW_GPP', Reco='DW_Reco', NEE='DW_NEE_uStar_f', positive='wco2-+wh2o_uStar_f', negative='wco2--wh2o_uStar_f', PAR='PPFD_IN'):
    #lightresponse = lambda p: np.where(np.isnan(p), 1, (p-np.nanmin(p))/(np.nanmax(p)-np.nanmin(p)))
    #data["DW_GPP_withPARratio"] = np.where((np.isnan(data.PPFD)==False) * (data.PPFD<10), 0, 1) * (
    #    data["dwt_wco2-+h2o_uStar_f"] + lightresponse(data["PPFD"]) * (data["dwt_wco2--h2o_uStar_f"]))
    #data["DW_Reco_withPARratio"] = (data["DWT_NEE_uStar_f"] - data["DW_GPP_withPARratio"])
    
    islight = np.where((np.isnan(data[PAR]) == False) * (data[PAR] < 10), 0, 1)
    data[GPP] = islight * (data[positive] + 0.5*data[negative])
    data[Reco] = (data[NEE] - data[GPP])
    return data

def summarisestats(X, y):
    statisticsToReturn = structuredData()
    X = np.array(X)
    y = np.array(y)
    NaN = np.isnan(X) + np.isnan(y)
    X = X[NaN==0]
    y = y[NaN==0]
    statisticsToReturn.me = np.nanmean((X-y)).round(2)
    statisticsToReturn.mae = np.nanmean(abs(X-y)).round(2)
    regression = LinearRegression(fit_intercept=False)
    regression.fit(X.reshape(-1, 1), y.reshape(-1, 1))
    statisticsToReturn.m = regression.coef_[0][0]
    b = regression.intercept_
    statisticsToReturn.b = b
    b_ = np.round(b[0], 2) if b else 0
    b_ = "+" + str(b_) if b_ >= 0 else str(b_)
    statisticsToReturn.r2 = regression.score(X.reshape(-1, 1), y.reshape(-1, 1))
    return statisticsToReturn

def summarisestatslabel(meta, xn, yn):    
    stat_label = f"R²: {np.round(meta.r2, 2)}"
    stat_label = stat_label + f"\nME: {np.round(meta.me, 2)}"
    stat_label = stat_label + f"\nMAE: {np.round(meta.mae, 2)}"
    stat_label = stat_label + f"\n{yn}={np.round(meta.m, 2)} {xn}"
    return stat_label
    
def summarisestatstext(meta, xn='x', yn='y'):    
    stat_label = f"R²= {np.round(meta.r2, 2)}"
    stat_label = stat_label + f", ME= {np.round(meta.me, 2)} µmol m-2 s-1"
    stat_label = stat_label + f", MAE= {np.round(meta.mae, 2)} µmol m-2 s-1"
    stat_label = stat_label + f", {yn}={np.round(meta.m, 2)}×{xn} linear fit"
    return stat_label

def get_r2(X, y):
    if len(X)==0:
        return 0
    X = np.array(X).ravel()
    y = np.array(y).ravel()
    finite = np.isfinite(X*y)
    X = X[finite].reshape(-1, 1)
    y = y[finite].reshape(-1, 1)
    regression = LinearRegression(fit_intercept=True)
    regression.fit(X, y)
    r2 = regression.score(X, y)
    return r2

def get_all_sites(fc, *args, **kwargs):
    datasetToReturn = structuredData()
    datasetToReturn.data = {}
    for sitename in SITES_TO_STUDY:
        data = fc(*args, sitename=sitename, **kwargs)
        if data is None: data = pd.DataFrame({"TIMESTAMP": []})
        datasetToReturn.data[sitename] = data
    datasetToReturn.alldata = []
    for k, v in datasetToReturn.data.items():
        v.insert(0, 'co_site', k)
        datasetToReturn.alldata += [copy.deepcopy(v).reset_index(drop=True)]
    datasetToReturn.alldata = pd.concat(
        datasetToReturn.alldata).reset_index(drop=True)
    return datasetToReturn.alldata

def get_cospectra(sitename=None, mergeorconcat='merge'):
    if sitename is None:
        return get_all_sites(get_cospectra)

    wv_path = f'data/{sitename}/'
    data = []
    for name in os.listdir(wv_path):
        if name.endswith('.csv'):
            if re.findall('_full_cospectra', name) and re.findall('.30mn', name):
                data.append(pd.read_csv(wv_path+name, na_values=[-9999, 'NAN']))
    
    if len(data) == 0:
        return None
    elif len(data) == 1:
        data = data[0]
    elif mergeorconcat == 'concat':
        data = pd.concat(data)
    else:
        data = reduce(lambda left, right: pd.merge(left, right, on=['TIMESTAMP'], how="outer", suffixes=('', '_DUP')),
                        data)
    
    for tc in data.columnstartswith('TIMESTAMP'):
        data[tc] = pd.to_datetime(
            data[tc])
    return data
    
def get_metadata(sitename=None):
    if sitename is None:
        return get_all_sites(get_metadata)
    
    if os.path.exists(f'data/{sitename}/{sitename}_metadata.yaml'):
        meta = yaml.safe_load(open(f'data/{sitename}/{sitename}_metadata.yaml', 'r'))
        return pd.DataFrame(meta, index=[0])
    return None


def get_eddypro_output(sitename=None, mergeorconcat='concat'):
    if sitename is None:
        return get_all_sites(get_eddypro_output)
    
    ep_read_params = {'skiprows': [0,2], 'na_values': [-9999, 'NAN']}
    
    ep_path = f'data/{sitename}/eddypro_output/'
    files = {'FLUX': [], 'QCQA': [], 'META': []}
    for name in os.listdir(ep_path):
        if name.endswith('.csv'):
            if re.findall('_full_output_', name):
                if name.endswith('_adv.csv'):
                    files['FLUX'] += [pd.read_csv(ep_path+name, **ep_read_params)]
                else:
                    files['FLUX'] += [pd.read_csv(ep_path+name, na_values=[-9999, 'NAN'])]
            elif re.findall('_qc_details_', name):
                if name.endswith('_adv.csv'):
                    files['QCQA'] += [pd.read_csv(ep_path+name, **ep_read_params)]
                else:
                    files['QCQA'] += [pd.read_csv(ep_path+name, na_values=[-9999, 'NAN'])]
            elif re.findall('_metadata_', name):
                files['META'] += [pd.read_csv(ep_path+name, na_values=[-9999, 'NAN'])]
    
    for k in [k for k in list(files.keys()) if not files[k]]:
        del([files[k]])

    for k in files.keys():
        if len(files[k]) == 1:
            files[k] = files[k][0]
        elif mergeorconcat == 'concat':
            files[k] = pd.concat(files[k])
        else:
            files[k] = reduce(lambda left, right: pd.merge(
                left, right, on=["date", 'time'], how="outer", suffixes=('', '_DUP')), files[k])
    
    data = pd.DataFrame(files.pop('FLUX', {}))
    for name, dat in files.items():
        data = pd.merge(data, dat, on=["date", 'time'], how="outer", suffixes=('', f'_{name}'))
    
    data['TIMESTAMP'] = pd.to_datetime(data.date + ' ' + data.time)#.dt.tz_localize('UTC')
    
    data['Vd'] = (data.air_molar_volume * data.air_pressure /
                           (data.air_pressure - data.e))
    return data

def get_biomet(sitename=None):
    if sitename is None:
        return get_all_sites(get_biomet)
    
    D1 = pd.read_csv(os.path.join(
        'data', sitename, f"ICOSETC_{sitename}_METEO_L2.csv"), na_values=["NA", -9999.])
    
    # ICOS processed L2
    for tc in D1.columnstartswith('TIMESTAMP'):
        D1[tc] = pd.to_datetime(D1[tc], format='%Y%m%d%H%M')
    D1["TIMESTAMP"] = D1.TIMESTAMP_END

    # EddyPro
    D2 = pd.read_csv(os.path.join(
        'data', sitename, f"{sitename}_full_biomet.30mn.csv"), na_values=["NA", -9999.])
    for tc in D2.columnstartswith('TIMESTAMP'):
        D2[tc] = pd.to_datetime(D2[tc])
    
    data = pd.merge(D1, D2, on='TIMESTAMP', how='outer', suffixes=('_L2', ''))
    return data

def get_gapfilling(sitename=None, only_f_columns=True):
    if sitename is None:
        return get_all_sites(get_gapfilling)
    
    data = pd.read_csv(os.path.join('data', sitename,
                                    f"{sitename}_flux_MDS_gapfilled.csv"))
    data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'])
    if only_f_columns:
        data = data[['TIMESTAMP'] + data.columnsmatch('(_uStar)?_f$')]
    return data
    
    data = pd.DataFrame({'TIMESTAMP': []})
    # include partitioning methods using different NEE inputs
    for step in ["_f", "_orig"]:
        for inputnee in ["EP" ,"DW"]:
            gapfillingDatasets = pd.read_csv(os.path.join('data', sitename,
                                    f"{sitename}_full_gapfill_MRF_{inputnee}.30mn.csv"), na_values=["NA"])
            #print(gapfillingDatasets)
            gapfillingDatasets["TIMESTAMP"] = pd.to_datetime(gapfillingDatasets.TIMESTAMP)
            gapfillingDatasets["TIMESTAMP"] = gapfillingDatasets.TIMESTAMP.dt.tz_localize(None)
            gapfillingDatasets = gapfillingDatasets[["TIMESTAMP"] + [c for c in gapfillingDatasets.columns if (c.startswith('NEE')) and (c.endswith(step))]]
            gapfillingDatasets.rename(columns={v: f"{inputnee}_{v.split('_', 1)[0]}_MDS_{v.split('_', 1)[1]}" for v in gapfillingDatasets.columns if v not in ["TIMESTAMP"]}, 
            inplace=True)
            
            for tc in gapfillingDatasets.columnstartswith('TIMESTAMP'):
                gapfillingDatasets[tc] = pd.to_datetime(gapfillingDatasets[tc])
            data = data.merge(gapfillingDatasets, on='TIMESTAMP', how='outer')
    return data

def get_partitioning(sitename=None):
    if sitename is None:
        return get_all_sites(get_partitioning)
    
    data = pd.DataFrame({'TIMESTAMP': []})
    # include partitioning methods using different NEE inputs
    for method in ["MRF", "TKF", "GLF"]:
        for inputnee in ["EP" ,"DW"]:
            gapfillingDatasets = pd.read_csv(os.path.join('data', sitename,
                                    f"{sitename}_full_gapfill_{method}_{inputnee}.30mn.csv"), na_values=["NA"])
            #print(gapfillingDatasets)
            gapfillingDatasets["TIMESTAMP"] = pd.to_datetime(gapfillingDatasets.TIMESTAMP)
            gapfillingDatasets["TIMESTAMP"] = gapfillingDatasets.TIMESTAMP.dt.tz_localize(None)
            gapfillingDatasets = gapfillingDatasets[["TIMESTAMP"] + [c for c in gapfillingDatasets.columns if (c.startswith('GPP') or c.startswith('Reco')) and (c.endswith('uStar') or c.endswith('uStar_f'))]]
            gapfillingDatasets.rename(columns={v: f"{inputnee}_{v.split('_', 1)[0]}_{method}_uStar" for v in gapfillingDatasets.columns if v not in ["TIMESTAMP"]}, #["GPP_DT_uStar", "Reco_DT_uStar", "GPP_uStar_f", "Reco_uStar"]}, 
            inplace=True)
            # drop absurd estimations (>1e3)
            for c in gapfillingDatasets.columns:
                if (c.startswith(f'{inputnee}_GPP') or c.startswith(f'{inputnee}_Reco')) and (c.endswith('uStar')):
                    gapfillingDatasets[c] = np.where(abs(gapfillingDatasets[c])>1e3, np.nan, gapfillingDatasets[c])
            
            for tc in gapfillingDatasets.columnstartswith('TIMESTAMP'):
                gapfillingDatasets[tc] = pd.to_datetime(gapfillingDatasets[tc])
            data = data.merge(gapfillingDatasets, on='TIMESTAMP', how='outer')
    return data

def get_expected_cospectra(data, source='horst97', samp_rate=10):
    zL = np.array(data['(z-d)/L'])
    u_ = np.array(data['u_rot'])
    z = np.array(data['zm'] - data['zd'])
    nm = np.where(zL > 0, 2.0-1.915/(1+0.5*zL), 0.085)
    fm = nm*u_/z
    data['fm'] = fm
    sj2n = [[((samp_rate/2)/2**i)*y/x for i in range(17)] for x, y in list(zip(u_, z))]
    
    horst97 = lambda _n_, _nm_, _zL_: np.where(_zL_>0, (0.637*(_n_/_nm_))/(1+0.91*(_n_/_nm_)**2.1),
        np.where(_n_<=1, (1.05*_n_/_nm_)/(1+1.33*_n_/_nm_)**(7/4), (0.387*(_n_/_nm_))/(1+0.38*_n_/_nm_)**(7/3)))
    kaimal72 = lambda _n_: np.where(_n_<=1, (11*_n_)/(1+13.3*_n_)**(7/4), (4*_n_)/(1+3.8*_n_)**(7/3))

    if source=='horst97':
        horst97_cospectra = np.array([[horst97(i, nm_, zL_) for i in nl_] for nm_, nl_, zL_ in list(zip(nm, sj2n, zL))])
        #horst97_cospectra1 = (horst97_cospectra.T * abs(np.array(np.sum(chunkdf_[['wco2++wh2o', 'wco2+-wh2o']], axis=1)))).T
        horst97_cospectra = (horst97_cospectra.T * abs(np.array(data['wco2']))).T
        return horst97_cospectra
    elif source=='kaimal72':
        kaimal72_cospectra = np.array([[kaimal72(i) for i in nl_] for nl_ in sj2n])
        kaimal72_cospectra = (kaimal72_cospectra.T * abs(np.array(data['wco2']))).T
        return kaimal72_cospectra

    return horst97_cospectra


def j2sj(e, samp_rate=10): return 1/((samp_rate/2)*(2**-float(e)))

def get_dic_flux_data(data): return {k: copy.deepcopy(data.query(
    f"co_site == '{k}'").reset_index(drop=True)) for k in data.co_site.unique()}

##########################################
###     GENERIC FUNCTIONS                           
##########################################

def matrixtotimetable(time, mat, c0name="TIMESTAMP", **kwargs):
    assert len(time) in mat.shape, f"Time ({time.shape}) and matrix ({mat.shape}) do not match."
    mat = np.array(mat)

    if len(time) != mat.shape[0] and len(time) == mat.shape[1]:
        mat = mat.T

    __temp__ = pd.DataFrame(mat, **kwargs)
    __temp__.insert(0, c0name, time)

    return __temp__


def yaml_to_dict(path):
    with open(path, 'r+') as file:
        file = yaml.safe_load(file)
    return file


def list_time_in_period(tmin, tmax, fastfreq, slowfreq, include='both'):
    if include=="left":
        return [(pd.date_range(max(p, pd.to_datetime(tmin)), min(p + pd.Timedelta(slowfreq), pd.to_datetime(tmax)),
          freq=fastfreq)[:-1]) for p in pd.date_range(tmin, tmax, freq=slowfreq).floor(slowfreq)]
    elif include == "right":
        return [(pd.date_range(max(p, pd.to_datetime(tmin)), min(p + pd.Timedelta(slowfreq), pd.to_datetime(tmax)),
          freq=fastfreq)[1:]) for p in pd.date_range(tmin, tmax, freq=slowfreq).floor(slowfreq)]
    elif include == "both":
        return [(pd.date_range(max(p, pd.to_datetime(tmin)), min(p + pd.Timedelta(slowfreq), pd.to_datetime(tmax)),
          freq=fastfreq)) for p in pd.date_range(tmin, tmax, freq=slowfreq).floor(slowfreq)]
    return


def checkifinprogress(path, LIMIT_TIME_OUT=30*60):
    if os.path.exists(path) and (time.time()-os.path.getmtime(path)) < LIMIT_TIME_OUT:
        return 1
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a+"):
            pass
        return 0


def nanminmax(x):
    return [np.nanmin(x), np.nanmax(x)]


def mkdirs(filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    
def nearest(items, pivot, direction=0):
    if direction == 0:
        nearest = min(items, key=lambda x: abs(x - pivot))
        difference = abs(nearest - pivot)
        
    elif direction == -1:
        nearest = min(items, key=lambda x: abs(x - pivot) if x<pivot else pd.Timedelta(999, "d"))
        difference = (nearest - pivot)
        
    elif direction == 1:
        nearest = min(items, key=lambda x: abs(x - pivot) if x>pivot else pd.Timedelta(999, "d"))
        difference = (nearest - pivot)
    return nearest, difference


def update_nested_dict(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = update_nested_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def update_nested_dicts(*ds, fstr=None):
    r = {}
    for d in ds:
        if isinstance(d, str) and fstr:
            try:
                d = fstr(d)
            except Exception as e:
                continue
        r = update_nested_dict(r, d)
    return r

from functools import reduce

def concat_into_single_file(path, pattern, output_path=None, **kwargs):
    print('\nCONSOLIDATING DATASET\n')
    if output_path is None: output_path = os.path.join(path, 'concat_into_single_file') 
    
    files_to_concat = []
    for name in os.listdir(path):
        if re.findall(pattern, name):
            files_to_concat.append(os.path.join(path, name))
    
    files_to_concat = [pd.read_csv(f, **kwargs) for f in files_to_concat]
    data = reduce(lambda left, right: pd.concat([left, right]), files_to_concat)
    
    mkdirs(output_path)
    data.to_csv(output_path, index=False)
    print(os.path.basename(output_path), ': Saved.', ' '*15, end='\n', sep='')
    
    return

##########################################
###     UNIVERSAL READER                            
##########################################

DEFAULT_FILE_RAW = {
    'file_pattern': '.*_raw_dataset_([0-9]{12}).csv', 
    'date_format': '%Y%m%d%H%M', 
    'dt': 0.05, 
    'tname': "TIMESTAMP", 
    'id': None,
    'datefomatfrom': '%Y%m%d%H%M%S.%f', 
    'datefomatto': '%Y-%m-%dT%H:%M:%S.%f'
}

DEFAULT_READ_CSV = {
}

DEFAULT_READ_GHG = {
    'skiprows': 7,
    'sep': r"\t"
}

DEFAULT_FMT_DATA = {
}


class structuredDataFrame:
    def __init__(self, data=None, dt=None, **kwargs):
        if data is None:
            loopvar = kwargs.pop('lookup', [])
            loopvar = [l.to_list() if isinstance(l, list)==False else l for l in loopvar]
            for l in loopvar:
                result = universal_reader(lookup=l, **kwargs, verbosity=0)
                self.__dict__.update(result.__dict__)
        
        else:
            assert dt is not None, 'Missing measurement frequency (dt).'
            self.data = data
            self.dt = dt
            self.__dict__.update(**kwargs)
    
    def filter(self, items: dict):
        for k, v in items.items():
            if isinstance(v, tuple):
                self.data = self.data.loc[(self.data[k] > v[0])
                                          & (self.data[k] < v[1])].copy()
            else:
                self.data = self.data[self.data[k].isin(v)].copy()
        return self

    def rename(self, names: dict):
        self.data = self.data.rename(columns=names)
        return self

    def modify(self, items: dict):
        for k, v in items.items():
            self.data[k] = v
        return self

    def format(self, 
               cols={'t':'ts'}, 
               keepcols=['u', 'v', 'w', 'ts', 'co2', 'co2_dry', 'h2o', 'h2o_dry', 'ch4', 'n2o'],
               addkeep=[],
               colsfunc=str.lower, cut=False, **kwargs):
        
        if isinstance(self, pd.DataFrame):
            formated = self
        else:
            fmt_clas = structuredDataFrame(**self.__dict__)
            formated = fmt_clas.data

        if colsfunc is not None:
            formated.columns = map(colsfunc, formated.columns)
        #cols.update(kwargs)
        cols.update({v.lower(): k.lower() for k, v in kwargs.items() if isinstance(v, list)==False})
        cols = {v: k for k, v in {v: k for k, v in cols.items()}.items()}
        cols.update({'timestamp': 'TIMESTAMP'})
        #formated.TIMESTAMP = formated.TIMESTAMP.apply(np.datetime64)
        if cut:
            #formated = formated[[
            #    c for c in formated.columns if c in cols.keys()]]
            formated = formated.loc[:, np.isin(formated.columns, keepcols+addkeep+list(cols.keys()))]
        
        formated = formated.rename(columns=cols)

        if isinstance(self, pd.DataFrame):
            return formated
        else:
            fmt_clas.data = formated
            return fmt_clas
    
    def interpolate(self, cols=["co2", "w"], qcname="qc"):
        interpolated = structuredDataFrame(**self.__dict__)
        interpolated.data[qcname] = 0
        for c_ in list(cols):
            interpolated.data[qcname] = interpolated.data[qcname] + 0 * \
                np.array(interpolated.data[c_])
            interpolated.data.loc[np.isnan(interpolated.data[qcname]), qcname] = 1
            interpolated.data[qcname] = interpolated.data[qcname].astype(int)
            interpolated.data[c_] = interpolated.data[c_].interpolate(method='pad')
            
        return interpolated


def universal_reader(path, lookup=[], fill=False, fmt={}, onlynumeric=True, verbosity=1, fkwargs={}, tipfile="readme.txt", **kwargs):
    df_site = pd.DataFrame()
    
    folders = [path + p + '/' for p in os.listdir(path) if os.path.isdir(path + p)]
    folders = folders if folders else [path]
    
    for path_ in folders:
        df_td = pd.DataFrame()

        # read tips file
        kw_ = update_nested_dicts({"FILE_RAW": DEFAULT_FILE_RAW, "READ_CSV": DEFAULT_READ_CSV, "FMT_DATA": DEFAULT_FMT_DATA}, 
                                  os.path.join(path, tipfile), os.path.join(path_, tipfile),
                                  {"FILE_RAW": fkwargs, "READ_CSV": kwargs, "FMT_DATA": fmt},
                                  fstr=lambda d: yaml_to_dict(d))
        
        kw = structuredData(**kw_['FILE_RAW'])
        kw_csv = kw_['READ_CSV']
        
        try:
            if ('header_file' in kw_csv.keys()) and (os.path.exists(kw_csv['header_file'])):
                kw_csv['header_file'] = "[" + open(kw_csv['header_file']).readlines()[0].replace("\n", "") + "]"
        except:
            None
        
        lookup_ = list(set([f.strftime(kw.date_format) for f in lookup]))
        files_list = {}

        for root, directories, files in os.walk(path_):
            for name in files:
                dateparts = re.findall(kw.file_pattern, name, flags=re.IGNORECASE)
                if len(dateparts) == 1:
                    files_list[dateparts[0]] = os.path.join(root, name)
                    
        for td in set(lookup_) & files_list.keys() if lookup_ != [] else files_list.keys():
            path_to_tdfile = files_list[td]
            if os.path.exists(path_to_tdfile):
                if path_to_tdfile.endswith('.gz'): kw_csv.update(**{'compression': 'gzip'})
                elif path_to_tdfile.endswith('.csv'): kw_csv.pop('compression', None)
                if path_to_tdfile.endswith('.ghg'):
                    with zipfile.ZipFile(path_to_tdfile, 'r') as zip_ref:
                        datafile = [zip_ref.read(name) for name in zip_ref.namelist() if name.endswith(".data")][0]
                    datafile = str(datafile, 'utf-8')
                    path_to_tdfile = StringIO(datafile)
                    # DEFAULT_READ_GHG
                    kw_csv.update(DEFAULT_READ_GHG)
                try:
                    df_td = pd.read_csv(path_to_tdfile, **kw_csv)
                except Exception as e:
                    # (EOFError, pd.errors.ParserError, pd.errors.EmptyDataError):
                    try:
                        if verbosity>1: warnings.warn(f'{e}, when opening {path_to_tdfile}, using {kw_csv}. Re-trying using python as engine and ignoring bad lines.')
                        df_td = pd.read_csv(path_to_tdfile, on_bad_lines='warn', engine='python', **kw_csv)
                    except Exception as ee:
                        warnings.warn(f'{ee}, when opening {str(path_to_tdfile)}, using {kw_csv}')
                        continue
                
                """
                if kw.tname in df_td.columns:
                    try:
                        df_td.loc[:, kw.tname] = pd.to_datetime(df_td.loc[:, kw.tname].astype(str))
                        print(max(df_td[kw.tname].dt.year), max(df_td[kw.tname]), min(df_td[kw.tname].dt.year))
                        assert max(df_td[kw.tname].dt.year) > 1990 and min(df_td[kw.tname].dt.year) > 1990
                    except:
                        df_td.rename({kw.tname+'_orig': kw.tname})
                """
                if kw.datefomatfrom == 'drop':
                    df_td = df_td.rename({kw.tname: kw.tname+'_orig'})
                
                if kw.tname not in df_td.columns or kw.datefomatfrom == 'drop':
                    if "date" in df_td.columns and "time" in df_td.columns:
                        df_td[kw.tname] = pd.to_datetime(
                            df_td.date + " " + df_td.time, format='%Y-%m-%d %H:%M')
                    else:
                        df_td[kw.tname] = pd.to_datetime(
                            td, format=kw.date_format) - datetime.timedelta(seconds=kw.dt) * (len(df_td)-1 + -1*df_td.index)
                            #td, format=kw.date_format) + datetime.timedelta(seconds=kw.dt) * (df_td.index)
                        df_td[kw.tname] = df_td[kw.tname].dt.strftime(
                            kw.datefomatto)
                else:
                    try:
                        if is_numeric_dtype(df_td[kw.tname]):
                            df_td.loc[:, kw.tname] = df_td.loc[:, kw.tname].apply(lambda e: pd.to_datetime('%.2f' % e, format=kw.datefomatfrom).strftime(kw.datefomatto))
                        elif is_object_dtype(df_td[kw.tname]):
                            df_td.loc[:, kw.tname] = df_td.loc[:, kw.tname].apply(lambda e: pd.to_datetime(e).strftime(kw.datefomatto))
                        else:
                            df_td.loc[:, kw.tname] = pd.to_datetime(df_td[kw.tname], format=kw.datefomatfrom).strftime(kw.datefomatto)
                    except:
                        warnings.warn(f'error when converting {kw.tname} from {kw.datefomatfrom} to {kw.datefomatto}.')
                        continue
                
                df_td['file'] = td
                #df_site = df_site.append(df_td)
                df_site = pd.concat([df_site, df_td], ignore_index=True).reset_index(drop=True)
        
        if df_td.empty == False:
            break
        
    #print('df_td.empty ', df_td.empty)
    if onlynumeric:
        valcols = [i for i in df_site.columns if i.lower() not in [kw.tname.lower(), 'file']]
        _bf = df_site.dtypes
        #df_site.loc[:, valcols] = df_site.loc[:, valcols].apply(pd.to_numeric, errors='coerce')
        df_site[valcols] = df_site[valcols].apply(pd.to_numeric, errors='coerce')
        _af = df_site.dtypes
        if verbosity>1:
            _bfaf = []
            for (k, b) in _bf.items():
                if b!=_af[k]:
                    _nonnum = [s for s in np.unique(df_site[k].apply(lambda s: str(s) if re.findall('[A-z/]+', str(s)) else '')) if s]
                    _bfaf += ['{}, changed from {} to {}. ({})'.format(k, b, _af[k], ', '.join(_nonnum) if _nonnum else 'All numeric')]
            if _bfaf:
                warnings.warn(', '.join(_bfaf))
    
    #if kw_fmt:
    df_site = structuredDataFrame.format(df_site, **kw_['FMT_DATA'])

    if fill:
        if lookup:
            minmax = [min(lookup), max(lookup)]
        else:
            minmax = [np.nanmin(df_site[kw.tname]),
                      np.nanmax(df_site[kw.tname])]
        df_site = df_site.set_index(kw.tname).join(pd.DataFrame({kw.tname: pd.date_range(*minmax, freq=str(kw.dt) + ' S')}).set_index(kw.tname),
                how='outer').ffill().reset_index()
        #if 'co2' in df_site.columns and (abs(np.max(df_site.co2)) < 1000) and (abs(np.min(df_site.co2)) < 1000):
        #    df_site.loc[:, "co2"] = df_site.loc[:, "co2"] * 1000  # mmol/m3 -> μmol/m3
    
    if kw.id is not None:
        return {kw.id: structuredDataFrame(df_site, dt=kw.dt)}
    else:
        return structuredDataFrame(df_site, dt=kw.dt)


def loaddatawithbuffer(d0, d1=None, freq=None, buffer=None, 
                       tname="TIMESTAMP", **kwargs):
    if isinstance(d0, (pd.DatetimeIndex)):
        d0, d1 = [np.nanmin(d0), np.nanmax(d0)]
    
    if buffer == None:
        datarange = [pd.date_range(start=d0, end=d1, freq=freq)[:-1] + pd.Timedelta(freq)]
    else:
        freqno = int(re.match("\d*", "30min")[0])
        
        bufi = np.ceil(buffer / (freqno*60)) * freqno
        datarange = [
            pd.date_range(
                start=pd.to_datetime(d0) - pd.Timedelta(bufi, unit='min'),
                end=pd.to_datetime(d1) + pd.Timedelta(bufi, unit='min'),
                freq=freq)[:-1] + pd.Timedelta(freq)]
                
    if not datarange:
        return pd.DataFrame()
    
    data = structuredDataFrame(lookup=datarange, **kwargs)
    if data == None or data.data.empty:
        return data.data
    data.data[tname] = pd.to_datetime(data.data[tname])
    
    if buffer:
        d0 = pd.to_datetime(d0) - pd.Timedelta(buffer*1.1, unit='s')
        d1 = pd.to_datetime(d1) + pd.Timedelta(buffer*1.1, unit='s')
        data.filter({tname: (d0, d1)})

    # garantee all data points, if any valid time, else empty dataframe
    if np.sum(np.isnat(data.data.TIMESTAMP)==False):
        data.data = pd.merge(pd.DataFrame({tname: pd.date_range(*nanminmax(data.data.TIMESTAMP), freq="0.05S")}),
                            data.data,
                            on=tname, how='outer').reset_index(drop=True)
        return data.data
    else:
        pd.DataFrame()

##########################################
###     METEO                            
##########################################

def vapour_deficit_pressure(T, RH):
    if np.nanquantile(T, 0.1) < 100 and np.nanquantile(T, 0.9) < 100:
        T = T + 274.15

    # Saturation Vapor Pressure (es)
    #es = 0.6108 * np.exp(17.27 * T / (T + 237.3))
    es = (T **(-8.2)) * (2.7182)**(77.345 + 0.0057*T-7235*(T**(-1)))

    # Actual Vapor Pressure (ea)
    ea = es * RH / 100

    # Vapor Pressure Deficit (Pa)
    return (es - ea)# * 10**(-3)

##########################################
###     DESPIKING                            
##########################################

def mauder2013(x, q=7):
    x = np.array(x)
    x_med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - x_med))
    bounds = (x_med - (q * mad) / 0.6745, x_med + (q * mad) / 0.6745)
    #print("median", x_med, "mad", mad, "bounds", bounds)
    x[x < min(bounds)] = np.nan
    x[x > max(bounds)] = np.nan

    #if fill is not None:
    #    x = fill(pd.Series(x) if fill in (pd.Series.ffill, pd.Series.interpolate) else x)
    return x


##########################################
###     GAP FILLING                            
##########################################

def gap_filling(path, latitude, longitude,
                cols={'DW': {'var': 'co2w', 'flag': {'flag(w)': 0}},
                      'EC': {'var': 'cov_wco2', 'flag': {'flag(w)': 0, 'flag(w/co2)': 0}}}):
    latitude = float(latitude)
    longitude = float(longitude)
    r2gapfill.sink_reset()
    #r2gapfill.silenceR()

    # prepare data
    if isinstance(path, str): 
        fluxResult = pd.read_csv(path)
        fluxResult.TIMESTAMP = pd.to_datetime(fluxResult.TIMESTAMP)
    elif isinstance(path, pd.DataFrame): fluxResult = path
    else: return None

    fluxData = structuredData(data=fluxResult)

    fluxData.data = fluxData.data.rename(columns=
        {'air_temperature': 'Tair', 'TA': 'Tair', 'SW_IN': 'Rg', 'RH': 'rH', 'ustar': 'Ustar', 'us': 'Ustar', 'u*': 'Ustar'})
    
    fluxData.data["Tair"] = fluxData.data.Tair-273 if np.nanmean(fluxData.data.Tair) > 100 else fluxData.data.Tair
    fluxData.data["Tsoil"] = fluxData.data.Tair
    #    #'VPD': fluxData.data.VPD/1000,
    #    #'date': fluxData.data.TIMESTAMP.dt.strftime('%Y%m%d')
    #    }).data
            
    if "VPD" not in fluxData.data.columns:
        fluxData.data["VPD"] = vapour_deficit_pressure(fluxData.data.Tair, fluxData.data.rH)

    # assert continuous time series
    fluxData.data = pd.merge(pd.DataFrame({'TIMESTAMP': pd.date_range(min(fluxData.data.TIMESTAMP), max(fluxData.data.TIMESTAMP), freq='30min')}),
                            fluxData.data,
                            on='TIMESTAMP', how='outer').reset_index(drop=True)
    fluxData.data.drop_duplicates("TIMESTAMP", keep="first", inplace=True)

    # do it
    fluxData.fdta = {}

    #fluxData.data['Ustar'] = fluxData.data['us']
    #'co2_flux': 'EP', , 'co2_w_y': 'WV_Mo'
    for k, e in cols.items():
        print(k, end='\r')
        v = e.get('var', '')
        f = e.get('flag', {})
        l = e.get('flux', [])
        print(v)

        fill_data = copy.deepcopy(fluxData.data)
        fill_data = fill_data.rename(columns={v: 'NEE', 'qc_' + v: 'qc_NEE'})
        # drop flagged data
        for flag, flagth in f.items():
            # correct flag name due to previous variable renaming
            flagname = flag if flag != 'qc_' + v else 'qc_NEE'
            for flux in ['NEE'] + l:
                fill_data.loc[fill_data[flagname] > flagth, flux] = np.nan
        

        fill_data.loc[fill_data.Rg < 0, 'Rg'] = 0

        datetimecols = fill_data.select_dtypes(include=['datetime64', 'datetime64[ns, UTC]']).columns
        for c in datetimecols:
            fill_data[c] = fill_data[c].dt.strftime('%Y-%m-%dT%H:%M')
        fill_data.rename(columns={c: c.replace('+', 'i').replace('-', 'o') for c in l}, inplace=True)
        l = [c.replace('+', 'i').replace('-', 'o') for c in l]
        #fill_data.TIMESTAMP_END = pd.to_datetime(fill_data.TIMESTAMP_END).dt.strftime(
        #    '%Y%m%d%H%M')
        #print('4')
        #print(["{}: {} {}".format(c, np.mean(fill_data[c]), np.nanmean(fill_data[c])) for c in ['Tair', 'Rg', 'VPD', 'NEE', 'Ustar', 'rH'] if c in fill_data.columns])
        mkdirs('data/tmp/')
        fill_data.to_csv(f'data/tmp/data_to_be_filled_{k}.csv', index=False)
        _ = r2gapfill.do_gapfill(f'data/tmp/data_to_be_filled_{k}.csv', 
                                         climcols=['Tair', 'Rg', 'VPD'], fluxcols=['NEE'] + l,  # , 'LE'
                                         Lon=longitude, Lat=latitude, partitioning='none',
                                         verbosity=1,
                                         save=True, path_to_output_file=f'data/tmp/data_gap_filled_{k}.csv')
        #fill_data = pandas2ri.rpy2py(fill_data).reset_index(drop=True)
        fill_data = pd.read_csv(f'data/tmp/data_gap_filled_{k}.csv', skiprows=0)
        fill_data.rename(columns={c: c.replace(re.findall("wco2..wh2o", c)[0],
                                               "wco2"+re.findall("(?<=wco2)..(?=wh2o.*)", c)[0].replace("o", "-").replace("i", "+")+"wh2o") 
                                               for c in fill_data.columns if re.findall("(?<=wco2)..(?=wh2o.*)", c)}, inplace=True)
        fill_data['TIMESTAMP'] = pd.to_datetime(fill_data.TIMESTAMP).dt.tz_localize(None)
        for c in datetimecols:
            fill_data[c] = pd.to_datetime(fill_data[c])
        #print('5')
        #print("fill_data.columns", [
        #      c for c in fill_data.columns if c.upper().startswith('RECO') or c.upper().startswith('GPP')], fill_data.columns)
        #fill_data = fill_data[[c for c in fill_data.columns if ((c in ['TIMESTAMP', 'Tair_f', 'Rg_f', 'VPD_f']) or (
        #    c.startswith('NEE')) or (c.startswith('Reco')) or (c.startswith('GPP')))]]
        #print("fill_data.columns after", [
        #      c for c in fill_data.columns if c.upper().startswith('RECO') or c.upper().startswith('GPP')])

        fluxData.fdta[k] = fill_data
        del k, v, f, fill_data

    # wrap it up
    fluxData.fdata = copy.deepcopy(fluxData.data)
    fluxData.fdata['TIMESTAMP'] = pd.to_datetime(fluxData.fdata.TIMESTAMP).dt.tz_localize(None)

    # climatic variables (take from first run)
    for c in ['Tair_f', 'Rg_f']:  # , 'VPD_f']:
        fluxData.fdata[c] = fluxData.fdta[list(fluxData.fdta.keys())[0]][c]

    # fluxes variables (each run is unique)
    for k, d in fluxData.fdta.items():
        print(d.columns.to_list())
        for c in d.columns:
            #cs = ['TIMESTAMP']
            if c.startswith('Reco') or c.startswith('GPP') or c.startswith('NEE'):
                #c = c.replace('uStar_', '')
                d.rename(columns={c: f'{k}_{c}'}, inplace=True)
                #d[f'{k}_{c}'] = d[c]
                #cs += [f'{k}_{c}']
            
            d['TIMESTAMP'] = d.TIMESTAMP.dt.tz_localize(None)
            fluxData.fdata = pd.merge(fluxData.fdata, d[['TIMESTAMP']+list(set(d.columns)-set(fluxData.fdata.columns))], on='TIMESTAMP', how='left')
                
        """
        fluxData.fdata[k+'_FCO2_f'] = d.NEE_uStar_f
        fluxData.fdata[k+'_FCO2_o'] = d.NEE_uStar_orig
        if 'GPP_uStar_f' in d.columns:
            fluxData.fdata[k+'_GPP'] = d.GPP_uStar_f
        if 'Reco_uStar' in d.columns:
            fluxData.fdata[k+'_RECO'] = d.Reco_uStar
        """
    
    # save
    return(fluxData.fdata)


##########################################
###     WAVELET-RELATED                            
##########################################

def bufferforfrequency_dwt(N=0, n_=30*60*20, fs=20, level=None, f0=None, max_iteration=10**4, wave='db6'):
    if level is None and f0 is None: level = 18
    lvl = level if level is not None else int(np.ceil(np.log2(fs/f0)))
    n0 = N
    cur_iteration = 0
    while True:
        n0 += pd.to_timedelta(n_)/pd.to_timedelta("1S") * fs if isinstance(n_, str) else n_
        if lvl <= pywt.dwt_max_level(n0, wave):
            break
        cur_iteration += 1
        if cur_iteration > max_iteration:
            warnings.warn('Limit of iterations attained before buffer found. Current buffer allows up to {} levels.'.format(
                pywt.dwt_max_level(n0, wave)))
            break
    return (n0-N) * fs**-1


def bufferforfrequency(f0, dt=0.05, param=6, mother="MORLET", wavelet=pycwt.Morlet(6)):
    #check if f0 in right units
    # f0 ↴
    #    /\
    #   /  \
    #  /____\
    # 2 x buffer
    
    c = wavelet.flambda() * wavelet.coi()
    n0 = 1 + (2 * (1/f0) * (c * dt)**-1)
    N = int(np.ceil(n0 * dt))
    return N
