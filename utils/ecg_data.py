import numpy as np
import wfdb

from utils.ecg_filtering import filter_signal, remove_baseline_wander, adjust_baseline, derive_filter_values


def load_and_preprocess_ecg(record_id, ecg_filters, subsampling_window_size, subsample_start, src_dir='./examples/data/timeseries/'):
    leads = load_raw_ecg_wfdb('{}{}'.format(src_dir, record_id))
    ecg = convert_lead_dict_to_matrix(leads)
    ecg = filter_ecg(ecg, ecg_filters)
    ecg = extract_subsample_from_ecg(ecg, subsample_start, subsample_start + subsampling_window_size)
    return perform_shape_switch(ecg)


def load_raw_ecg_wfdb(filepath, verbose=True):
    if verbose:
        print('Loading {}'.format(filepath))

    signal, meta = wfdb.rdsamp(filepath)
    signal = perform_shape_switch(signal)
    signal = np.nan_to_num(signal)
    meta['sig_name'] = list(meta['sig_name'])

    leads = {}

    lead_id_count = {k: meta['sig_name'].count(k) for k in meta['sig_name']}
    lead_id_counter = {k: 1 for k in meta['sig_name']}

    for lead_id, lead_signal in zip(meta['sig_name'], signal):
        if lead_id_count[lead_id] > 1:
            l_id = '{}_{}'.format(lead_id, lead_id_counter[lead_id])
            lead_id_counter[lead_id] += 1
        else:
            l_id = lead_id
        leads[l_id] = lead_signal

    return leads


def convert_lead_dict_to_matrix(leads, shape_switch=False):
    collected = []

    for lead_id in leads:
        collected.append(leads[lead_id])

    collected = np.array(collected)

    if shape_switch:
        collected = perform_shape_switch(collected)

    return collected


def filter_lead(lead, filters, fs):
    filter_values = derive_filter_values(filters)

    for f in filter_values:
        if f == 'AC':
            lead = filter_signal(lead, filter_values[f], sample_rate=fs, filtertype='notch')
        elif f == 'HP':
            lead = filter_signal(lead, filter_values[f], sample_rate=fs, filtertype='highpass')
        elif f == 'LP':
            lead = filter_signal(lead, filter_values[f], sample_rate=fs, filtertype='lowpass')
        elif f == 'BWR':
            lead = remove_baseline_wander(lead, fs, cutoff=0.05)
        elif f == 'BLA':
            lead = adjust_baseline(lead, fs)
        else:
            raise Exception('Unknown ECG filer: "{}"'.format(f))

    return lead


def filter_ecg(ecg, filters, fs=500):
    for i, lead in enumerate(ecg):
        ecg[i] = filter_lead(ecg[i], filters, fs)

    return ecg


def extract_subsample_from_ecg(ecg, start, end):
    return ecg[:, start:end]


def perform_shape_switch(a):
    a = np.asarray(a)

    dimx, dimy = a.shape

    output = np.zeros((dimy, dimx))

    for i in range(dimx):
        output[:, i] = a[i, :]

    return output