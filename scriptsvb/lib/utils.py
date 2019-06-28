#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""
This module contains tools for reading and writing files in HDF5 or HTK format
"""

__version__ = '1.2'
__author__ = 'Omid Sadjadi'
__email__ = 'omid.sadjadi@nist.gov'

import os
import errno
import h5py
import struct
import re
import mmap
import numpy as np


def mkdir_p(directory):
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise  # raises the error again


def h5read(filename, datasets):
    if type(datasets) != list:
        datasets = [datasets]
    with h5py.File(filename, 'r') as h5f:
        data = [h5f[ds][:] for ds in datasets]
    return data


def h5write(filename, data, datasets, dtype='f'):
    dirname = os.path.dirname(filename)
    mkdir_p(dirname)
    if type(datasets) != list:
        datasets = [datasets]
    if type(data) != list:
        data = [data]
    if len(data) != len(datasets):
        raise ValueError('data and datasets must have the same length. len(data)={} while len(datasets)={}.'.format(len(data), len(datasets)))
    with h5py.File(filename, 'w') as h5f:
        for dt, ds in zip(data, datasets):
            h5f.create_dataset(ds, data=dt, chunks=True, dtype=dtype,
                               compression='gzip', compression_opts=9)


def htkwrite_np(filename, data, frate=100000, feakind=9):
    dirname = os.path.dirname(filename)
    mkdir_p(dirname)
    ndim, nobs = data.shape
    header = struct.pack('>IIHH', nobs, frate, 4 * ndim, feakind)
    assert len(header) == 12
    with open(filename, 'wb') as fid:
        fid.write(header)
        data.astype('f').byteswap().ravel(order='F').tofile(fid)


def htkread_np(filename):
    with open(filename, 'rb') as fid:
        header = fid.read(12)
        s = np.fromfile(fid, dtype='>f')
    nobs, frate, nbytes, feakind = struct.unpack('>IIHH', header)
    ndim = nbytes // 4
    return s.reshape(ndim, nobs)


def nnet3read(dnnFilename, outFilename="", write_to_disk=False):
    """ This is a simple, yet fast, routine that reads in Kaldi NNet3 Weight
        and Bias parameters, and converts them into lists of 64-bit floating
        point numpy arrays and optionally dumps the parameters to disk in HDF5
        format.

        :param dnnFilename: input DNN file name (it is assumed to be in text format)
        :param outFilename: output hdf5 filename [optional]
        :param write_to_disk: whether the parameters should be dumped to disk [optional]

        :type dnnFilename: string
        :type outFilename: string
        :type write_to_diks: bool

        :return: returns the NN weight and bias parameters (optionally dumps to disk)
        :rtype: tuple (b,W) of list of 64-bit floating point numpy arrays

        :Example:

        >>> b, W = nnet3read('final.txt', 'DNN_1024.h5', write_to_disk=True)
    """
    # nn_elements = ['LinearParams', 'BiasParams']
    with open(dnnFilename, 'r') as f:
        pattern = re.compile(rb'<(\bLinearParams\b|\bBiasParams\b)>\s+\[\s+([-?\d\.\de?\s]+)\]')
        with mmap.mmap(f.fileno(), 0,
                       access=mmap.ACCESS_READ) as m:
            b = []
            W = []
            ix = 0
            for arr in pattern.findall(m):
                if arr[0] == b'BiasParams':
                    b.append(arr[1].split())
                    print("layer{}: [{}x{}]".format(ix, len(b[ix]), len(W[ix])//len(b[ix])))
                    ix += 1
                elif arr[0] == b'LinearParams':
                    W.append(arr[1].split())
                else:
                    raise ValueError('oops... NN element not recognized!')

    # converting list of strings into lists of 64-bit floating point numpy arrays and reshaping
    b = [np.array(b[ix], dtype=np.float).reshape(-1, 1) for ix in range(len(b))]
    W = [np.array(W[ix], dtype=np.float).reshape(len(b[ix]), len(W[ix])//len(b[ix])) for ix in range(len(W))]

    if write_to_disk:
        # writing the DNN parameters to an HDF5 file
        if not outFilename:
            raise ValueError('oops... output file name not specified!')
        dw = ['w'+str(ix) for ix in range(len(W))]
        db = ['b'+str(ix) for ix in range(len(b))]
        h5write(outFilename, W+b, dw+db, dtype='f8')

    return b, W
