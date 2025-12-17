""""LICENSE:

The copyright in this software is being made available under the Clear BSD
License, included below. No patent rights, trademark rights and/or
other Intellectual Property Rights other than the copyrights concerning
the Software are granted under this license.

The Clear BSD License

Copyright (c) 2023, Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V. & the authors: Johanna Vielhaben, Sebastian Lapuschkin, Grégoire Montavon, Wojciech Samek.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted (subject to the limitations in the disclaimer
below) provided that the following conditions are met:

     * Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.

     * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

     * Neither the name of the copyright holder nor the names of its
     contributors may be used to endorse or promote products derived from this
     software without specific prior written permission.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

"""

"""
    Original codes adapted and extended by Nils Gumpfer, Technische Hochschule Mittelhessen
"""


import numpy as np
import torch
import torch.nn as nn


def create_fourier_weights(signal_length, inverse=False, symmetry=False, real=False):  
    """
    symmetry: use that DFT of real signal is symmteric and use only half of transformed signal for inverse trafo
    """
    k_vals, n_vals = np.mgrid[0:signal_length, 0:signal_length]
    theta_vals = 2 * np.pi * k_vals * n_vals / signal_length
    sign = 1.0 if inverse else -1.0
    norm = 1/np.sqrt(signal_length)
    if symmetry:
        nyquist_k = signal_length//2
        if inverse:
            w_0 = np.ones(signal_length)[np.newaxis,:]
            w_nyquist = np.ones(signal_length)[np.newaxis,:]
            if real:
                return norm*np.vstack([w_0, 2*np.cos(theta_vals[1:nyquist_k]), w_nyquist, -2*np.sin(theta_vals[1:nyquist_k])])
            else:
                return norm*np.vstack([w_0, 2*np.exp(sign * 1j * theta_vals[1:nyquist_k]), w_nyquist])
        else:
            if real:
                return norm*np.hstack([np.cos(theta_vals[:,:nyquist_k+1]), -np.sin(theta_vals[:,1:nyquist_k])])
            else:
                return norm*np.exp(sign * 1j * theta_vals[:,:nyquist_k+1]) 
    else:
        if real:
            if inverse: 
                return norm*np.vstack([np.cos(theta_vals), -np.sin(theta_vals)])
            else:
                return norm*np.hstack([np.cos(theta_vals), -np.sin(theta_vals)])
        else:
            # inverse handled by sign
            return norm*np.exp(sign * 1j * theta_vals) 
def rectangle_window(m, width, signal_length, shift):
    """
    m: int, window shift (in units of time points)
    width: int, window width
    shift: int, fraction of window width by which window is shifted

    Caution: Perfect reconstruction condition fullfilled only for shift=(1,2)
    """
    w_nm = np.zeros(signal_length)
    w_nm[m:m+width] = 1/np.sqrt(shift)
    return w_nm


def halfsine_window(m, width, signal_length, shift=None):
    """
    m: int, window shift (in units of time points)
    width: int, window width
    shift: dummy parameter
    """
    w_nm = np.zeros(signal_length)
    w_nm[m:m+width] = np.sin(np.pi/width*(np.arange(width) + 0.5))
    return w_nm
WINDOWS = {"rectangle": rectangle_window, "halfsine": halfsine_window}
def create_window_mask(shift, width, signal_length, window_function):
    ms = np.arange(0, signal_length-width+1, width//shift)

    W_mn = [window_function(m, width, signal_length, shift)[np.newaxis] for m in ms]
    W_mn = np.concatenate(W_mn, axis=0)
    return W_mn.transpose((1,0))


def create_short_time_fourier_weights(signal_length, shift, window_width, window_shape, inverse=False, real=False, symmetry=False):
    assert window_shape in ("rectangle", "halfsine", "hann"), "Available window shapes: rectangle, halfsine"

    if window_shape=="rectangle":
        window_function = rectangle_window
    elif window_shape=="halfsine":
        window_function = halfsine_window
    #elif window_shape=="hann":
    #    window_function = hann_window

    W_mn = create_window_mask(shift, window_width, signal_length, window_function)
    
    DFT_kn = create_fourier_weights(signal_length, inverse=inverse, symmetry=symmetry, real=real)
    
    dtype = np.complex64 if not real else np.float16

    if inverse:
        W = W_mn.sum(axis=1)

        DFT_kn_m = np.zeros((W_mn.shape[1]*DFT_kn.shape[0], DFT_kn.shape[1]), dtype=dtype)
        for ki, i in enumerate(range(0, DFT_kn_m.shape[0],DFT_kn.shape[0])):
            DFT_kn_m[i:i+DFT_kn.shape[0]] = DFT_kn

        STDFT_mkn = DFT_kn_m / W.astype(dtype)
    else:
        STDFT_mkn = np.zeros((DFT_kn.shape[0], W_mn.shape[1]*DFT_kn.shape[1]), dtype=dtype)
        for m, k in enumerate(range(0, W_mn.shape[1]*DFT_kn.shape[1], DFT_kn.shape[1])):
            STDFT_mkn[:,k:k+DFT_kn.shape[1]] = DFT_kn * W_mn[:,m][:,np.newaxis]
    return STDFT_mkn


class DFTLRP():
    def __init__(self, signal_length, precision=32, cuda=True, leverage_symmetry=False, window_shift=None,
                 window_width=None, window_shape=None, create_inverse=True, create_transpose_inverse=True,
                 create_forward=True, create_dft=True, create_stdft=True) -> None:
        """
        Class for Discrete Fourier transform in pytorch and relevance propagation through DFT layer.

        Args:
        signal_length: number of time steps in the signal
        leverage_symmetry: if True, levereage that for real signal the DF transformed signal is symmetric and compute only first half it
        cuda: use gpu
        precision: 32 or 16 for reduced precision with less memory usage

        window_width: width of the window for short time DFT
        window_shift: width/hopsize of window for short time DFT
        window_shape: shape of window for STDFT, options are 'rectangle' and 'halfsine'

        create_inverse: create weights for inverse DFT
        create_transpose: cretae weights for transpose inverse DFT (for DFT-LRP)
        create_forward: create weights for forward DFT
        create_stdft: create weights for short time DFT
        create_stdft: create weights DFT
        """
        self.signal_length = signal_length
        self.nyquist_k = signal_length // 2
        self.precision = precision
        self.cuda = cuda
        self.symmetry = leverage_symmetry
        self.stdft_kwargs = {"window_shift": window_shift, "window_width": window_width, "window_shape": window_shape}

        # create fourier layers
        # dft
        if create_dft:
            if create_forward:
                self.fourier_layer = self.create_fourier_layer(signal_length=self.signal_length, symmetry=self.symmetry,
                                                               transpose=False, inverse=False, short_time=False,
                                                               cuda=self.cuda, precision=self.precision)
            # inverse dft
            if create_inverse:
                self.inverse_fourier_layer = self.create_fourier_layer(signal_length=self.signal_length,
                                                                       symmetry=self.symmetry, transpose=False,
                                                                       inverse=True, short_time=False, cuda=self.cuda,
                                                                       precision=self.precision)
            # transpose inverse dft for dft-lrp
            if create_transpose_inverse:
                self.transpose_inverse_fourier_layer = self.create_fourier_layer(signal_length=self.signal_length,
                                                                                 symmetry=self.symmetry, transpose=True,
                                                                                 inverse=True, short_time=False,
                                                                                 cuda=self.cuda,
                                                                                 precision=self.precision)

        if create_stdft:
            # stdft
            if create_forward:
                self.st_fourier_layer = self.create_fourier_layer(signal_length=self.signal_length,
                                                                  symmetry=self.symmetry, transpose=False,
                                                                  inverse=False, short_time=True, cuda=self.cuda,
                                                                  precision=self.precision, **self.stdft_kwargs)
            # inverse stdft
            if create_inverse:
                self.st_inverse_fourier_layer = self.create_fourier_layer(signal_length=self.signal_length,
                                                                          symmetry=self.symmetry, transpose=False,
                                                                          inverse=True, short_time=True, cuda=self.cuda,
                                                                          precision=self.precision, **self.stdft_kwargs)
            # transpose inverse stdft for dft-lrp
            if create_transpose_inverse:
                self.st_transpose_inverse_fourier_layer = self.create_fourier_layer(signal_length=self.signal_length,
                                                                                    symmetry=self.symmetry,
                                                                                    transpose=True, inverse=True,
                                                                                    short_time=True, cuda=self.cuda,
                                                                                    precision=self.precision,
                                                                                    **self.stdft_kwargs)

    @staticmethod
    def _array_to_tensor(input: np.ndarray, precision: float, cuda: bool) -> torch.tensor:
        dtype = torch.float32 if precision == 32 else torch.float16
        input = torch.tensor(input, dtype=dtype)
        if cuda:
            input = input.cuda()
        return input

    @staticmethod
    def create_fourier_layer(signal_length: int, inverse: bool, symmetry: bool, transpose: bool, short_time: bool,
                             cuda: bool, precision: int, **stdft_kwargs):
        """
        Create linear layer with Discrete Fourier Transformation weights

        Args:
        inverse: if True, create weights for inverse DFT
        symmetry: if True, levereage that for real signal the DF transformed signal is symmetric and compute only first half it
        transpose: create layer with transposed DFT weights for explicit relevance propagation
        short_time: short time DFT
        cuda: use gpu
        precision: 32 or 16 for reduced precision with less memory usage
        """
        if short_time:
            weights_fourier = create_short_time_fourier_weights(signal_length, stdft_kwargs["window_shift"],
                                                                          stdft_kwargs["window_width"],
                                                                          stdft_kwargs["window_shape"], inverse=inverse,
                                                                          real=True, symmetry=symmetry)
        else:
            weights_fourier = create_fourier_weights(signal_length=signal_length, real=True, inverse=inverse,
                                                               symmetry=symmetry)

        if transpose:
            weights_fourier = weights_fourier.T

        weights_fourier = DFTLRP._array_to_tensor(weights_fourier, precision, cuda).T

        n_in, n_out = weights_fourier.shape
        fourier_layer = torch.nn.Linear(n_in, n_out, bias=False)
        with torch.no_grad():
            fourier_layer.weight = nn.Parameter(weights_fourier)
        del weights_fourier

        if cuda:
            fourier_layer = fourier_layer.cuda()

        return fourier_layer

    @staticmethod
    def reshape_signal(signal: np.ndarray, signal_length: int, relevance: bool, short_time: bool, symmetry: bool):
        """
        Restructure array from concatenation of real and imaginary parts to complex (if array contains signal) or sum of real and imaginary part (if array contains relevance). Additionallty, reshapes time-frequenc

        Args:
        relevance: True if array contains relevance, not signal itself
        symmetry: if True, levereage that for real signal the DF transformed signal is symmetric and compute only first half it
        short_time: short time DFT
        """
        bs = signal.shape[0]
        if symmetry:
            nyquist_k = signal_length // 2
            if short_time:
                n_windows = signal.shape[-1] // signal_length
                signal = signal.reshape(bs, n_windows, signal_length)
            zeros = np.zeros_like(signal[..., :1])
            if relevance:
                signal = signal[..., :nyquist_k + 1] + np.concatenate([zeros, signal[..., nyquist_k + 1:], zeros],
                                                                      axis=-1)
            else:
                signal = signal[..., :nyquist_k + 1] + 1j * np.concatenate([zeros, signal[..., nyquist_k + 1:], zeros],
                                                                           axis=-1)
        else:
            if short_time:
                n_windows = signal.shape[-1] // signal_length // 2
                signal = signal.reshape(bs, n_windows, signal_length * 2)
            if relevance:
                signal = signal[..., :signal_length] + signal[..., signal_length:]
            else:
                signal = signal[..., :signal_length] + 1j * signal[..., signal_length:]
        return signal

    def fourier_transform(self, signal: np.ndarray, real: bool = True, inverse: bool = False,
                          short_time: bool = False) -> np.ndarray:
        """
        Discrete Fourier transform (DFT) of signal in time (inverse=False) or inverse DFT of signal in frequency.

        Args:
        inverse: if True, perform inverse DFT
        short_time: if True, perform short time DFT
        real: if real, the output is split into real and imaginary parts of the signal in freq. domain y_k, i.e. (y_k^real, y_k^imag)
        """
        if inverse:
            if short_time:
                transform = self.st_inverse_fourier_layer
            else:
                transform = self.inverse_fourier_layer
        else:
            if short_time:
                transform = self.st_fourier_layer
            else:
                transform = self.fourier_layer

        signal = self._array_to_tensor(signal, self.precision, self.cuda)

        with torch.no_grad():
            signal_hat = transform(signal).cpu().numpy()

        # render y_k as complex number of shape (n_windows, signal_length) //2
        if not real and not inverse:
            signal_hat = self.reshape_signal(signal_hat, self.signal_length, relevance=False, short_time=short_time,
                                             symmetry=self.symmetry)
        return signal_hat

    def dft_lrp(self, relevance: np.ndarray, signal: np.ndarray, signal_hat=None, short_time=False, epsilon=1e-6,
                real=False) -> np.ndarray:
        """
        Relevance propagation thorugh DFT

        relevance: relevance in time domain
        signal: signal in time domain, same shape as relevance
        signal_hat: signal in frequency domain, if None it is computed using signal
        short_time: relevance propagation through short time DFT
        epsilon: small constant to stabilize denominantor in DFT-LRP
        real: if True, the signal_hat after DFT and correspondong relevance is split into real and imaginary parts of the signal in freq. domain y_k, i.e. (y_k^real, y_k^imag)
        """
        if short_time:
            transform = self.st_fourier_layer
            dft_transform = self.st_transpose_inverse_fourier_layer
        else:
            transform = self.fourier_layer
            dft_transform = self.transpose_inverse_fourier_layer

        signal = self._array_to_tensor(signal, self.precision, self.cuda)
        if signal_hat is None:
            signal_hat = transform(signal)

        relevance = self._array_to_tensor(relevance, self.precision, self.cuda)
        norm = signal + epsilon
        relevance_normed = relevance / norm

        relevance_normed = self._array_to_tensor(relevance_normed, self.precision, self.cuda)
        signal_hat = self._array_to_tensor(signal_hat, self.precision, self.cuda)
        with torch.no_grad():
            relevance_hat = dft_transform(relevance_normed)
            relevance_hat = signal_hat * relevance_hat

        relevance_hat = relevance_hat.cpu().numpy()
        signal_hat = signal_hat.cpu().numpy()

        # add real and imaginary part of relevance and signal
        if not real:
            signal_hat = self.reshape_signal(signal_hat, self.signal_length, relevance=False, short_time=short_time,
                                             symmetry=self.symmetry)
            relevance_hat = self.reshape_signal(relevance_hat, self.signal_length, relevance=True,
                                                short_time=short_time, symmetry=self.symmetry)

        return signal_hat, relevance_hat


def calculate_dft_explanation(signal_time, relevance_time, leverage_symmetry=True, precision=32, window_shift=1, window_width=128, window_shape="rectangle", cuda=False):
    """
        Combined call for DFT-LRP

        Args:
        leverage_symmetry: if True, levereage that for real signal the DF transformed signal is symmetric and compute only first half it
        cuda: use gpu
        precision: 32 or 16 for reduced precision with less memory usage

        window_width: width of the window for short time DFT
        window_shift: width/hopsize of window for short time DFT
        window_shape: shape of window for STDFT, options are 'rectangle' and 'halfsine'
    """

    signal_length = np.shape(signal_time)[1]

    dftlrp = DFTLRP(
        signal_length,
        leverage_symmetry=leverage_symmetry,
        precision=precision,
        create_stdft=False,
        create_inverse=False,
        cuda=cuda,
    )

    signal_freq, relevance_freq = dftlrp.dft_lrp(
        relevance_time,
        signal_time,
        real=False,
        short_time=False,
    )

    del dftlrp

    dftlrp = DFTLRP(
        signal_length,
        leverage_symmetry=leverage_symmetry,
        precision=precision,
        window_shift=window_shift,
        window_width=window_width,
        window_shape=window_shape,
        create_dft=False,
        create_inverse=False,
        cuda=cuda,
    )

    signal_timefreq, relevance_timefreq = dftlrp.dft_lrp(
        relevance_time,
        signal_time,
        real=False,
        short_time=True,
    )

    del dftlrp

    return signal_freq, relevance_freq, signal_timefreq, relevance_timefreq