import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, filtfilt
from scipy.fft import fft, fftfreq, next_fast_len
from scipy.interpolate import interp1d


def getSpeed(xMagnetic, fs, maxSpeed=200, b=firwin(9, 0.1), maxValidSpeedDiff=3, isPlot=False, figsize=[15, 5]):
    # Implementation of the speed estimation, based on phase zero crossing and low-pass filtering of the estimated
    # speed. Assumptions:
    # 1. The speed is stable at the edges of the section.
    # 2. The machine is continuously operating during the processed section.

    # inputs:
    # xMagnetic - signal of the magnetic flux (numpy vector)
    # fs - sampling rate [samples / second] (integer)
    # maxSpeed - maximal speed [rounds / second] (scalar)
    # b - Finite Impulse Response filter coefficients
    # isPlot - plotting input (boolean)
    # figsize - size of the plotted figure

    # outputs:
    # filtSpeed - estimated speed vector
    # speedTime - time vector for which the speed is estimated

    # Further improvements should include elimination of the assumption of the stationary speed at the edges.
    # Continuous operation assumption might be handled by thresholding of abs(diff(rollingMax(filtSpeed-speed)))

    # detection of the zero crossing of the magnetic flux signal
    crossLocs = np.where(np.diff(np.sign(xMagnetic - np.median(xMagnetic))))[0]
    # minimal distance between two zero crossings
    minCrossDist = int(fs / maxSpeed / 2)
    # avoiding faulty detection of crossing due to noise
    crossLocs = crossLocs[np.where(np.diff(crossLocs) > minCrossDist)]
    # Validation that full cycles are sampled - sampling of half-cycle leads to leakage
    if crossLocs.shape[0] % 2:
        crossLocs = crossLocs[:-1]
    # locations of the estimated speed
    speedLocs = (crossLocs[:-1] + crossLocs[1:]) / 2
    # time vector of the estimated speed
    speedTime = speedLocs / fs
    # estimation of the speed
    speed = fs / np.diff(crossLocs) / 2
    # due to the presence of higher harmonics in the signal, we have to filter the estimated speed.
    if speed.shape[0] > 27:
        filtSpeed = filtfilt(b, 1, speed, padtype='even')
    else:
        return None, None


    if isPlot:
        plt.figure(figsize=figsize)
        plt.plot(speedTime, speed)
        plt.plot(speedTime, filtSpeed)
        plt.legend(['Gross speed estimation', 'Filtered speed estimation'])
        plt.xlabel('Time [sec]')
        plt.ylabel('Shaft speed [Cycles / second]')
        plt.grid()
        plt.show()
    
    if np.any(np.abs(speed - filtSpeed) > maxValidSpeedDiff):
        return None, None

    return filtSpeed, speedTime


def orderTrack(x, t, speed, speedTime):
    # this function implements the order-tracking. Following it should allow iterative implementation.

    # inputs:
    # x - numpy matrix of the signals for the resampling
    # t - sampling time vector of the signals to resample [second]
    #
    # speed - speed numpy vector [cycles / second]
    # speedTime - time numpy vector of the speed [second]

    # outputs:
    # resampledPhase - shaft phase of the resampled signals [cycle]
    # resampledX - resampled signal numpy matrix (vector)
    # t - trimmed time vector of the original signal to resample [second]
    # phase - trimmed phase of the signal to resample (based on the input shaft speed) [cycles]
    # interpSpeed - numpy vector speed interpolated to the trimmed time vector - t
    # x - trimmed signal for the interpolation

    if (speed is None) or (speedTime is None):
        return None, None, None, None, None, None

    # setting the dimensions of x
    x = np.atleast_2d(x)

    if (x.shape[1] > x.shape[0]):
        x = x.T

    # trimming the samples with a non-defined speed
    cond = (t > speedTime[0]) & (t < speedTime[-1])
    x = x[cond, :]
    t = t[cond]

    # interpolation of speed to the time of the signal that should be interpolated
    interpSpeedFun = interp1d(speedTime, speed, axis=0)
    interpSpeed = interpSpeedFun(t)

    # shaft phase by integration of the shaft speed [cycles]
    phase = np.cumsum(interpSpeed) * (t[1] - t[0])
    # zeroing the phase with respect to the first sample
    phase -= phase[0]

    # Following lines are defining the new phase increment, considering following aspects:
    # avoiding aliasing by sampling the with phase increments at least as small as previously
    minDphase = np.min(speed) * (t[1] - t[0])
    # Forcing Fourier domain aggregation points at round shaft speed orders
    roundPhase = np.round(phase[-1])
    # fast FFT calculation
    dPhase = roundPhase / next_fast_len(np.ceil(roundPhase / minDphase).astype(int))

    # creation of the new phase vector with a constant phase increments
    resampledPhase = np.arange(0, roundPhase, dPhase)
    # resampling the signal to locations with a constant phase increments
    interpFun = interp1d(phase, x, axis=0)
    resampledX = interpFun(resampledPhase)

    # squeezing the signals for cases with one resampled signal (rather then multiple signals)
    resampledX = np.squeeze(resampledX)
    x = np.squeeze(x)

    return resampledX, resampledPhase, t, phase, interpSpeed, x


def getTestCase():
    # this function provides the test case of magnetic flux signal created by a motor with a speed of
    # speed = 10 + 2 * varSpeedFun, where varSpeedFun goes from 0 at the edges of the signal to 1 at the centre.

    # sampling rate [samples / sec]
    fs = 10 ** 4
    # sampling time [seconds]
    T = 4
    # nominal speed - cycles / sec
    nominalSpeed = 10
    # maximal additional speed [cycles / sec]
    additionalSpeed = 2

    # time vector
    t = np.arange(0, T, 1 / fs)
    # additional speed phase
    speedVarPhase = t * 2 * np.pi / T
    # additional speed function
    speedVariation = (1 - np.cos(speedVarPhase)) * 0.5
    # speed vector
    speed = nominalSpeed + additionalSpeed * speedVariation
    # shaft phase by integration of the shaft speed [cycles]
    phase = np.cumsum(speed) / fs
    # simulation of the magnetic flux
    xMagnetic = np.sin(phase * 2 * np.pi)
    # output packing
    res = {
        'fs': fs,
        'time': t,
        'speed': speed,
        'xMagnetic': xMagnetic,
        'phase': phase
    }
    return res


def testGetSpeed():
    # import test case
    params = getTestCase()
    # get the speed vector from the magnetic flux signal
    speedEstimated, timeEstimated = getSpeed(params['xMagnetic'], params['fs'])
    # reference function
    interpSpeedFun = interp1d(params['time'], params['speed'])
    # expected speed at the locations where the speed was estimated
    speedRef = interpSpeedFun(timeEstimated)
    # relative error
    relErr = np.std(np.abs(speedRef - speedEstimated)) / speedRef.mean()
    assert (relErr < 10**-3)


def testOrderTrack():
    # this test should demonstrate that the order tracking is reducing the smearing of the spectrum.
    # For the test case, all the energy is concentrated in a single peak,
    # at the expected location of the shaft speed frequency harmonics (here only the first harmonic)

    # testing values:
    # first shaft speed harmonic is the expected location for the peak of the amplitude
    expectedShaftSpeed = 1
    # expected amplitude for a theoretical sin/cos function
    expectedDFT_Amp = 0.5

    # import test case
    params = getTestCase()

    # get the speed vector from the magnetic flux signal
    speedEstimated, timeEstimated = getSpeed(params['xMagnetic'], params['fs'])
    # order tracking - resampling from the time domain to the shaft phase domain
    resampledMagnetic, resampledPhase, t, phase, interpSpeed, x = orderTrack(
        params['xMagnetic'], params['time'], speedEstimated, timeEstimated
    )

    # Discrete Fourier Transform - for testing of the resampled signal
    f = fftfreq(resampledMagnetic.shape[0], d=(resampledPhase[1] - resampledPhase[0]))
    dft = fft(resampledMagnetic, n=resampledMagnetic.shape[0], norm="forward")

    # index of the expected shaft speed order
    ind = np.where(f == expectedShaftSpeed)
    # estimated amplitude at the expected shaft speed order
    estimatedAmp = np.abs(dft[ind])
    # estimation amplitude error
    err = np.abs(expectedDFT_Amp - estimatedAmp)
    assert (err < 5e-4)


if __name__ == '__main__':
    testGetSpeed()
    testOrderTrack()
