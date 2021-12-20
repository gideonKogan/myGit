import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, filtfilt
from scipy.fft import fft, fftfreq, next_fast_len
from scipy.interpolate import interp1d


def getSpeed(xMagnetic, fs, b=firwin(9, 0.1), isPlot=False, figsize=[15, 5]):
    # this function implements the speed estimation based on phase zero crossing and
    # low-pass filtering of the estimated speed

    # Further improvements should include automatic optimization of the filtering boundary conditions
    # and allowing iterative implementation

    crossLocs = np.where(np.diff(np.sign(xMagnetic - np.median(xMagnetic))))[0]
    # avoiding fault detection of crossing due to noise
    crossLocs = crossLocs[np.where(np.diff(crossLocs) > 30)]
    # avoiding sampling of half-cycle, which leads to leakage
    if crossLocs.shape[0] % 2:
        crossLocs = crossLocs[:-1]
    # locations of the estimated speed
    speedLocs = (crossLocs[:-1] + crossLocs[1:]) / 2
    speedTime = speedLocs / fs
    # could skip estimation of the speed but it is nice to have it
    speed = fs / np.diff(crossLocs) / 2
    # due to the presence of higher harmonics in the signal, we have to filter the estimated speed.
    # Here, I don't care that the distance between the samples is not constant
    filtSpeed = filtfilt(b, 1, speed, padtype='even')

    if isPlot:
        plt.figure(figsize=figsize)
        plt.plot(speedTime, speed)
        plt.plot(speedTime, filtSpeed)
        plt.legend(['Gross speed estimation', 'Filtered speed estimation'])
        plt.xlabel('Time [sec]')
        plt.ylabel('Shaft speed [Cycles / second]')
        plt.grid()
        plt.show()

    return filtSpeed, speedTime


def orderTrack(x, t, speed, speedTime):
    # this function implements the order-tracking

    # setting x dimensionality
    x = np.atleast_2d(x)
    if (x.shape[1] > x.shape[0]):
        x = x.T

    # trimming the samples with a non-defined speed
    cond = (t > speedTime[0]) & (t < speedTime[-1])
    x = x[cond, :]
    t = t[cond]

    # interpolation of speed
    interpSpeedFun = interp1d(speedTime, speed, axis=0)
    interpSpeed = interpSpeedFun(t)

    # phase in cycle units
    phase = np.cumsum(interpSpeed) * (t[1] - t[0])
    phase -= phase[0]
    # avoiding aliasing by sampling at least as fast as previously
    minDphase = np.min(speed) * (t[1] - t[0])
    # finding the increment for a fast fft calculation -
    # considers having aggregation points at round shaft speed orders
    dPhase = phase[-1] / next_fast_len(np.ceil(phase[-1] / minDphase).astype(int))
    resampledPhase = np.arange(0, phase[-1], dPhase)
    interpFun = interp1d(phase, x, axis=0)
    resampledX = interpFun(resampledPhase)

    return resampledPhase, np.squeeze(resampledX), t, phase, interpSpeed, np.squeeze(x)


def getTestCase():
    fs = 10 ** 4  # samples / sec
    T = 4  # sec
    meanSpeed = 10  # cycles / sec
    additionalSpeed = 2  # cycles / sec

    t = np.arange(0, T, 1 / fs)
    speedVarPhase = t * 2 * np.pi / T
    speedVariation = (1 - np.cos(speedVarPhase)) * 0.5
    speed = meanSpeed + additionalSpeed * speedVariation  # cycles / sec
    phase = np.cumsum(speed) / fs
    xMagnetic = np.sin(phase * 2 * np.pi)
    res = {
        'fs': fs,
        'time': t,
        'speed': speed,
        'xMagnetic': xMagnetic,
        'phase': phase
    }
    return res


def testGetSpeed():
    params = getTestCase()
    speedEstimated, timeEstimated = getSpeed(params['xMagnetic'], params['fs'])
    interpSpeedFun = interp1d(params['time'], params['speed'])
    speedRef = interpSpeedFun(timeEstimated)
    relErr = np.median(np.abs(speedRef - speedEstimated)) / speedRef.mean()
    assert (relErr < 10**-3)


def testOrderTrack():
    expectedDFT_Amp = 0.5  # only positive frequencies
    expectedShaftSpeed = 1
    params = getTestCase()

    speedEstimated, timeEstimated = getSpeed(params['xMagnetic'], params['fs'])
    resampledPhase, resampledMagnetic, t, phase, interpSpeed, x = orderTrack(
        params['xMagnetic'], params['time'], speedEstimated, timeEstimated
    )

    f = fftfreq(resampledMagnetic.shape[0], d=(resampledPhase[1] - resampledPhase[0]))
    dft = fft(resampledMagnetic, n=resampledMagnetic.shape[0], norm="forward")
    positiveCond = f > 0
    f = f[positiveCond]
    dft = dft[positiveCond]
    dft = np.abs(dft)
    err = np.abs(expectedDFT_Amp - dft[np.argmin(np.abs(f - expectedShaftSpeed))])
    assert (err < 0.0005)


if __name__ == '__main__':
    testGetSpeed()
    testOrderTrack()
