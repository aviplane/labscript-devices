import ctypes
import PyDAQmx as daqmx
import PyDAQmx.DAQmxConstants as c
import PyDAQmx.DAQmxTypes as types

"""This file is distinct from utils.py as it requires PyDAQmx to be installed,
whereas the contents of utils.py do not."""


def get_devices():
    BUFSIZE = 4096
    result = ctypes.create_string_buffer(BUFSIZE)
    daqmx.DAQmxGetSysDevNames(result, types.uInt32(BUFSIZE))
    return result.value.decode('utf8').split(',')


def get_product_type(device_name):
    BUFSIZE = 4096
    result = ctypes.create_string_buffer(BUFSIZE)
    daqmx.DAQmxGetDevProductType(device_name, result, types.uInt32(BUFSIZE))
    return result.value.decode('utf8')


def get_CI_chans(device_name):
    BUFSIZE = 4096
    result = ctypes.create_string_buffer(BUFSIZE)
    daqmx.DAQmxGetDevCIPhysicalChans(device_name, result, types.uInt32(BUFSIZE))
    return result.value.decode('utf8').split(', ')


def is_simulated(device_name):
    result = types.bool32()
    daqmx.DAQmxGetDevIsSimulated(device_name, result)
    return result.value


def supports_period_measurement(device_name):
    import warnings

    with warnings.catch_warnings():
        # PyDAQmx warns about a positive return value, but actually this is how you are
        # supposed to figure out the size of the array required.
        warnings.simplefilter("ignore")
        # Pass in null pointer and 0 len to ask for what array size is needed:
        npts = daqmx.DAQmxGetDevCISupportedMeasTypes(device_name, types.int32(), 0)
    # Create that array
    result = (types.int32 * npts)()
    daqmx.DAQmxGetDevCISupportedMeasTypes(device_name, result, npts)
    return c.DAQmx_Val_Period in [result[i] for i in range(npts)]


def incomplete_sample_detection(device_name):
    """Introspect whether a device has 'incomplete sample detection', described here:

        www.ni.com/documentation/en/ni-daqmx/latest/devconsid/incompletesampledetection/

    The result is determined empirically by outputting a pulse on one counter and
    measuring it on another, and seeing whether the first sample was discarded or not.
    This requires a non-simulated device with at least two counters. No external signal
    is actually generated by the device whilst this test is performed. Credit for this
    method goes to Kevin Price, who provided it here:

        forums.ni.com/t5/Multifunction-DAQ/_/td-p/3849429

    This workaround will hopefully be deprecated if and when NI provides functionality
    to either inspect this feature's presence directly, or to disable it regardless of
    its presence.
    """

    if is_simulated(device_name):
        msg = "Can only detect incomplete sample detection on non-simulated devices"
        raise ValueError(msg)

    if not supports_period_measurement(device_name):
        msg = "Device doesn't support period measurement"
        raise ValueError(msg)

    CI_chans = get_CI_chans(device_name)

    if len(CI_chans) < 2:
        msg = "Need at least two counters to detect incomplete sample detection"
        raise ValueError(msg)

    # The counter we will produce a test signal on:
    out_chan = CI_chans[0]
    # The counter we will measure it on:
    meas_chan = CI_chans[1]

    # Set up the output task:
    out_task = daqmx.Task()
    out_task.CreateCOPulseChanTime(
        out_chan, "", c.DAQmx_Val_Seconds, c.DAQmx_Val_Low, 0, 1e-3, 1e-3
    )
    # Prevent the signal being output on the physical terminal:
    out_task.SetCOPulseTerm("", "")
    # Force CO into idle state to prevent spurious edges when the task is started:
    out_task.TaskControl(c.DAQmx_Val_Task_Commit)

    # Set up the measurement task
    meas_task = daqmx.Task()
    meas_task.CreateCIPeriodChan(
        meas_chan,
        "",
        1e-3,
        1.0,
        c.DAQmx_Val_Seconds,
        c.DAQmx_Val_Rising,
        c.DAQmx_Val_LowFreq1Ctr,
        10.0,
        0,
        "",
    )
    meas_task.CfgImplicitTiming(c.DAQmx_Val_ContSamps, 1)
    # Specify that we are measuring the internal output of the other counter:
    meas_task.SetCIPeriodTerm("", '/' + out_chan + 'InternalOutput')

    try:
        meas_task.StartTask()
        out_task.StartTask()
        out_task.WaitUntilTaskDone(10.0)
        # How many samples are in the read buffer of the measurement task?
        samps_avail = types.uInt32()
        meas_task.GetReadAvailSampPerChan(samps_avail)
        if samps_avail.value == 0:
            # The device discarded the first edge
            return True
        elif samps_avail.value == 1:
            # The device did not discard the first edge
            return False
        else:
            # Unexpected result
            msg = "Unexpected number of samples: %d" % samps_avail.value
            raise ValueError(msg)
    finally:
        out_task.ClearTask()
        meas_task.ClearTask()


if __name__ == '__main__':
    # List whether attached devices have incomplete sample detection:
    print('Device    '.rjust(16), 'Incomplete sample detection')
    for name in get_devices():
        if not is_simulated(name):
            model = get_product_type(name)
            print((model + '    ').rjust(16), end=' ')
            try:
                result = incomplete_sample_detection(name)
            except ValueError as e:
                result = str(e)
            print(result)
