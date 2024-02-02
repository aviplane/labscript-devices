import math
from ctypes import *

import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from .py_header.regs import *

#
# **************************************************************************
# szTypeToName: doing name translation
# **************************************************************************
#


def szTypeToName(lCardType):
    sName = ''
    lVersion = (lCardType & TYP_VERSIONMASK)
    if (lCardType & TYP_SERIESMASK) == TYP_M2ISERIES:
        sName = 'M2i.%04x' % lVersion
    elif (lCardType & TYP_SERIESMASK) == TYP_M2IEXPSERIES:
        sName = 'M2i.%04x-Exp' % lVersion
    elif (lCardType & TYP_SERIESMASK) == TYP_M3ISERIES:
        sName = 'M3i.%04x' % lVersion
    elif (lCardType & TYP_SERIESMASK) == TYP_M3IEXPSERIES:
        sName = 'M3i.%04x-Exp' % lVersion
    elif (lCardType & TYP_SERIESMASK) == TYP_M4IEXPSERIES:
        sName = 'M4i.%04x-x8' % lVersion
    elif (lCardType & TYP_SERIESMASK) == TYP_M4XEXPSERIES:
        sName = 'M4x.%04x-x4' % lVersion
    elif (lCardType & TYP_SERIESMASK) == TYP_M2PEXPSERIES:
        sName = 'M2p.%04x-x4' % lVersion
    else:
        sName = 'unknown type'
    return sName


#
# **************************************************************************
# pvAllocMemPageAligned: creates a buffer for DMA that's page-aligned
# **************************************************************************
#
def pvAllocMemPageAligned(qwBytes):
    dwAlignment = 4096
    dwMask = dwAlignment - 1

    # allocate non-aligned, slightly larger buffer
    qwRequiredNonAlignedBytes = qwBytes * sizeof(c_char) + dwMask
    pvNonAlignedBuf = (c_char * qwRequiredNonAlignedBytes)()

    # get offset of next aligned address in non-aligned buffer
    misalignment = addressof(pvNonAlignedBuf) & dwMask
    if misalignment:
        dwOffset = dwAlignment - misalignment
    else:
        dwOffset = 0
    return (c_char * qwBytes).from_buffer(pvNonAlignedBuf, dwOffset)

##### Time conversion functions ###############################################


def time_s_to_c(t, clock_freq, extend=False):
    """
    Convert from time in seconds to time in sample chunks (1 chunk = 32 samples)
    clock_freq is in Hz. If extend is false, the function rounds down. If True,
    it rounds up to an integer number of chunks. Also returns the residual time
    delta in units of samples
    """
    if extend:
        func = math.ceil
    else:
        func = math.floor

    t_c = int(
        np.round(
        func(float(t/32 * clock_freq)),
        4
        ))  # chunks
    t_c = int(np.round(t/32* clock_freq, 4))
    #print(t, t/32, t/32* clock_freq, np.round(t/32* clock_freq, 3), int(np.round(t/32* clock_freq, 3)))
#    delta = np.abs(time_c_to_s(t_c, clock_freq) - t) #seconds
    #t_c = t
    delta = np.abs(t_c * 32 - time_s_to_sa(t, clock_freq))  #samples
    return t_c, delta


def time_c_to_s(t, clock_freq):
    """
    Convert form time in sample chunks to time in seconds (1 chunk = 32 samples)
    clock_freq is in Hz
    """
    return float(t * 32.0) / float(clock_freq)


def time_s_to_sa(t, clock_freq):
    """
    Convert from time in seconds to time in samples, rounding down
    """
    return math.floor(np.round(
                        clock_freq/1e6 * t * 1e6, 5)
                      )


def time_sa_to_s(t, clock_freq):
    """
    convert from time in samples to time in seconds
    """
    return float(t / clock_freq)

##### Output verification functions ###########################################


def check_groups_and_instructions(waveform_groups, sequence_instrs, clock_freq,
                                  dummy_loop_dur):
    for group in waveform_groups:
        print('--------------')
        print(time_c_to_s(group.time, clock_freq))
        print(group.waveforms)
        print(time_c_to_s(group.time + group.duration * group.loops, clock_freq))

    for instr in sequence_instrs:
        print('**************')
        print(instr.step)
        print(instr.next_step)
        print(instr.segment)
        print(instr.loops)
        if instr.segment == 0:
            print(time_c_to_s(instr.loops * dummy_loop_dur, clock_freq))


def draw_waveform_groups(waveform_groups, clock_freq, ax):

    segm_rects = []
    group_rects = []

    last_t = 0

    for i, group in enumerate(waveform_groups):

        for k in range(group.loops):
            group_rect = Rectangle((time_c_to_s(group.time + k * group.duration, clock_freq), 0),
                                   time_c_to_s(group.duration, clock_freq), 4)
            group_rects.append(group_rect)

            if k == 0:
                text = 'group ' + str(i)
                if group.loops > 1:
                    text += (' (x' + str(group.loops) + ')')
                ax.text(time_c_to_s(group.time + (group.loops * group.duration * 0.5), clock_freq), -0.2, text, ha='center')
                ax.axvline(time_c_to_s(group.time + k * group.duration, clock_freq), color='k')

            if group.time + (k + 1) * group.duration > last_t:
                last_t = group.time + (k + 1) * group.duration

            for j, waveform in enumerate(group.waveforms):  # TODO: update this with delta start/end
                for m in range(waveform.loops):
                    segm_rect = Rectangle((time_c_to_s(waveform.time + k * group.duration + m * waveform.duration,
                                                       clock_freq), waveform.port), time_c_to_s(waveform.duration, clock_freq), 1)
                    segm_rects.append(segm_rect)

                    if k == 0 and m == 0:
                        s_text = 's' + str(j)
                        if waveform.loops > 1:
                            s_text += (' (x' + str(waveform.loops) + ')')
                        ax.text(time_c_to_s(waveform.time + (0.5 * waveform.loops * waveform.duration),
                                            clock_freq), waveform.port + 0.5, s_text, ha='center')

    # Create patch collection with specified colour/alpha
    group_pc = PatchCollection(group_rects, facecolor='lightgray', alpha=1, edgecolor='gray')
    segm_pc = PatchCollection(segm_rects, facecolor='r', alpha=0.5, edgecolor='r')

    # Add collection to axes
    ax.add_collection(group_pc)
    ax.add_collection(segm_pc)

    ax.set_xlim(0, time_c_to_s(last_t * 1.02, clock_freq))
    ax.set_ylim(-0.5, 4.2)

    ax.set_yticks([0.5, 1.5, 2.5, 3.5], minor=True)
    ax.set_yticklabels([0, 1, 2, 3], minor=True)
    ax.set_yticklabels(['', '', '', '', '', '', '', '', '', ''], minor=False)
    ax.tick_params(axis='y', which='minor', length=0)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Channel')
    ax.title('Waveform Groups')

    return


def draw_sequence_plot(waveform_groups, sequence_instrs, clock_freq, ax):

    segm_rects = []
    group_rects = []
    dummy_group_rects = []

    last_t = 0

    seq_index = 0
    cur_time = 0
    while True:
        seq_instr = sequence_instrs[seq_index]
        group = waveform_groups[seq_instr.segment]

        if group.waveforms == 'dummy' and seq_instr.segment == 0:  # TODO: update this with delta start/end
            group_rect = Rectangle((time_c_to_s(cur_time, clock_freq), 0),
                                   time_c_to_s(group.duration * seq_instr.loops, clock_freq), 4)
            dummy_group_rects.append(group_rect)

            text = 'group ' + str(seq_instr.segment)
            if seq_instr.loops > 1:
                text += (' (x' + str(seq_instr.loops) + ')')
            ax.text(time_c_to_s(cur_time + (seq_instr.loops * group.duration * 0.5), clock_freq), -0.2, text, ha='center')
            ax.axvline(time_c_to_s(cur_time, clock_freq), color='k')

            cur_time += group.duration * seq_instr.loops

        else:
            for k in range(seq_instr.loops):
                group_rect = Rectangle((time_c_to_s(cur_time, clock_freq), 0),
                                       time_c_to_s(group.duration, clock_freq), 4)
                group_rects.append(group_rect)

                if k == 0:
                    text = 'group ' + str(seq_instr.segment)
                    if seq_instr.loops > 1:
                        text += (' (x' + str(seq_instr.loops) + ')')
                    ax.text(time_c_to_s(cur_time + (seq_instr.loops * group.duration * 0.5),
                                        clock_freq), -0.2, text, ha='center')
                    ax.axvline(time_c_to_s(cur_time, clock_freq), color='k')

                if cur_time + group.duration > last_t:
                    last_t = cur_time + group.duration

                if group.waveforms != 'dummy':
                    for j, waveform in enumerate(group.waveforms):
                        for m in range(waveform.loops):
                            segm_rect = Rectangle((time_c_to_s(cur_time + waveform.time - group.time + m * waveform.duration,
                                                               clock_freq), waveform.port), time_c_to_s(waveform.duration, clock_freq), 1)
                            segm_rects.append(segm_rect)

                            if k == 0 and m == 0:
                                s_text = 's' + str(j)
                                if waveform.loops > 1:
                                    s_text += (' (x' + str(waveform.loops) + ')')
                                ax.text(time_c_to_s(waveform.time + (0.5 * waveform.loops * waveform.duration),
                                                    clock_freq), waveform.port + 0.5, s_text, ha='center')

            cur_time += group.duration

        seq_index = seq_instr.next_step
        if seq_index == 0:
            break

    print('Drawing seq plot')

    # Create patch collection with specified colour/alpha
    group_pc = PatchCollection(group_rects, facecolor='lightgray', alpha=1, edgecolor='gray')
    dummy_group_pc = PatchCollection(dummy_group_rects, facecolor='gray', alpha=1, edgecolor='gray')
    segm_pc = PatchCollection(segm_rects, facecolor='r', alpha=0.5, edgecolor='r')

    # Add collection to axes
    ax.add_collection(group_pc)
    ax.add_collection(dummy_group_pc)
    ax.add_collection(segm_pc)

    ax.set_xlim(0, time_c_to_s(last_t * 1.02, clock_freq))
    ax.set_ylim(-0.5, 4.2)

    ax.set_yticks([0.5, 1.5, 2.5, 3.5], minor=True)
    ax.set_yticklabels([0, 1, 2, 3], minor=True)
    ax.set_yticklabels(['', '', '', '', '', '', '', '', '', ''], minor=False)
    ax.tick_params(axis='y', which='minor', length=0)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Channel')
    ax.title('Sequence')
    return
