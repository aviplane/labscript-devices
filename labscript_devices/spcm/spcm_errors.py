# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 11:55:59 2020

@author: Quantum Engineer
"""

from . import pyspcm as sp
from .spcm_tools import szTypeToName
from .py_header import regs
            

def decorate_functions(module, decorator):
    function_list = ['spcm_dwDefTransfer_i64',
                     'spcm_dwGetContBuf_i64',
                     'spcm_dwGetParam_i32',
                     'spcm_dwGetParam_i64',
                     'spcm_dwInvalidateBuf',
                     'spcm_dwSetParam_i32',
                     'spcm_dwSetParam_i64',
                     'spcm_dwSetParam_i64m']
    
    for name in dir(module):
        if name in function_list:
            obj = getattr(module, name)
            setattr(module, name, decorator(obj))


def error_decorator(func):
    def wrapper(*args, **kwargs):
        try:
            error_checking = kwargs.pop('error_checking')
        except:
            error_checking = True
        
        err = func(*args, **kwargs)
        
        if err and error_checking:            
            card = args[0]
            
            error_buffer = sp.create_string_buffer(sp.ERRORTEXTLEN)
            sp.spcm_dwGetErrorInfo_i32 (card, None, None, error_buffer)            
            error_string = error_buffer.value.decode()
            
            name = func.__name__
            
            if ('set' in name) or ('get' in name):
                target = identify_register(args[1])
            else:
                target = ''
            
            sp.spcm_dwSetParam_i32(card, sp.SPC_M2CMD, sp.M2CMD_CARD_STOP | sp.M2CMD_DATA_STOPDMA, error_checking=False)
            sp.spcm_vClose(card)
            raise RuntimeError('Spectrum error {} during call to {}({}): \n {}'.format(err, name, target, error_string))
            
        return err
        
    return wrapper


def identify_register(val):
    for name in dir(regs):
        register = getattr(regs, name)
        if register == val:
            return name


# def error_decorator(func, error=RuntimeError):
#     def wrapper(*args, **kwargs):
#         err = func(*args, **kwargs)
#         if err:
#             raise error("SPCM ERROR {}: {}".format(err, SPCM_ERRORS[err]))
            
#     return wrapper

#not sure if we can use this, numbers don't seem to match up with spcm_dwGetErrorInfo_i32
SPCM_ERRORS = {0: "Execution OK, no error",
               1: "An error occurred when initializing the given card. Either the card has already been opened by another process or a hardware error occurred.",
               3: "Initialization only: The type of board is unknown. This is a critical error. Please check whether the board is correctly plugged in the slot and whether you have the latest driver version.",
               4: "This function is not supported by the hardware version.",
               5: "The board index re map table in the registry is wrong. Either delete this table or check it carefully for double values",
               6: "The version of the kernel driver is not matching the version of the DLL. Please do a complete re-installation of the hardware driver. This error normally only occurs if someone copies the driver library and the kernel driver manually. ",
               7: "The hardware needs a newer driver version to run properly. Please install the driver that was delivered together withthe card.",
               8: "One of the address ranges is disabled (fatal error), can only occur under Linux",
               9: "The used handle is not valid.",
               10: "A card with the given name has not been found",
               11: "A card with given name is already in use by another application",
               12: "Express hardware version not able to handle 64 bit addressing -> update needed.",
               13: "Firmware versions of synchronized cards or for this driver do not match -> update needed.",
               14: "Synchronization protocol versions of synchronized cards do not match -> update needed.",
               16: "Old error waiting to be read. Please read the full error information before proceeding. The driver is locked until the error information has been read.",
               17: "Board is already used by another application. It is not possible to use one hardware from two different programs at the same time",
               32: "Abort of wait function. This return value just tells that the function has been aborted from another thread. The driver library is not locked if this error occurs.",
               48: "The card is already in access and therefore locked by another process. It is not possible to access one card through multiple processes. Only one process can access a specific card at the time.",
               50: "The device is mapped to an invalid device. The device mapping can be accessed via the Control Center",
               64: "The network setup of a digitizerNETBOX has failed.",
               65: "The network data transfer from/to a digitizerNETBOX has failed",
               66: "Power cycle (PC off/on) is needed to update the card's firmware (a simple OS reboot is not sufficient !)",
               67: "A network timeout has occurred.",
               68: "The buffer size is not sufficient (too small).",
               69: "The access to the card has been intentionally restricted",
               70: "An invalid parameter has been used for a certain function.",
               71: "The temperature of at least one of the card’s sensors measures a temperature, that is too high for the hardware. ",
               256: "The register is not valid for this type of board",
               257: " The value for this register is not in a valid range. The allowed values and ranges are listed in the board specific documentation.",
               258: " Feature (option) is not installed on this board. It’s not possible to access this feature if it’s not installed. ",
               259: "Command sequence is not allowed. Please check the manual carefully to see which command sequences are possible.",
               260: "Data read is not allowed after aborting the data acquisition.",
               261: " Access to this register is denied. This register is not accessible for users",
               263: "",
               264: "",
               265: "",
               266: "",
               267: "",
               268: "",
               269: "",
               270: "",
               271: "",
               272: "",
               273: "",
               288: "",
               304: "",
               320: "",
               321: "",
               322: "",
               323: "",
               324: "",
               325: "",
               326: "",
               327: "",
               328: "",
               329: "",
               330: "",
               331: "",
               332: "",
               515: "",
               517: "",
               518: "",
               519: "",
               520: "",
               769: "",
               770: "",
               784: "",
               800: "",
               65535: "",}


if __name__ == '__main__':
    decorate_functions(sp, error_decorator)
    
    card = sp.spcm_hOpen(sp.create_string_buffer(b'TCPIP[0]::171.64.56.18::inst1::INSTR'))
    
    card_type = sp.int32(0)
    err = sp.spcm_dwGetParam_i32(card, sp.SPC_PCITYP, sp.byref(card_type))
    if err:
        raise LabscriptError('Error code {}'.format(err))
    
    print(szTypeToName(card_type.value))
    
    sp.spcm_vClose(card)
