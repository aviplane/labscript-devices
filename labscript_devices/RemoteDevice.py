from __future__ import division, unicode_literals, print_function, absolute_import

from labscript_utils import check_version
check_version('labscript', '2.0.1', '4')
check_version('zprocess', '2.4.8', '4')
# # from labscript_utils import PY2
# if PY2:
#     str = unicode
import labscript_utils.h5_lock

from labscript_devices import BLACS_tab
from labscript import Device, LabscriptError, set_passed_properties
import numpy as np
from enum import Enum


class RemoteDevice(Device):
    description = 'Interface for setting picomotor values'
    @set_passed_properties(
        property_names = {
            "connection_table_properties": ["BIAS_port"],
        }
    )
    def __init__(self, name,  BIAS_port, **kwargs):
        Device.__init__(self, name, **kwargs)
        self.BLACS_connection = BIAS_port

    def do_checks(self):
        pass

    def generate_code(self, hdf5_file):
        group = self.init_device_group(hdf5_file)
        pass


import os

from qtutils.qt.QtCore import *
from qtutils.qt.QtGui import *

from blacs.tab_base_classes import Worker, define_state
from blacs.tab_base_classes import MODE_MANUAL, MODE_TRANSITION_TO_BUFFERED, MODE_TRANSITION_TO_MANUAL, MODE_BUFFERED

from blacs.device_base_class import DeviceTab

from qtutils import UiLoader
import qtutils.icons

@BLACS_tab
class RemoteDeviceTab(DeviceTab):
    def initialise_GUI(self):
        layout = self.get_tab_layout()
        ui_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'camera.ui')
        self.ui = UiLoader().load(ui_filepath)
        layout.addWidget(self.ui)

        port = int(self.settings['connection_table'].find_by_name(self.settings["device_name"]).BLACS_connection)
        self.ui.port_label.setText(str(port))

        self.ui.check_connectivity_pushButton.setIcon(QIcon(':/qtutils/fugue/arrow-circle'))

        self.ui.host_lineEdit.returnPressed.connect(self.update_settings_and_check_connectivity)
        self.ui.use_zmq_checkBox.toggled.connect(self.update_settings_and_check_connectivity)
        self.ui.check_connectivity_pushButton.clicked.connect(self.update_settings_and_check_connectivity)

    def get_save_data(self):
        return {'host': str(self.ui.host_lineEdit.text()), 'use_zmq': self.ui.use_zmq_checkBox.isChecked()}

    def restore_save_data(self, save_data):
        if save_data:
            host = save_data['host']
            self.ui.host_lineEdit.setText(host)
            if 'use_zmq' in save_data:
                use_zmq = save_data['use_zmq']
                self.ui.use_zmq_checkBox.setChecked(use_zmq)
        else:
            self.logger.warning('No previous front panel state to restore')

        # call update_settings if primary_worker is set
        # this will be true if you load a front panel from the file menu after the tab has started
        if self.primary_worker:
            self.update_settings_and_check_connectivity()

    def initialise_workers(self):
        worker_initialisation_kwargs = {'port': self.ui.port_label.text()}
        self.create_worker("main_worker", RemoteDeviceWorker, worker_initialisation_kwargs)
        self.primary_worker = "main_worker"
        self.update_settings_and_check_connectivity()

    @define_state(MODE_MANUAL, queue_state_indefinitely=True, delete_stale_states=True)
    def update_settings_and_check_connectivity(self, *args):
        icon = QIcon(':/qtutils/fugue/hourglass')
        pixmap = icon.pixmap(QSize(16, 16))
        status_text = 'Checking...'
        self.ui.status_icon.setPixmap(pixmap)
        self.ui.server_status.setText(status_text)
        kwargs = self.get_save_data()
        responding = yield(self.queue_work(self.primary_worker, 'update_settings_and_check_connectivity', **kwargs))
        self.update_responding_indicator(responding)

    def update_responding_indicator(self, responding):
        if responding:
            icon = QIcon(':/qtutils/fugue/tick')
            pixmap = icon.pixmap(QSize(16, 16))
            status_text = 'Server is responding'
        else:
            icon = QIcon(':/qtutils/fugue/exclamation')
            pixmap = icon.pixmap(QSize(16, 16))
            status_text = 'Server not responding'
        self.ui.status_icon.setPixmap(pixmap)
        self.ui.server_status.setText(status_text)


class RemoteDeviceWorker(Worker):
    def init(self):
        global socket; import socket
        global zmq; import zmq
        global zprocess; import zprocess
        global shared_drive; import labscript_utils.shared_drive as shared_drive

        self.host = ''
        self.use_zmq = False

    def update_settings_and_check_connectivity(self, host, use_zmq):
        self.host = host
        self.use_zmq = use_zmq
        if not self.host:
            return False
        if not self.use_zmq:
            return self.initialise_sockets(self.host, self.port)
        else:
            response = zprocess.zmq_get_string(self.port, self.host, data='hello')
            if response == 'hello':
                return True
            else:
                raise Exception('invalid response from server: ' + str(response))

    def initialise_sockets(self, host, port):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        assert port, 'No port number supplied.'
        assert host, 'No hostname supplied.'
        assert str(int(port)) == port, 'Port must be an integer.'
        s.settimeout(10)
        s.connect((host, int(port)))
        s.send(b'hello\r\n')
        response = s.recv(1024).decode('utf8')
        s.close()
        if 'hello' in response:
            return True
        else:
            raise Exception('invalid response from server: ' + response)

    def transition_to_buffered(self, device_name, h5file, initial_values, fresh):
        h5file = shared_drive.path_to_local(h5file)
        if not self.use_zmq:
            return self.transition_to_buffered_sockets(h5file,self.host, self.port)
        response = zprocess.zmq_get_string(self.port, self.host, data=h5file)
        if response != 'ok':
            raise Exception('invalid response from server: ' + str(response))
        response = zprocess.zmq_get_string(self.port, self.host, timeout = 10)
        if response != 'done':
            raise Exception('invalid response from server: ' + str(response))
        return {} # indicates final values of buffered run, we have none

    def transition_to_buffered_sockets(self, h5file, host, port):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(120)
        s.connect((host, int(port)))
        s.send(b'%s\r\n' % h5file.encode('utf8'))
        response = s.recv(1024).decode('utf8')
        if not 'ok' in response:
            s.close()
            raise Exception(response)
        response = s.recv(1024).decode('utf8')
        if not 'done' in response:
            s.close()
            raise Exception(response)
        return {} # indicates final values of buffered run, we have none

    def transition_to_manual(self):
        if not self.use_zmq:
            return self.transition_to_manual_sockets(self.host, self.port)
        response = zprocess.zmq_get_string(self.port, self.host, 'done')
        if response != 'ok':
            raise Exception('invalid response from server: ' + str(response))
        response = zprocess.zmq_get_string(self.port, self.host, timeout = 30)
        if response != 'done':
            raise Exception('invalid response from server: ' + str(response))
        return True # indicates success

    def transition_to_manual_sockets(self, host, port):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(120)
        s.connect((host, int(port)))
        s.send(b'done\r\n')
        response = s.recv(1024).decode('utf8')
        if response != 'ok\r\n':
            s.close()
            raise Exception(response)
        response = s.recv(1024).decode('utf8')
        if not 'done' in response:
            s.close()
            raise Exception(response)
        return True # indicates success

    def abort_buffered(self):
        return self.abort()

    def abort_transition_to_buffered(self):
        return self.abort()

    def abort(self):
        if not self.use_zmq:
            return self.abort_sockets(self.host, self.port)
        response = zprocess.zmq_get_string(self.port, self.host, 'abort')
        if response != 'done':
            raise Exception('invalid response from server: ' + str(response))
        return True # indicates success

    def abort_sockets(self, host, port):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(120)
        s.connect((host, int(port)))
        s.send(b'abort\r\n')
        response = s.recv(1024).decode('utf8')
        if not 'done' in response:
            s.close()
            raise Exception(response)
        return True # indicates success

    def program_manual(self, values):
        return {}

    def shutdown(self):
        return
