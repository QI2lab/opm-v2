'''
QtWidget for setting the OPM configuration.

2025/03/07 Sheppard: Initial setup
'''
import sys
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (QWidget, QApplication, QDoubleSpinBox, 
                             QHBoxLayout, QVBoxLayout, QGroupBox, 
                             QLabel, QComboBox, QSlider, QCheckBox, QSpinBox)
from pathlib import Path
import json

MIN_AO_POSITIONS = 1
MAX_AO_POSITIONS = 20
MAX_AO_ITERATIONS = 10
MAX_STAGE_RANGE = 1000
class OPMSettings(QWidget):
    
    settings_changed = pyqtSignal()
    
    def __init__(self,
                 config_path: Path):

        super().__init__()
        
        self.config_path = config_path
        with open(self.config_path, 'r') as config_file:
            config = json.load(config_file)
        
        self.config = config
        self.widgets = {}       
        self.create_ui()
        self.update_config()

    def create_spinbox(
        self,
        value: int = 0,
        min: int = 0,
        max: int = 100,
        width: int = 80,
        connect_to_fn = None
    ):
        spbx = QSpinBox()
        spbx.setRange(min, max)
        spbx.setValue(value)
        spbx.setFixedWidth(width)
        spbx.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if isinstance(connect_to_fn, list):
            for fn in connect_to_fn:
                spbx.valueChanged.connect(fn)
        elif connect_to_fn:
            spbx.valueChanged.connect(connect_to_fn)
        
        return spbx
    
    def create_combobox(
        self,
        items: list,
        width: int = 80,
        connect_to_fn = None
    ):
        cmbx =  QComboBox()
        cmbx.addItems(items)
        cmbx.setFixedWidth(width)
        if isinstance(connect_to_fn, list):
            for fn in connect_to_fn:
                cmbx.currentIndexChanged.connect(fn)
        elif connect_to_fn:
            cmbx.currentIndexChanged.connect(connect_to_fn)
        return cmbx
    
    def create_slider(
        self,
        value: int = 0,
        min: int = 0,
        max: int = 100,
        interval: int = 1,
        connect_to_fn = None
    ):
        sldr = QSlider(Qt.Orientation.Horizontal)
        sldr.setValue(int(value))
        sldr.setRange(min, max)
        sldr.setTickInterval(interval)
        if isinstance(connect_to_fn, list):
            for fn in connect_to_fn:
                sldr.valueChanged.connect(fn)
        elif connect_to_fn:
            sldr.valueChanged.connect(connect_to_fn)
        return sldr
    
    def create_dbspinbox(
        self,
        value: int = 0,
        min: int = 0,
        max: int = 100,
        precision: int = 0,
        interval: int = 1,
        width: int = 80,
        connect_to_fn = None
    ):
        dbspbx = QDoubleSpinBox()
        dbspbx.setRange(min, max)
        dbspbx.setDecimals(precision)
        dbspbx.setSingleStep(interval)
        dbspbx.setValue(value)
        dbspbx.setFixedWidth(width)
        dbspbx.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if isinstance(connect_to_fn, list):
            for fn in connect_to_fn:
                dbspbx.valueChanged.connect(fn)
        elif connect_to_fn:
            dbspbx.valueChanged.connect(connect_to_fn)
        return dbspbx
    
    def create_ui(self):

        #--------------------------------------------------------------------#
        # Create AO optimization settings
        #--------------------------------------------------------------------#
            
        self.cmbx_ao_active_channel =  self.create_combobox(self.config['OPM']['channel_ids'])
        self.layout_ao_active_channel = QHBoxLayout()
        self.layout_ao_active_channel.addWidget(QLabel('Active Channel'))
        self.layout_ao_active_channel.addWidget(self.cmbx_ao_active_channel)
        
        self.spbx_active_channel_power = self.create_dbspinbox(
            value = self.config['acq_config']['AO']['active_channel_power'],
            connect_to_fn = self.update_config
        )                
        self.layout_active_channel_power = QHBoxLayout()
        self.layout_active_channel_power.addWidget(QLabel('Laser power (%):'))
        self.layout_active_channel_power.addWidget(self.spbx_active_channel_power)
                
        self.spbx_ao_exposure = self.create_dbspinbox(
            min = 20,
            max = 1000,
            connect_to_fn=self.update_config
        )
        self.layout_ao_exposure = QHBoxLayout()
        self.layout_ao_exposure.addWidget(QLabel('Exposure (ms):'))
        self.layout_ao_exposure.addWidget(self.spbx_ao_exposure)

        self.spbx_ao_mirror_range = self.create_dbspinbox(
            value=int(self.config['acq_config']['AO']['image_mirror_range_um']),
            max=250,
            connect_to_fn=self.update_config
        )        
        self.layout_ao_mirror_range = QHBoxLayout()
        self.layout_ao_mirror_range.addWidget(QLabel('Projection/Image mirror range (\u00B5):'))
        self.layout_ao_mirror_range.addWidget(self.spbx_ao_mirror_range)
        
        self.cmbx_ao_mode = self.create_combobox(
            items=self.config['OPM']['ao_modes'],
            width=125,
            connect_to_fn=self.update_config
        )        
        self.layout_ao_mode = QHBoxLayout()
        self.layout_ao_mode.addWidget(QLabel('AO mode:'))
        self.layout_ao_mode.addWidget(self.cmbx_ao_mode)
                
        self.cmbx_ao_metric = self.create_combobox(
            items=self.config['OPM']['ao_metrics'],
            width=125,
            connect_to_fn=self.update_config
        )
        
        self.layout_ao_metric = QHBoxLayout()
        self.layout_ao_metric.addWidget(QLabel('Metric:'))
        self.layout_ao_metric.addWidget(self.cmbx_ao_metric)
        
        self.cmbx_ao_daq_mode =  self.create_combobox(
            items=self.config['OPM']['daq_modes'],
            width=125,
            connect_to_fn=self.update_config
        )
        self.layout_ao_daq_mode = QHBoxLayout()
        self.layout_ao_daq_mode.addWidget(QLabel('DAQ mode:'))
        self.layout_ao_daq_mode.addWidget(self.cmbx_ao_daq_mode)
        
        self.spbx_num_iterations = self.create_spinbox(
            value=self.config['acq_config']['AO']['num_iterations'],
            max=10,
            connect_to_fn=self.update_config
        )
        self.layout_num_iterations = QHBoxLayout()
        self.layout_num_iterations.addWidget(QLabel('Number of iterations:'))
        self.layout_num_iterations.addWidget(self.spbx_num_iterations)
        
        self.spbx_mode_delta = self.create_dbspinbox(
            value=self.config['acq_config']['AO']['mode_delta'],
            max=1.0,
            interval = 0.01,
            precision=2,
            connect_to_fn=self.update_config
        )
        self.layout_mode_delta = QHBoxLayout()
        self.layout_mode_delta.addWidget(QLabel('Initial mode delta:'))
        self.layout_mode_delta.addWidget(self.spbx_mode_delta)
        
        self.spbx_mode_alpha = self.create_dbspinbox(
            value=self.config['acq_config']['AO']['mode_alpha'],
            max=1.0,
            precision=2,
            interval=0.1,
            connect_to_fn=self.update_config
        )        
        self.layout_mode_alpha = QHBoxLayout()
        self.layout_mode_alpha.addWidget(QLabel('Mode range alpha:'))
        self.layout_mode_alpha.addWidget(self.spbx_mode_alpha)
        
        self.spbx_num_scan_positions = self.create_spinbox(
            value=MIN_AO_POSITIONS,
            min=MIN_AO_POSITIONS,
            max=MAX_AO_POSITIONS,
            connect_to_fn=self.update_config
        )        
        self.spbx_num_tile_positions = self.create_spinbox(
            value=MIN_AO_POSITIONS,
            min=MIN_AO_POSITIONS,
            max=MAX_AO_POSITIONS,
            connect_to_fn=self.update_config
        )
        self.layout_num_scan_positions = QHBoxLayout()
        self.layout_num_scan_positions.addWidget(QLabel('AO grid scan axis positions:'))
        self.layout_num_scan_positions.addWidget(self.spbx_num_scan_positions)
        self.layout_num_tile_positions = QHBoxLayout()
        self.layout_num_tile_positions.addWidget(QLabel('AO grid tile axis positions:'))
        self.layout_num_tile_positions.addWidget(self.spbx_num_tile_positions)
                
        #--------------------------------------------------------------------#
        # Populate AO group 
        #--------------------------------------------------------------------#
                
        self.group_ao_main = QGroupBox('AO optimization settings')
        self.layout_ao_main = QVBoxLayout()
        self.layout_ao_main.addLayout(self.layout_ao_mode)
        self.layout_ao_main.addLayout(self.layout_ao_metric)
        self.layout_ao_main.addLayout(self.layout_ao_daq_mode)
        self.layout_ao_main.addLayout(self.layout_ao_active_channel)
        self.layout_ao_main.addLayout(self.layout_active_channel_power)
        self.layout_ao_main.addLayout(self.layout_ao_exposure)     
        self.layout_ao_main.addLayout(self.layout_ao_mirror_range)
        self.layout_ao_main.addLayout(self.layout_num_iterations)
        self.layout_ao_main.addLayout(self.layout_mode_delta)
        self.layout_ao_main.addLayout(self.layout_mode_alpha)
        self.layout_ao_main.addLayout(self.layout_num_scan_positions)
        self.layout_ao_main.addLayout(self.layout_num_tile_positions)
        self.group_ao_main.setLayout(self.layout_ao_main)

        #--------------------------------------------------------------------#
        # Update widgets dict with new entries
        #--------------------------------------------------------------------#

        self.widgets.update(
            {'AO':{
                    'metric':self.cmbx_ao_metric,
                    'mode_delta': self.spbx_mode_delta,
                    'mode_alpha': self.spbx_mode_alpha,
                    'num_iterations': self.spbx_num_iterations,
                    'num_scan_positions': self.spbx_num_scan_positions,
                    'num_tile_positions': self.spbx_num_tile_positions,
                    'image_mirror_range_um': self.spbx_ao_mirror_range,
                    'active_channel_id': self.cmbx_ao_active_channel,
                    'active_channel_power': self.spbx_active_channel_power,
                    'exposure_ms': self.spbx_ao_exposure,
                    'ao_mode': self.cmbx_ao_mode
                    }  
                }
        )
        

        #--------------------------------------------------------------------#
        # Create imaging mode + options widgets
        #--------------------------------------------------------------------#
        
        self.group_opm_main = QGroupBox('OPM imaging settings')
        self.layout_opm_main = QVBoxLayout()
        
        self.cmbx_opm_mode = self.create_combobox(
            items=self.config['OPM']['imaging_modes'],
            width=125,
            connect_to_fn=self.update_config
        )
        self.layout_opm_mode = QHBoxLayout()
        self.layout_opm_mode.addWidget(QLabel('OPM mode:'))
        self.layout_opm_mode.addWidget(self.cmbx_opm_mode)        
        
        self.cmbx_o2o3_mode = self.create_combobox(
            items=self.config['OPM']['autofocus_frequencies'],
            width=125,
            connect_to_fn=self.update_config
        )
        self.layout_o2o3_mode = QHBoxLayout()
        self.layout_o2o3_mode.addWidget(QLabel('O2O3 Autofocus:'))
        self.layout_o2o3_mode.addWidget(self.cmbx_o2o3_mode)
                
        self.cmbx_fluidics_mode = self.create_combobox(
            items=self.config['OPM']['fluidics_modes'],
            width=125,
            connect_to_fn=self.update_config
        )
        self.layout_fluidics_mode = QHBoxLayout()
        self.layout_fluidics_mode.addWidget(QLabel('Fluidics rounds:'))
        self.layout_fluidics_mode.addWidget(self.cmbx_fluidics_mode)
        
        self.cmbx_laser_blanking = self.create_combobox(
            items=["on", "off"],
            connect_to_fn=self.update_config
        )
        self.layout_laser_blanking = QHBoxLayout()
        self.layout_laser_blanking.addWidget(QLabel('Laser blanking:'))
        self.layout_laser_blanking.addWidget(self.cmbx_laser_blanking)
        
        #--------------------------------------------------------------------#
        # Populate OPM mode settings group
        #--------------------------------------------------------------------#
        
        self.layout_opm_settings = QVBoxLayout()
        self.layout_opm_settings.addLayout(self.layout_opm_mode)
        self.layout_opm_settings.addLayout(self.layout_o2o3_mode)
        self.layout_opm_settings.addLayout(self.layout_fluidics_mode)
        self.layout_opm_settings.addLayout(self.layout_laser_blanking)
        self.group_opm_settings = QGroupBox('OPM modes/settings')
        self.group_opm_settings.setLayout(self.layout_opm_settings)
        
        #--------------------------------------------------------------------#
        # Create channel widgets
        #--------------------------------------------------------------------#
        
        self.sldr_405_power = self.create_slider(
            connect_to_fn=self.update_405_spbx
        )   
        self.spbx_405_power = self.create_dbspinbox(
            connect_to_fn=[
                self.update_405_slider,
                self.update_405_state]
        )
        self.spbx_405_exp = self.create_dbspinbox(
            max=1000,
            connect_to_fn=self.update_405_state
        )
        self.chx_405_state = QCheckBox()
        self.chx_405_state.setChecked(False)
        self.chx_405_state.checkStateChanged.connect(self.update_405_state)
        
        self.layout_405 = QHBoxLayout()
        self.layout_405.addWidget(QLabel('405nm:'))
        self.layout_405.addWidget(self.sldr_405_power)
        self.layout_405.addWidget(self.spbx_405_power)
        self.layout_405.addWidget(QLabel('%'))
        self.layout_405.addWidget(self.spbx_405_exp)
        self.layout_405.addWidget(QLabel('ms'))
        self.layout_405.addWidget(self.chx_405_state)
        
        self.sldr_488_power = self.create_slider(
            connect_to_fn=self.update_488_spbx
        )   
        self.spbx_488_power = self.create_dbspinbox(
            connect_to_fn=[
                self.update_488_slider,
                self.update_488_state]
        )
        self.spbx_488_exp = self.create_dbspinbox(
            max=1000,
            connect_to_fn=self.update_488_state
        )
        
        self.chx_488_state = QCheckBox()
        self.chx_488_state.setChecked(False)
        self.chx_488_state.checkStateChanged.connect(self.update_488_state)
        
        self.layout_488 = QHBoxLayout()
        self.layout_488.addWidget(QLabel('488nm:'))
        self.layout_488.addWidget(self.sldr_488_power)
        self.layout_488.addWidget(self.spbx_488_power)
        self.layout_488.addWidget(QLabel('%'))
        self.layout_488.addWidget(self.spbx_488_exp)
        self.layout_488.addWidget(QLabel('ms'))
        self.layout_488.addWidget(self.chx_488_state)
        
        self.sldr_561_power = self.create_slider(
            connect_to_fn=self.update_561_spbx
        )   
        self.spbx_561_power = self.create_dbspinbox(
            connect_to_fn=[
                self.update_561_slider,
                self.update_561_state]
        )
        self.spbx_561_exp = self.create_dbspinbox(
            max=1000,
            connect_to_fn=self.update_561_state
        )
        
        self.chx_561_state = QCheckBox()
        self.chx_561_state.setChecked(False)
        self.chx_561_state.checkStateChanged.connect(self.update_561_state)
        
        self.layout_561 = QHBoxLayout()
        self.layout_561.addWidget(QLabel('561nm:'))
        self.layout_561.addWidget(self.sldr_561_power)
        self.layout_561.addWidget(self.spbx_561_power)
        self.layout_561.addWidget(QLabel('%'))
        self.layout_561.addWidget(self.spbx_561_exp)
        self.layout_561.addWidget(QLabel('ms'))
        self.layout_561.addWidget(self.chx_561_state)
        
        self.sldr_638_power = self.create_slider(
            connect_to_fn=self.update_638_spbx
        )   
        self.spbx_638_power = self.create_dbspinbox(
            connect_to_fn=[
                self.update_638_slider,
                self.update_638_state]
        )
        self.spbx_638_exp = self.create_dbspinbox(
            max=1000,
            connect_to_fn=self.update_638_state
        )
        
        self.chx_638_state = QCheckBox()
        self.chx_638_state.setChecked(False)
        self.chx_638_state.checkStateChanged.connect(self.update_638_state)
        
        self.layout_638 = QHBoxLayout()
        self.layout_638.addWidget(QLabel('638nm:'))
        self.layout_638.addWidget(self.sldr_638_power)
        self.layout_638.addWidget(self.spbx_638_power)
        self.layout_638.addWidget(QLabel('%'))
        self.layout_638.addWidget(self.spbx_638_exp)
        self.layout_638.addWidget(QLabel('ms'))
        self.layout_638.addWidget(self.chx_638_state)
        
        self.sldr_705_power = self.create_slider(
            connect_to_fn=self.update_705_spbx
        )   
        self.spbx_705_power = self.create_dbspinbox(
            connect_to_fn=[
                self.update_705_slider,
                self.update_705_state]
        )
        self.spbx_705_exp = self.create_dbspinbox(
            max=1000,
            connect_to_fn=self.update_705_state
        )        
        self.chx_705_state = QCheckBox()
        self.chx_705_state.setChecked(False)
        self.chx_705_state.checkStateChanged.connect(self.update_705_state)
        
        self.layout_705 = QHBoxLayout()
        self.layout_705.addWidget(QLabel('705nm:'))
        self.layout_705.addWidget(self.sldr_705_power)
        self.layout_705.addWidget(self.spbx_705_power)
        self.layout_705.addWidget(QLabel('%'))
        self.layout_705.addWidget(self.spbx_705_exp)
        self.layout_705.addWidget(QLabel('ms'))
        self.layout_705.addWidget(self.chx_705_state)
        
        #--------------------------------------------------------------------#
        # Populate channels group
        #--------------------------------------------------------------------#
        
        self.layout_channels = QVBoxLayout()
        self.layout_channels.addLayout(self.layout_405)
        self.layout_channels.addLayout(self.layout_488)
        self.layout_channels.addLayout(self.layout_561)
        self.layout_channels.addLayout(self.layout_638)
        self.layout_channels.addLayout(self.layout_705)
        self.group_channels = QGroupBox('Channels')
        self.group_channels.setLayout(self.layout_channels)
        
        #--------------------------------------------------------------------#
        # Create acquisition position widgets
        #--------------------------------------------------------------------#
        
        self.spbx_mirror_image_range = self.create_dbspinbox(
            value=self.config['acq_config']['mirror_scan']['scan_range_um'],
            max=250,
            connect_to_fn=self.update_config
        )        
        self.layout_mirror_image_range = QHBoxLayout()
        self.layout_mirror_image_range.addWidget(QLabel('Mirror scan range (\u00B5m):'))
        self.layout_mirror_image_range.addWidget(self.spbx_mirror_image_range)

        self.spbx_scan_step_size = self.create_dbspinbox(
            value=self.config['acq_config']['stage_scan']['scan_step_size_um'],
            min=0.05,
            max=2.0,
            interval=0.1,
            precision=2,
            connect_to_fn=self.update_config
        )       
        self.layout_scan_step_size = QHBoxLayout()
        self.layout_scan_step_size.addWidget(QLabel('Scan step size (\u00B5m):'))
        self.layout_scan_step_size.addWidget(self.spbx_scan_step_size)
        
        self.spbx_stage_image_range = self.create_dbspinbox(
            value = MAX_STAGE_RANGE,
            max=MAX_STAGE_RANGE,
            connect_to_fn=self.update_config
        )        
        self.layout_stage_image_range = QHBoxLayout()
        self.layout_stage_image_range.addWidget(QLabel('Stage scan range (\u00B5m):'))
        self.layout_stage_image_range.addWidget(self.spbx_stage_image_range)

        self.spbx_stage_slope_x = self.create_dbspinbox(
            min=-0.10, 
            max=0.10,
            precision=4,
            interval=0.0001,
            connect_to_fn=self.update_config
        )
        self.spbx_stage_slope_y = self.create_dbspinbox(
            min=-0.10, 
            max=0.10,
            precision=4,
            interval=0.0001,
            connect_to_fn=self.update_config
        )
        self.layout_stage_slope = QHBoxLayout()
        self.layout_stage_slope.addWidget(QLabel('Coverslip slope (rise/run):'))
        self.layout_stage_slope.addStretch()
        self.layout_stage_slope.addWidget(QLabel('x:'))
        self.layout_stage_slope.addWidget(self.spbx_stage_slope_x)
        self.layout_stage_slope.addWidget(QLabel('y:'))
        self.layout_stage_slope.addWidget(self.spbx_stage_slope_y)
        
        self.spbx_proj_image_range = self.create_dbspinbox(
            value=self.config['acq_config']['projection_scan']['scan_range_um'],
            max=250,
            connect_to_fn=self.update_config
        )        
        self.layout_proj_image_range = QHBoxLayout()
        self.layout_proj_image_range.addWidget(QLabel('Projection scan range (\u00B5m):'))
        self.layout_proj_image_range.addWidget(self.spbx_proj_image_range)
        
        #--------------------------------------------------------------------#
        # Populate scan settings group
        #--------------------------------------------------------------------#
        self.layout_scan_settings = QVBoxLayout()
        self.layout_scan_settings.addLayout(self.layout_scan_step_size)
        self.layout_scan_settings.addLayout(self.layout_mirror_image_range)
        self.layout_scan_settings.addLayout(self.layout_stage_image_range)
        self.layout_scan_settings.addLayout(self.layout_proj_image_range)
        self.layout_scan_settings.addLayout(self.layout_stage_slope)
        self.group_scan_settings = QGroupBox('Scan Settings')
        self.group_scan_settings.setLayout(self.layout_scan_settings)
        
        #--------------------------------------------------------------------#
        # Create camera option entries
        #--------------------------------------------------------------------#
        
        self.spbx_roi_crop_x = self.create_dbspinbox(
            value=self.config['acq_config']['camera_roi']['crop_x'],
            max=2000,
            connect_to_fn=self.update_config
        )
        self.spbx_roi_crop_y = self.create_dbspinbox(
            value=self.config['acq_config']['camera_roi']['crop_y'],
            max=2000,
            connect_to_fn=self.update_config
        )        
        self.spbx_roi_center_x = self.create_dbspinbox(
            value=self.config['acq_config']['camera_roi']['center_x'],
            max=2000,
            connect_to_fn=self.update_config
        )
        self.spbx_roi_center_y = self.create_dbspinbox(
            value=self.config['acq_config']['camera_roi']['center_y'],
            max=2000,
            connect_to_fn=self.update_config
        )
        
        self.layout_roi_x = QHBoxLayout()
        self.layout_roi_x.addWidget(QLabel('ROI center x:'))
        self.layout_roi_x.addWidget(self.spbx_roi_center_x)
        self.layout_roi_x.addStretch()
        self.layout_roi_x.addWidget(QLabel('ROI crop x:'))
        self.layout_roi_x.addWidget(self.spbx_roi_crop_x)
        self.layout_roi_y = QHBoxLayout()
        self.layout_roi_y.addWidget(QLabel('ROI center y:'))
        self.layout_roi_y.addWidget(self.spbx_roi_center_y)
        self.layout_roi_y.addStretch()
        self.layout_roi_y.addWidget(QLabel('ROI crop y:'))
        self.layout_roi_y.addWidget(self.spbx_roi_crop_y)
        self.layout_camera_roi = QVBoxLayout()
        self.layout_camera_roi.addLayout(self.layout_roi_x)
        self.layout_camera_roi.addLayout(self.layout_roi_y)
        self.group_camera_roi = QGroupBox('Camera ROI')
        self.group_camera_roi.setLayout(self.layout_camera_roi)
        
        #--------------------------------------------------------------------#
        # Update widgets dictionary with widgets
        #--------------------------------------------------------------------#
        
        self.widgets.update({
            'O2O3-autofocus': {
              'o2o3_mode': self.cmbx_o2o3_mode  
            },
            'mirror_scan': {
                'scan_range_um': self.spbx_mirror_image_range,
                'scan_step_size_um': self.spbx_scan_step_size
            },
            'projection_scan':{
                'scan_range_um': self.spbx_proj_image_range
            },
            'stage_scan': {
                'scan_step_size_um': self.spbx_scan_step_size,
                'stage_scan_range_um': self.spbx_stage_image_range,
                'coverslip_slope_x': self.spbx_stage_slope_x,
                'coverslip_slope_y': self.spbx_stage_slope_y
            },
            'camera_roi': {
                'center_x': self.spbx_roi_center_x,
                'center_y': self.spbx_roi_center_y,
                'crop_x': self.spbx_roi_crop_x,
                'crop_y': self.spbx_roi_crop_y,
            }
        })
        
        #--------------------------------------------------------------------#
        #--------------------------------------------------------------------#
        # update main layout
        #--------------------------------------------------------------------#
        #--------------------------------------------------------------------#
        
        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.group_ao_main)
        self.main_layout.addWidget(self.group_opm_settings)
        self.main_layout.addWidget(self.group_scan_settings)
        self.main_layout.addWidget(self.group_channels)
        self.main_layout.addWidget(self.group_camera_roi)
        self.setLayout(self.main_layout)
        self.layout()

    #--------------------------------------------------------------------#
    # Methods to update sliders and spinboxes channel states
    #--------------------------------------------------------------------#
    def update_405_spbx(self):
        # Update the spinbox value when the slider value changes
        self.spbx_405_power.setValue(self.sldr_405_power.value())

    def update_405_slider(self):
        # Update the slider value when the spinbox value changes
        self.sldr_405_power.setValue(int(self.spbx_405_power.value()))
    
    def update_488_spbx(self):
        # Update the spinbox value when the slider value changes
        self.spbx_488_power.setValue(self.sldr_488_power.value())

    def update_488_slider(self):
        # Update the slider value when the spinbox value changes
        self.sldr_488_power.setValue(int(self.spbx_488_power.value()))
        
    def update_561_spbx(self):
        # Update the spinbox value when the slider value changes
        self.spbx_561_power.setValue(self.sldr_561_power.value())

    def update_561_slider(self):
        # Update the slider value when the spinbox value changes
        self.sldr_561_power.setValue(int(self.spbx_561_power.value()))
    
    def update_638_spbx(self):
        # Update the spinbox value when the slider value changes
        self.spbx_638_power.setValue(self.sldr_638_power.value())

    def update_638_slider(self):
        # Update the slider value when the spinbox value changes
        self.sldr_638_power.setValue(int(self.spbx_638_power.value()))
        
    def update_705_spbx(self):
        # Update the spinbox value when the slider value changes
        self.spbx_705_power.setValue(self.sldr_705_power.value())

    def update_705_slider(self):
        # Update the slider value when the spinbox value changes
        self.sldr_705_power.setValue(int(self.spbx_705_power.value()))
 
    #--------------------------------------------------------------------#
    # Methods to update acquisition channel states
    #--------------------------------------------------------------------#

    def update_405_state(self):
        checked = self.chx_405_state.isChecked()
        power = self.spbx_405_power.value()
        exposure_ms = self.spbx_405_exp.value()
        for _mode in self.config['OPM']['imaging_modes']:
            self.config['acq_config'][_mode+'_scan']['channel_states'][0] = checked
            self.config['acq_config'][_mode+'_scan']['channel_powers'][0] = power
            self.config['acq_config'][_mode+'_scan']['channel_exposures_ms'][0] = exposure_ms
            
        self.update_config(update_config=False)
               
    def update_488_state(self):
        checked = self.chx_488_state.isChecked()
        power = self.spbx_488_power.value()
        exposure_ms = self.spbx_488_exp.value()
        for _mode in self.config['OPM']['imaging_modes']:
            self.config['acq_config'][_mode+'_scan']['channel_states'][1] = checked
            self.config['acq_config'][_mode+'_scan']['channel_powers'][1] = power
            self.config['acq_config'][_mode+'_scan']['channel_exposures_ms'][1] = exposure_ms
            
        self.update_config(update_config=False)
    
    def update_561_state(self):
        checked = self.chx_561_state.isChecked()
        power = self.spbx_561_power.value()
        exposure_ms = self.spbx_561_exp.value()
        for _mode in self.config['OPM']['imaging_modes']:
            self.config['acq_config'][_mode+'_scan']['channel_states'][2] = checked
            self.config['acq_config'][_mode+'_scan']['channel_powers'][2] = power
            self.config['acq_config'][_mode+'_scan']['channel_exposures_ms'][2] = exposure_ms
            
        self.update_config(update_config=False)
        
    def update_638_state(self):
        checked = self.chx_638_state.isChecked()
        power = self.spbx_638_power.value()
        exposure_ms = self.spbx_638_exp.value()
        for _mode in self.config['OPM']['imaging_modes']:
            self.config['acq_config'][_mode+'_scan']['channel_states'][3] = checked
            self.config['acq_config'][_mode+'_scan']['channel_powers'][3] = power
            self.config['acq_config'][_mode+'_scan']['channel_exposures_ms'][3] = exposure_ms

        self.update_config(update_config=False)
    
    def update_705_state(self):
        checked = self.chx_705_state.isChecked()
        power = self.spbx_705_power.value()
        exposure_ms = self.spbx_705_exp.value()
        for _mode in self.config['OPM']['imaging_modes']:
            self.config['acq_config'][_mode+'_scan']['channel_states'][4] = checked
            self.config['acq_config'][_mode+'_scan']['channel_powers'][4] = power
            self.config['acq_config'][_mode+'_scan']['channel_exposures_ms'][4] = exposure_ms
       
        self.update_config(update_config=False)
        
    #--------------------------------------------------------------------#
    # Methods to update configuration file and emit a signal when settings are updated.
    #--------------------------------------------------------------------#
    
    def update_config(self, update_config: bool = True):
        '''
        Update configuration file and local dict.
        '''
        if update_config:
            with open(self.config_path, 'r') as config_file:
                config = json.load(config_file)
        else:
            config = self.config
            
        for key_id in self.widgets.keys():
            for key in self.widgets[key_id]:
                widget = self.widgets[key_id][key]
                    
                if isinstance(widget, QSpinBox): 
                    config['acq_config'][key_id][key] = widget.value()
                elif isinstance(widget, QDoubleSpinBox): 
                    config['acq_config'][key_id][key] = widget.value()
                elif isinstance(widget, QComboBox): 
                        config['acq_config'][key_id][key] = widget.currentText()
        
        config['acq_config']['opm_mode'] = self.cmbx_opm_mode.currentText()
        config['acq_config']['fluidics'] = self.cmbx_fluidics_mode.currentText()
        config['acq_config']['AO']['daq_mode'] = self.cmbx_ao_daq_mode.currentText()
        config['acq_config']['AO']['metric'] = self.cmbx_ao_metric.currentText()
        
        if self.cmbx_fluidics_mode.currentText()=='on':
            laser_blanking = True
        else:
            laser_blanking = False
        for _mode in config['OPM']['imaging_modes']:
            config['acq_config'][_mode+'_scan']['laser_blanking'] = laser_blanking

        self.config = config
        
        with open(self.config_path, 'w') as file:
                json.dump(self.config, file, indent=4)
        
        self.settings_changed.emit()
        

if __name__ ==  '__main__':
    app = QApplication(sys.argv)
    config_path = Path(Path(Path(sys.path[0]).parent).parent) / Path('opm_config.json')
    window = OPMSettings(config_path)
    window.show()
    
    def signal_recieved():
        print('signal triggered')
    window.settings_changed.connect(signal_recieved)
    
    sys.exit(app.exec())
