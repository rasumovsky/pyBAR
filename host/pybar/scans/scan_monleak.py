"""A script that changes a scan parameter (usually PlsrDAC, in inner loop) in a certain range for selected pixels and measures the length of the hit OR signal with the FPGA TDC. This calibration can be used to measure charge information for single pixels with higher precision than with the quantized TOT information.
"""

import logging
import numpy as np
import tables as tb
import progressbar

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from pybar.analysis.analyze_raw_data import AnalyzeRawData
from pybar.fei4.register_utils import invert_pixel_mask, make_pixel_mask_from_col_row, make_box_pixel_mask_from_col_row
from pybar.fei4_run_base import Fei4RunBase
from pybar.run_manager import RunManager
from pybar.analysis.analysis_utils import get_scan_parameter, get_hits_of_scan_parameter, get_unique_scan_parameter_combinations


class MonLeakScan(Fei4RunBase):
    ''' Mon leak scan
    '''
    _default_run_conf = {
        "repeat_command": 1,
        "scan_parameters": [('column', None),
                            ('row', None),
                            ('GADCout', None),
                           ('GADCStatus', None)],
        "plot_tdc_histograms": False,
        # list of (col, row) tupels. From 1 to 80/336.
        "pixels": (np.dstack(np.where(make_box_pixel_mask_from_col_row([1, 80], [1, 336]) == 1)) + 1)[0],
        "enable_masks": ["Enable"],
        "disable_masks": ["Imon"]
        }
    
    def configure(self):
        commands = []
        
        # First enter configuration mode to issue slow commands:
        commands.extend(self.register.get_commands("ConfMode"))
        
        # Settings for the mon-leak scan:
        self.register.set_global_register_value("GADCStart",1)
        self.register.set_global_register_value("GADCSel",7)# for readback test read,write
        self.register.set_global_register_value("GADCVref",170)
        self.register.set_global_register_value("GADCCompBias",100)
        self.register.set_global_register_value("EventLimit",10)
        self.register.set_global_register_value("TempSensDisable",1)
        self.register.set_global_register_value("EN_PLL",1)
        self.register.set_global_register_value("GateHitOr",1)
        self.register.set_global_register_value("CalEn",0)
        self.register.set_global_register_value("Efuse_Sense",0)
        self.register.set_global_register_value("Latch_En",0)
        self.register.set_global_register_value("ReadErrorReq",0)
        self.register.set_global_register_value("SR_Clr",0)
        self.register.set_global_register_value("SR_Clock",0)
        self.register.set_global_register_value("StopClkPulse",0)
        self.register.set_global_register_value("Trig_Count",0)
        commands.extend(self.register.get_commands("WrRegister", name=["GADCStart","GADCSel","GADCVref","GADCCompBias","EventLimit","TempSensDisable","EN_PLL","GateHitOr","CalEn","Efuse_Sense","Latch_En","ReadErrorReq","SR_Clr","SR_Clock","StopClkPulse","Trig_Count"]))
        self.register_utils.send_commands(commands)

    def scan(self):
        def write_double_column(column):
            return (column - 1) / 2

        def inject_double_column(column):
            if column == 80:
                return 39
            else:
                return (column) / 2
                                
        # Store list of columns and rows to unmask:
        masked_row = []
        masked_col = []
        
        # Loop over individual pixels:
        for pixel_index, pixel in enumerate(self.pixels):
                        
            column = pixel[0]# define column
            row = pixel[1]#define row
            #logging.info('Scanning pixel: %d / %d (column / row)' % (column, row))
            
            # Add the current pixel coordinates to unmask list:
            masked_row.append(row)
            masked_col.append(column)
            
            
            # To speed up the scan by factor of 100x
            # Only actually scan the value every 10th pixel:
            if pixel_index % 100 == 0:
                
                logging.info('Scanning pixel: %d / %d (column / row)' % (column, row))
                
                # Re-enter Configuration mode:
                commands = []
                commands.extend(self.register.get_commands("ConfMode"))
                
                # Don't bother with enable mask for the moment:
                # Instead create disable (imon) masks:
                make_pixel_mask_from_col_row(column, row, default=0, value=1)
                multi_pixel_disable_mask = make_pixel_mask_from_col_row(masked_col, masked_row, default=1, value=0)
                map(lambda mask_name: self.register.set_pixel_register_value(mask_name, multi_pixel_disable_mask), self.disable_masks)
                commands.extend(self.register.get_commands("WrFrontEnd", name=self.disable_masks))
                
                # Set global register column pair mode:
                self.register.set_global_register_value("Colpr_Addr", inject_double_column(column))
                # Set column pair mode bit 0:
                commands.append(self.register.get_commands("WrRegister", name=["Colpr_Addr"])[0])
                self.register_utils.send_commands(commands)
                    
                # Enable the DUT to read/write?
                self.dut['tdc_rx2']['ENABLE'] = True
                            
                commands = []
                commands.extend(self.register.get_commands("RunMode"))
                self.register_utils.send_commands(commands)
                
                self.dut['tdc_rx2']['EN_ARMING'] = True
                
                #prevents the buffer from filling up:
                #self.dut['sram']['RESET']

                # read the value of the GADCout and GADCStatus 
                GADCout = self.register.get_global_register_value("GADCout")
                GADCStatus = self.register.get_global_register_value("GADCStatus")
                logging.info('GADCout = %d, GACDStatus = %d' % (GADCout, GADCStatus))
                
                # Definition of command to send global pulse and wait 320 clock cycles:
                #GADCSelRB 3 bit, ?
                scan_globalpulse_command = self.register.get_commands("GlobalPulse",Width=5)[0] + self.register.get_commands("zeros", length=320)[0] + self.register.get_commands("LV1")[0] + self.register.get_commands("LV1")[0] + self.register.get_commands("zeros", length=200)[0]

                commands = []
                commands.extend(self.register.get_commands("RunMode"))
                self.register_utils.send_commands(commands)
                
                with self.readout(column=column, row=row, GADCout=GADCout, GADCStatus=GADCStatus):
                    self.register_utils.send_command(command=scan_globalpulse_command, repeat=self.repeat_command)
                
                # read the value of the GADCout and GADCStatus 
                GADCout = self.register.get_global_register_value("GADCout")
                GADCStatus = self.register.get_global_register_value("GADCStatus")
                GADCSel = self.register.get_global_register_value("GADCSel")
                GADCSelRB = self.register.get_global_register_value("GADCSelRB")
                val = self.register.get_global_register_value("EOCHLSkipped")
                logging.info('EOCHLSkipped = %d, GADCSel = %d, GADCSelRB = %d, GADCout = %d, GACDStatus = %d' % (val, GADCSel, GADCSelRB, GADCout, GADCStatus))
    
                self.dut['tdc_rx2']['EN_ARMING'] = False
                self.dut['tdc_rx2']['ENABLE'] = False

    # WORRY ABOUT ANALYSIS LATER:
    def analyze(self):
        logging.info('Analyze and plot results')
        '''
        def plot_calibration(col_row_combinations, scan_parameter, calibration_data, repeat_command, filename):  # Result calibration plot function
            for index, (column, row) in enumerate(col_row_combinations):
                logging.info("Plot calibration for pixel " + str(column) + '/' + str(row))
                fig = Figure()
                canvas = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                fig.patch.set_facecolor('white')
                ax.grid(True)
                ax.errorbar(scan_parameter, calibration_data[column - 1, row - 1, :, 0] * 25. + 25., yerr=[calibration_data[column - 1, row - 1, :, 2] * 25, calibration_data[column - 1, row - 1, :, 2] * 25], fmt='o', label='FE-I4 ToT [ns]')
                ax.errorbar(scan_parameter, calibration_data[column - 1, row - 1, :, 1] * 1.5625, yerr=[calibration_data[column - 1, row - 1, :, 3] * 1.5625, calibration_data[column - 1, row - 1, :, 3] * 1.5625], fmt='o', label='TDC ToT [ns]')
                ax.set_title('Calibration for pixel ' + str(column) + '/' + str(row) + '; ' + str(repeat_command) + ' injections per setting')
                ax.set_xlabel('Charge [PlsrDAC]')
                ax.set_ylabel('TOT')
                ax.legend(loc=0)
                filename.savefig(fig)
                if index > 100:  # stop for too many plots
                    break

        with AnalyzeRawData(raw_data_file=self.output_filename, create_pdf=True) as analyze_raw_data:  # Interpret the raw data file
            analyze_raw_data.create_occupancy_hist = False  # too many scan parameters to do in ram histograming
            analyze_raw_data.create_hit_table = True
            analyze_raw_data.create_tdc_hist = True
            analyze_raw_data.interpreter.use_tdc_word(True)  # align events at TDC words, first word of event has to be a tdc word
            analyze_raw_data.interpret_word_table()
            analyze_raw_data.interpreter.print_summary()
            analyze_raw_data.plot_histograms()

        with tb.open_file(self.output_filename + '_interpreted.h5', 'r') as in_file_h5:  # Get scan parameters from interpreted file
            scan_parameters_dict = get_scan_parameter(in_file_h5.root.meta_data[:])
            inner_loop_parameter_values = scan_parameters_dict[next(reversed(scan_parameters_dict))]  # inner loop parameter name is unknown
            scan_parameter_names = scan_parameters_dict.keys()
            n_par_combinations = len(get_unique_scan_parameter_combinations(in_file_h5.root.meta_data[:]))
            col_row_combinations = get_unique_scan_parameter_combinations(in_file_h5.root.meta_data[:], scan_parameters=('column', 'row'), scan_parameter_columns_only=True)

        with tb.openFile(self.output_filename + "_calibration.h5", mode="w") as calibration_data_file:
            logging.info('Create calibration')
            output_pdf = PdfPages(self.output_filename + "_calibration.pdf")
            calibration_data = np.zeros(shape=(80, 336, len(inner_loop_parameter_values), 4), dtype='f4')  # result of the calibration is a histogram with col_index, row_index, plsrDAC value, mean discrete tot, rms discrete tot, mean tot from TDC, rms tot from TDC

            progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=n_par_combinations, term_width=80)
            old_scan_parameters = None
            tot_data = None
            tdc_data = None

            for index, (actual_scan_parameters, hits) in enumerate(get_hits_of_scan_parameter(self.output_filename + '_interpreted.h5', scan_parameter_names, chunk_size=1.5e7)):
                if index == 0:
                    progress_bar.start()  # start after the event index is created to get reasonable ETA

                actual_col, actual_row, _ = actual_scan_parameters

                if len(hits[np.logical_and(hits['column'] != actual_col, hits['row'] != actual_row)]):
                    logging.warning('There are %d hits from not selected pixels in the data' % len(hits[np.logical_and(hits['column'] != actual_col, hits['row'] != actual_row)]))

                hits = hits[(hits['event_status'] & 0b0000011110001000) == 0b0000000100000000]  # only take hits from good events (one TDC word only, no error)
                column, row, tot, tdc = hits['column'], hits['row'], hits['tot'], hits['TDC']

                if old_scan_parameters != actual_scan_parameters:  # Store the data of the actual PlsrDAC value
                    if old_scan_parameters:  # Special case for the first PlsrDAC setting
                        inner_loop_scan_parameter_index = np.where(old_scan_parameters[-1] == inner_loop_parameter_values)[0][0]  # translate the scan parameter value to an index for the result histogram
                        calibration_data[column - 1, row - 1, inner_loop_scan_parameter_index, 0] = np.mean(tot_data)
                        calibration_data[column - 1, row - 1, inner_loop_scan_parameter_index, 1] = np.mean(tdc_data)
                        calibration_data[column - 1, row - 1, inner_loop_scan_parameter_index, 2] = np.std(tot_data)
                        calibration_data[column - 1, row - 1, inner_loop_scan_parameter_index, 3] = np.std(tdc_data)
                        progress_bar.update(index)
                    tot_data = np.array(tot)
                    tdc_data = np.array(tdc)
                    old_scan_parameters = actual_scan_parameters
                else:
                    np.concatenate((tot_data, tot))
                    np.concatenate((tdc_data, tdc))

            else:
                inner_loop_scan_parameter_index = np.where(old_scan_parameters[-1] == inner_loop_parameter_values)[0][0]  # translate the scan parameter value to an index for the result histogram
                calibration_data[column - 1, row - 1, inner_loop_scan_parameter_index, 0] = np.mean(tot_data)
                calibration_data[column - 1, row - 1, inner_loop_scan_parameter_index, 1] = np.mean(tdc_data)
                calibration_data[column - 1, row - 1, inner_loop_scan_parameter_index, 2] = np.std(tot_data)
                calibration_data[column - 1, row - 1, inner_loop_scan_parameter_index, 3] = np.std(tdc_data)

            calibration_data_out = calibration_data_file.createCArray(calibration_data_file.root, name='HitOrCalibration', title='Hit OR calibration data', atom=tb.Atom.from_dtype(calibration_data.dtype), shape=calibration_data.shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
            calibration_data_out[:] = calibration_data
            calibration_data_out.attrs.dimensions = scan_parameter_names
            calibration_data_out.attrs.scan_parameter_values = inner_loop_parameter_values
            plot_calibration(col_row_combinations, scan_parameter=inner_loop_parameter_values, calibration_data=calibration_data, repeat_command=self.repeat_command, filename=output_pdf)
            output_pdf.close()
            progress_bar.finish()
            '''

if __name__ == "__main__":
    RunManager('../configuration.yaml').run_run(MonLeakScan)
