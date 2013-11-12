""" Script to tune the the hole front end
"""
from datetime import datetime
import configuration
import logging

from tune_gdac import GdacTune
from tune_feedback import FeedbackTune
from tune_tdac import TdacTune
from tune_fdac import FdacTune
from analysis.plotting.plotting import plotThreeWay

logging.basicConfig(level=logging.INFO, format = "%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")

target_threshold = 50 #in PlsrDAC
target_charge = 270 #in PlsrDAC
target_tot = 5
global_iterations = 1 #set 1..5, 0 is global threshold tuning only
local_iterations = 1 #set 1..5, 0 is local threshold tuning only
cfg_name = "SCC_unknown_2"
      
if __name__ == "__main__":    
    startTime = datetime.now()
    gdac_tune_scan = GdacTune(config_file = configuration.config_file, bit_file = configuration.bit_file, scan_data_path = configuration.scan_data_path)
    gdac_tune_scan.set_target_threshold(target_threshold)
    
    feedback_tune_scan = FeedbackTune(config_file = configuration.config_file, bit_file = None, scan_data_path = configuration.scan_data_path, device = gdac_tune_scan.device)
    feedback_tune_scan.register = gdac_tune_scan.register
    feedback_tune_scan.set_target_charge(target_charge)
    feedback_tune_scan.set_target_tot(target_tot)
    
    tdac_tune_scan = TdacTune(config_file = configuration.config_file, bit_file = None, scan_data_path = configuration.scan_data_path, device = gdac_tune_scan.device)
    tdac_tune_scan.register = gdac_tune_scan.register
    tdac_tune_scan.set_target_threshold(target_threshold)
    tdac_tune_scan.set_start_tdac() # set TDAC = 0
    tdac_tune_scan.set_tdac_bit(bit_position = 4, bit_value = 1) # set start value TDAC = 15
    
    fdac_tune_scan = FdacTune(config_file = configuration.config_file, bit_file = None, scan_data_path = configuration.scan_data_path, device = gdac_tune_scan.device)
    fdac_tune_scan.register = gdac_tune_scan.register
    fdac_tune_scan.set_target_charge(target_charge)
    fdac_tune_scan.set_target_tot(target_tot)
    fdac_tune_scan.set_start_fdac() # set FDAC = 0
    fdac_tune_scan.set_fdac_bit(bit_position = 3, bit_value = 1) # set start value FDAC = 7
    
    difference_bit = int(8/(global_iterations if global_iterations > 0 else 1))
#     print difference_bit
    
    PrmpVbpf = 0 
    Vthin_AC = 0
    Vthin_AF = 0
    
    for iteration in range(0,global_iterations):    #tune iteratively with decreasing range to save time
        start_bit = 7-difference_bit*iteration
        logging.info("Global tuning iteration %d" % iteration)
        gdac_tune_scan.set_gdac_tune_bits(range(start_bit,-1,-1))
        feedback_tune_scan.set_feedback_tune_bits(range(start_bit,-1,-1))
        gdac_tune_scan.start(configure = True if iteration == 0 else False)
        gdac_tune_scan.stop()
        Vthin_AC = gdac_tune_scan.register.get_global_register_value("Vthin_AltCoarse")
        Vthin_AF = gdac_tune_scan.register.get_global_register_value("Vthin_AltFine")
        feedback_tune_scan.start(configure = True) #False)
        feedback_tune_scan.stop()
        PrmpVbpf = feedback_tune_scan.register.get_global_register_value("PrmpVbpf")
#       
    gdac_tune_scan.start(configure = True)# if global_iterations == 0 else False) # always stop with threshold tuning, it is more important
    gdac_tune_scan.stop()
    Vthin_AC = gdac_tune_scan.register.get_global_register_value("Vthin_AltCoarse")
    Vthin_AF = gdac_tune_scan.register.get_global_register_value("Vthin_AltFine")
    logging.info("Results of global tuning PrmpVbpf/Vthin_AltCoarse,Vthin_AltFine = %d/%d,%d" % (PrmpVbpf,Vthin_AC,Vthin_AF))
 
    difference_bit = int(5/(local_iterations if local_iterations > 0 else 1))
    fdac_mean_tot = []
     
    for iteration in range(0,local_iterations):    #tune iteratively
        start_bit = 4#-difference_bit*iteration
        logging.info("Lokal tuning iteration %d" % iteration)
        tdac_tune_scan.set_tdac_tune_bits(range(start_bit,-1,-1))
        fdac_tune_scan.set_fdac_tune_bits(range(start_bit-1,-1,-1))
        tdac_tune_scan.start(configure = True) #False)
        tdac_tune_scan.stop()
        fdac_tune_scan.start(configure = True) #False)
        fdac_tune_scan.stop()
     
    tdac_occ = tdac_tune_scan.start(configure = True) #False) # always stop with threshold tuning, it is more important
    tdac_tune_scan.stop()
 
    gdac_tune_scan.register.save_configuration(name = cfg_name)
     
    if(global_iterations > 0):
        plotThreeWay(hist = gdac_tune_scan.register.get_pixel_register_value("TDAC").transpose(), title = "TDAC distribution final", label = 'TDAC', filename = configuration.scan_data_path+"\TDAC_map.pdf")
        plotThreeWay(hist = tdac_occ.transpose(), title = "Occupancy final", label = 'Occupancy', filename = configuration.scan_data_path+"\occupancy_map.pdf")
     
    if(local_iterations > 0):
        plotThreeWay(hist = fdac_tune_scan.register.get_pixel_register_value("FDAC").transpose(), title = "FDAC distribution final", label = 'FDAC', filename = configuration.scan_data_path+"\FDAC_map.pdf")
        plotThreeWay(hist = fdac_mean_tot.transpose(), title = "TOT mean final", label = 'mean TOT', filename = configuration.scan_data_path+"\mean_tot_map.pdf")
    
    logging.info("Tuning finished in d seconds" + str(datetime.now()-startTime))
