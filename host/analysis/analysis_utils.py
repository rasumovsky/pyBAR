"""This class provides often needed analysis functions, for analysis that is done with python.
"""

import logging
import pandas as pd
import numpy as np
import tables as tb
from scipy.sparse import coo_matrix
from datetime import datetime

from analysis.RawDataConverter.data_histograming import PyDataHistograming

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")


class AnalysisUtils(object):
    """A class to analyze FE-I4 data with Python"""
    def __init__(self, raw_data_file=None, analyzed_data_file=None):
        self.set_standard_settings()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        pass

    def set_standard_settings(self):
        pass

    # @profile
    def correlate_events(self, data_frame_fe_1, data_frame_fe_2):
        '''Correlates events from different Fe by the event number

        Parameters
        ----------
        data_frame_fe_1 : pandas.dataframe
        data_frame_fe_2 : pandas.dataframe

        Returns
        -------
        Merged pandas dataframe.
        '''
        logging.info("Correlating events")
        return data_frame_fe_1.merge(data_frame_fe_2, how='left', on='event_number')  # join in the events that the triggered fe sees, only these are interessting

    def remove_duplicate_hits(self, data_frame):
        '''Removes duplicate hits, possible due to Fe error or Tot = 14 hits.

        Parameters
        ----------
        data_frame : pandas.dataframe

        Returns
        -------
        pandas dataframe.
        '''
        # remove duplicate hits from TOT = 14 hits or FE data error and count how many have been removed
        df_length = len(data_frame.index)
        data_frame = data_frame.drop_duplicates(cols=['event_number', 'col', 'row'])
        logging.info("Removed %d duplicates in trigger FE data" % (df_length - len(data_frame.index)))
        return data_frame

    # @profile
    def get_hits_with_n_cluster_per_event(self, hits_table, cluster_table, n_cluster=1):
        '''Selects the hits with a certain number of cluster.

        Parameters
        ----------
        hits_table : pytables.table
        cluster_table : pytables.table

        Returns
        -------
        pandas.DataFrame
        '''
        logging.info("Calculate hits with %d clusters" % n_cluster)
        data_frame_hits = pd.DataFrame({'event_number': hits_table[:]['event_number'], 'column': hits_table[:]['column'], 'row': hits_table[:]['row']})
        data_frame_hits = data_frame_hits.set_index(keys='event_number')
        n_cluster_in_events = self.get_n_cluster_in_events(cluster_table)
        events_with_n_cluster = n_cluster_in_events[n_cluster_in_events[:, 1] == n_cluster, 0]
        data_frame_hits = data_frame_hits.reset_index()
        return data_frame_hits.loc[events_with_n_cluster]

    # @profile
    def get_n_cluster_in_events(self, cluster_table):
        '''Calculates the number of cluster in every event.

        Parameters
        ----------
        cluster_table : pytables.table

        Returns
        -------
        numpy.Array
            First dimension is the event number
            Second dimension is the number of cluster of the event
        '''
        logging.info("Calculate the number of cluster in every event")
        event_number_array = cluster_table[:]['event_number']
        cluster_in_event = np.bincount(event_number_array)  # for one cluster one event number is given, counts how many different event_numbers are there for each event number from 0 to max event number
        event_number = np.nonzero(cluster_in_event)[0]
        return np.vstack((event_number, cluster_in_event[event_number])).T

    # @profile
    def get_n_cluster_per_event_hist(self, cluster_table):
        '''Calculates the number of cluster in every event.

        Parameters
        ----------
        cluster_table : pytables.table

        Returns
        -------
        numpy.Histogram
        '''
        logging.info("Histogram number of cluster per event")
        cluster_in_events = self.get_n_cluster_in_events(cluster_table)[:, 1]  # get the number of cluster for every event
        return np.histogram(cluster_in_events, bins=range(0, np.max(cluster_in_events) + 2))  # histogram the occurrence of n cluster per event

    # @profile
    def histogram_correlation(self, data_frame_combined):
        '''Takes a dataframe with combined hit data from two Fe and correlates for each event each hit from one Fe to each hit of the other Fe. 

        Parameters
        ----------
        data_frame_combined : pandas.DataFrame

        Returns
        -------
        [numpy.Histogram, numpy.Histogram]
        '''
        logging.info("Histograming correlations")
        corr_row = np.histogram2d(data_frame_combined['row_fe0'], data_frame_combined['row_fe1'], bins=(336, 336), range=[[1, 336], [1, 336]])
        corr_col = np.histogram2d(data_frame_combined['column_fe0'], data_frame_combined['column_fe1'], bins=(80, 80), range=[[1, 80], [1, 80]])
        return corr_col, corr_row

    def histogram_tot(self, array, label='tot'):
        '''Takes the numpy hit/cluster array and histograms the tot values.

        Parameters
        ----------
        hit_array : numpy.ndarray
        label: string

        Returns
        -------
        numpy.Histogram
        '''
        logging.info("Histograming tot values")
        return np.histogram(a=array[label], bins=16, range=(0,16))

    def histogram_tot_per_pixel(self, array, labels=['column', 'row', 'tot']):
        '''Takes the numpy hit/cluster array and histograms the tot values for each pixel

        Parameters
        ----------
        hit_array : numpy.ndarray
        label: string list

        Returns
        -------
        numpy.Histogram
        '''
        logging.info("Histograming tot values for each pixel")
        return np.histogramdd(sample=(array[labels[0]], array[labels[1]], array[labels[2]]), bins=(80, 336, 16), range=[[0, 80], [0, 336], [0, 16]])

    def histogram_mean_tot_per_pixel(self, array, labels=['column', 'row', 'tot']):
        '''Takes the numpy hit/cluster array and histograms the mean tot values for each pixel

        Parameters
        ----------
        hit_array : numpy.ndarray
        label: string list

        Returns
        -------
        numpy.Histogram
        '''
        tot_array = self.histogram_tot_per_pixel(array=array, labels=labels)[0]
        occupancy = self.histogram_occupancy_per_pixel(array=array)[0]  # needed for normalization
        tot_avr = np.average(tot_array, axis=2, weights=range(0, 16)) * sum(range(0, 16))
        tot_avr = np.divide(tot_avr, occupancy)
        return np.ma.array(tot_avr, mask=(occupancy == 0))  # return array with masked pixel without any hit

    def histogram_occupancy_per_pixel(self, array, labels=['column', 'row'], mask_no_hit=False, fast=False):
        if fast:
            occupancy = self.fast_histogram2d(x=array[labels[0]], y=array[labels[1]], bins=(80, 336))
        else:
            occupancy = np.histogram2d(x=array[labels[0]], y=array[labels[1]], bins=(80, 336), range=[[0, 80], [0, 336]])
        if mask_no_hit:
            return np.ma.array(occupancy[0], mask=(occupancy[0] == 0)), occupancy[1], occupancy[2]
        else:
            return occupancy

#     def fast_occupancy_histograming(self, hits): TODO bring this to work
#         hit_histograming = PyDataHistograming()
#         hit_histograming.set_info_output(True)
#         hit_histograming.set_debug_output(True)
#         hit_histograming.create_occupancy_hist(True)
# 
#         hit_histograming.add_hits(hits, hits.shape[0])
#         occupancy_hist = np.zeros(80 * 336 * hit_histograming.get_n_parameters(), dtype=np.uint32)  # create linear array as it is created in histogram class
#         hit_histograming.get_occupancy(occupancy_hist)
#         occupancy_hist = np.reshape(a=self.occupancy.view(), newshape=(80, 336, hit_histograming.get_n_parameters()), order='F')  # make linear array to 3d array (col,row,parameter)
#         occupancy_hist = np.swapaxes(occupancy_hist, 0, 1)
#         return occupancy_hist

    def fast_histogram2d(self, x, y, bins):
        '''WARNING: GIVES NOT EXACT RESULTS'''
        nx = bins[0]-1
        ny = bins[1]-1

        print nx, ny

        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        dx = (xmax - xmin) / (nx - 1.0)
        dy = (ymax - ymin) / (ny - 1.0)

        weights = np.ones(x.size)

        # Basically, this is just doing what np.digitize does with one less copy
        xyi = np.vstack((x,y)).T
        xyi -= [xmin, ymin]
        xyi /= [dx, dy]
        xyi = np.floor(xyi, xyi).T

        # Now, we'll exploit a sparse coo_matrix to build the 2D histogram...
        grid = coo_matrix((weights, xyi), shape=(nx, ny)).toarray()

        return grid, np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)

    def get_scan_parameter(self, meta_data_array):
        '''Takes the numpy meta data array and returns the different scan parameter settings and the name aligned in a dictionary

        Parameters
        ----------
        meta_data_array : numpy.ndarray

        Returns
        -------
        python.dict{string, numpy.Histogram}
        '''

        if len(meta_data_array.dtype.names) < 5:  # no meta_data found
            return
        scan_parameters = {}
        for scan_par_name in meta_data_array.dtype.names[len(meta_data_array.dtype.names) - 1:len(meta_data_array.dtype.names)]:
            scan_parameters[scan_par_name] = np.unique(meta_data_array[scan_par_name])
        return scan_parameters

if __name__ == "__main__":
    pass