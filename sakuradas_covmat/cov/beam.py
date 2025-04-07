#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Beamforming.

"""
import numpy as np
import sys


class Beam:
    def __init__(self, nwin, traveltimes):
        self.nwin = nwin
        self.nx = traveltimes.nx
        self.ny = traveltimes.ny
        self.nz = traveltimes.nz
        self.traveltimes = traveltimes
        self.likelihood = np.zeros((self.nwin, self.nz, self.ny, self.nx))*np.nan
        self.likelihood_dict = {}
        for _ in range(self.nwin):
            self.likelihood_dict[_] = np.zeros(4)*np.nan

        self.nrf = np.zeros(self.nwin)
        self.correlation_shifted = []
        self.correlation_unshifted = []

    # def set_extent(self, west, east, south, north, depth_top, depth_max, dlon, dlat, dz):
    #     """Limits of the beamforming domain.

    #     Args
    #     ----
    #         west, east, south, north, depth_top and depth_max (float):
    #             Extent of the 3D map in degrees for the azimuths and km for
    #             the depths. The depth is given in km and should be negative for
    #             elevation.
    #     """

    #     self.xmin = west
    #     self.xmax = east
    #     self.ymin = south
    #     self.ymax = north
    #     self.zmin = depth_top
    #     self.zmax = depth_max
    #     print('easting', west, east, self.nx)
    #     print('northing', south, north, self.ny)
    #     self.lon = np.linspace(west, east, self.nx)  #np.arange(west, east+dlon, dlon) #np.linspace(west, east, self.nx)
    #     self.lat = np.linspace(south, north, self.ny) #np.arange(south, north+dlat, dlat)[::-1] #np.linspace(south, north, self.ny)
    #     if dz>0:
    #         self.dep = np.arange(depth_top, depth_max+dz,dz) #np.linspace(depth_top, depth_max, self.nz)
    #     else:
    #         self.dep = np.array([self.zmin])
    #     self.meshgrid = np.meshgrid(self.dep, self.lat, self.lon)
    #     self.meshgrid_size = len(self.likelihood[0][0].ravel())
    #     self.dlon = dlon 
    #     self.dlat = dlat
        
    def set_extent(self, lon, lat, dep):
        """Limits of the beamforming domain.

        Args
        ----
            west, east, south, north, depth_top and depth_max (float):
                Extent of the 3D map in degrees for the azimuths and km for
                the depths. The depth is given in km and should be negative for
                elevation.
        """
        self.lon = lon
        self.lat = lat
        self.dep = dep
        dlon = np.abs( lon[1]-lon[0] )
        dlat = np.abs( lat[1]-lat[0] )
        self.meshgrid = np.meshgrid(dep, lat, lon)
        self.meshgrid_size = len(self.likelihood[0][0].ravel())
        self.dlon = dlon 
        self.dlat = dlat
        
    def max_likelihood(self, window_index):
        
        beam_max_indices = np.unravel_index(np.nanargmax(self.likelihood[window_index]), self.likelihood[window_index].shape)
        
        
        beammax_dep_arr = []
        for k in range(self.likelihood[window_index].shape[0]):
            beammax_dep_arr.append(np.nanmax(self.likelihood[window_index][k,:,:]))
        
        beammax_dep_idx = np.nanargmax(beammax_dep_arr)
        beammax_lat_idx, beammax_lon_idx = np.unravel_index(np.argmax(self.likelihood[window_index][beammax_dep_idx,:,:]), self.likelihood[window_index][beammax_dep_idx,:,:].shape)
        
        beam_max = self.likelihood[window_index][
            beam_max_indices
        ]  # return max likelihood value

        
        beam_max_lon = (
            self.lon[beam_max_indices[2]]
        )
        beam_max_lat = (
            self.lat[beam_max_indices[1]]
        )
        beam_max_depth = (
            self.dep[beam_max_indices[0]]
        )
        
        return (beam_max_depth, beam_max_lat, beam_max_lon, beam_max, np.meshgrid(self.lon, self.lat), np.meshgrid(self.lon, self.dep), np.meshgrid(self.lat, self.dep), beammax_dep_idx, beammax_lat_idx, beammax_lon_idx, self.likelihood[window_index], self.likelihood_dict)

    def calculate_nrf(self, window_index):
        self.nrf[window_index] = (
            np.nanmax(self.likelihood[window_index])
            * np.size(self.likelihood[window_index])
            / np.nansum(self.likelihood[window_index])
        )
        return self.nrf[window_index]

    def calculate_likelihood(
        self, cross_correlation, sampling_rate, window_index, close=None
    ):
        """Shift cross-correlation for each source in grid.
        cross_correlation.shape = (stat_combinations, lags)
        """

        # Initialization
        cross_correlation = cross_correlation.T
        self.correlation_unshifted.append(cross_correlation)

        beam_max = 0
        trii, trij = np.triu_indices(self.traveltimes.nsta, k=1)
        n_lon = self.nx
        n_lat = self.ny
        n_dep = self.nz
        center = (cross_correlation.shape[1] - 1) // 2 + 1
        
        beam_arr_arr = []
        for k in range(n_dep):
            beam_arr = []
            for j in range(n_lat):
                for i in range(n_lon):
                    #print(self.dep[k], self.lat[j], self.lon[i])
                    # Extract the travel times of all stations from specific source at i, j ,k
                    tt = self.traveltimes.grid[:, k, j, i]
                    # Increase the dimension and find the substraction between all the stations in a NxN matrix
                    tt = tt[:, None] - tt
                    # Extract the upper triangle values (and invert sign)
                    tt = -tt[trii, trij]

                    if np.any(np.isnan(tt)):
                        beam_arr.append([self.lat[j], self.lon[i], np.nan])
                        continue
                
                    if close is not None:
                        tt = tt[close]

                    # Shift = center of the CC + arrival time
                    dt_int = -(sampling_rate * tt).astype(int)
                    dt_int = center + dt_int

                    max_time_diff = np.nanmax(np.abs(dt_int))

                    if max_time_diff >= cross_correlation.shape[1]:
                        sys.exit(
                            "ERROR: the max time difference "
                            + str(max_time_diff)
                            + " is bigger than the correlation duration "
                            + str(cross_correlation.shape[1])
                        )

                    # beam is sum of the CCs after being shifted the arrival time difference
                    # extract for each stat comb, the sum of the CCs with the delay
                    beam = cross_correlation[range(cross_correlation.shape[0]), dt_int]
                    beam[beam==0.0] = np.nan
                    beam = np.nanmean(beam)
                    self.likelihood[window_index, k, j, i] = beam
                    
                    beam_arr.append([self.lat[j], self.lon[i], beam])
                    if beam_max < beam:
                        # Keep that into memory
                        beam_max = beam
                        dt_int_abs = -(dt_int - center)

                # Move
                rows, column_indices = np.ogrid[
                    : cross_correlation.shape[0], : cross_correlation.shape[1]
                ]

                # Find where the time delay is bigger than the possible correlation time; this condition will never be fulfilled
                dt_int_abs[np.abs(dt_int_abs) > cross_correlation.shape[1]] = (
                    cross_correlation.shape[1] - 1
                )
                # Move the negatives (almost all of them)
                dt_int_abs[dt_int_abs < 0] += cross_correlation.shape[1]
                column_indices = column_indices - dt_int_abs[:, np.newaxis]
                self.correlation_shifted.append(cross_correlation[rows, column_indices])
                # return self.likelihood[window_index]
                # return cross_correlation_best.T
                
                
            beam_arr_arr.append(np.array(beam_arr))
        
        self.likelihood_dict[window_index] = np.array(beam_arr_arr)   
        