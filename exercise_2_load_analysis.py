# -*- coding: utf-8 -*-
"""
Created on 2023-07-14

@author: ivespe

Intro script for Exercise 2 ("Load analysis to evaluate the need for flexibility") 
in specialization course module "Flexibility in power grid operation and planning" 
at NTNU (TET4565/TET4575) 

"""

# %% Dependencies

import pandapower as pp
import pandapower.plotting as pp_plotting
import pandas as pd
import os
import load_scenarios as ls
import load_profiles as lp
import pandapower_read_csv as ppcsv
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


# %% Define input data

# Location of (processed) data set for CINELDI MV reference system
# (to be replaced by your own local data folder)
path_data_set         = 'C:/Users/oscar/OneDrive/Dokumenter/Høst 2023/TET4565 Spesialiseringsemne/Cineldi/Data'

filename_load_data_fullpath = os.path.join(path_data_set,'load_data_CINELDI_MV_reference_system.csv')
filename_load_mapping_fullpath = os.path.join(path_data_set,'mapping_loads_to_CINELDI_MV_reference_grid.csv')

# Subset of load buses to consider in the grid area, considering the area at the end of the main radial in the grid
bus_i_subset = [90, 91, 92, 96]

# Assumed power flow limit in MW that limit the load demand in the grid area (through line 85-86)
P_lim = 0.637 

# Maximum load demand of new load being added to the system
P_max_new = 0.4

# Which time series from the load data set that should represent the new load
i_time_series_new_load = 90


# %% Read pandapower network

net = ppcsv.read_net_from_csv(path_data_set, baseMVA=10)
net_new = net

# %% Set up hourly normalized load time series for a representative day (task 2; this code is provided to the students)

load_profiles = lp.load_profiles(filename_load_data_fullpath)
load_new = load_profiles


# Get all the days of the year
repr_days = list(range(1,366))

# Get relative load profiles for representative days mapped to buses of the CINELDI test network;
# the column index is the bus number (1-indexed) and the row index is the hour of the year (0-indexed)
profiles_mapped = load_profiles.map_rel_load_profiles(filename_load_mapping_fullpath,repr_days)

# Retrieve load time series for new load to be added to the area
new_load_profiles = load_profiles.get_profile_days(repr_days)
new_load_time_series = new_load_profiles[i_time_series_new_load]*P_max_new

# Calculate load time series in units MW (or, equivalently, MWh/h) by scaling the normalized load time series by the
# maximum load value for each of the load points in the grid data set (in units MW); the column index is the bus number
# (1-indexed) and the row index is the hour of the year (0-indexed)
load_time_series_mapped = profiles_mapped.mul(net.load['p_mw'])
# %% TASK 1
#  TASK 1: Function to run a power flow analysis and return the minimum bus voltage
def run_power_flow(net):
    pp.runpp(net)
    return np.min(net.res_bus['vm_pu'])

# List to store results
scaling_factors = np.linspace(1, 2, 11)  # Range of scaling factors from 1 to 2
min_bus_voltage = []
aggregated_load_demand = []

original_load2 = {}
for i in bus_i_subset:
    original_load2[i] = net.load.at[i, 'p_mw']

# Iterate through scaling factors
for scaling_factor in scaling_factors:
    # Scale the load demand values proportionally for each bus in the subset
    for bus_id in bus_i_subset:
        original_load = net.load.at[bus_id, 'p_mw']
        net.load.at[bus_id, 'p_mw'] = original_load2[bus_id] * scaling_factor
    
    # Run power flow analysis and store the minimum bus voltage
    min_voltage = run_power_flow(net)
    min_bus_voltage.append(min_voltage)

    # Calculate the aggregated load demand in the area
    aggregated_load = sum(net.load.at[bus_id, 'p_mw'] for bus_id in bus_i_subset)
    aggregated_load_demand.append(aggregated_load)

    # Reset the load values to their original values for the next iteration
    for bus_id in bus_i_subset:
        net.load.at[bus_id, 'p_mw'] = original_load2[bus_id]

voltage_limit = 0.95

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(aggregated_load_demand, min_bus_voltage, marker='o', linestyle='-', color='b', label='Minimum Bus Voltage')
plt.axhline(y=voltage_limit, color='r', linestyle='--', label='Voltage Limit (0.95)')
plt.xlabel('Aggregated Load Demand (MW)')
plt.ylabel('Minimum Bus Voltage (pu)')
plt.title('Minimum Bus Voltage vs Aggregated Load Demand')
plt.legend()
plt.grid(True)
# Show the plot
plt.show()

# Compare minimum bus voltage values with the voltage limit of 0.95
for i, scaling_factor in enumerate(scaling_factors):
    if min_bus_voltage[i] < voltage_limit:
        print(f"Scaling Factor: {scaling_factor:.2f}, Minimum Bus Voltage: {min_bus_voltage[i]:.3f} (Below {voltage_limit:.2f})")
    else:
        print(f"Scaling Factor: {scaling_factor:.2f}, Minimum Bus Voltage: {min_bus_voltage[i]:.3f} (Above {voltage_limit:.2f})") 
          
# %% TASK 2 - Accessing load time series for a full year

# TASK 2: Load time series for all load points for a full year
max_load = sum(net.load['p_mw'][i] for i in bus_i_subset)

print(net.load['p_mw'][bus_i_subset])
print("The maximum load of the individual buses in the subset summed is:", max_load)

# %% TASK 3 - Calculate and plot aggregated load demand for 90,91,92,96

# Task 3: Calculate the aggregated load demand time series by summing the load values for the selected load points
aggregated_load_demand = load_time_series_mapped[bus_i_subset].sum(axis=1)
time_index = range(len(aggregated_load_demand))

plt.figure(figsize=(12, 6))
plt.plot(time_index, aggregated_load_demand, label='Aggregated Load Demand', color='blue')
plt.xlabel('Hour of the Year')
plt.ylabel('Load Demand (MW)')
plt.title('Aggregated Load Demand Time Series')
plt.legend()
plt.grid(True)
plt.show()

# %% TASK 4 - Maximum load demand
# Task 4: Calculate the aggregated load demand by summing the load for the specified load points
aggregated_load_demand = load_time_series_mapped[bus_i_subset].sum(axis=1)

# Find the maximum value in the aggregated load time series
max_load_demand = aggregated_load_demand.max()

# Print the maximum value and its corresponding time index
max_load_demand_time_index = aggregated_load_demand.idxmax()

print(f"Maximum Aggregated Load Demand: {max_load_demand} MW")
# %% TASK 5 - Load duration curve

# Task 5: Sort the aggregated load demand values in descending order
sorted_load_demand = np.sort(aggregated_load_demand)[::-1]

# Create a list of hours from 1 to the number of hours in a year
hours = range(1, len(sorted_load_demand) + 1)

# Plot the Load Duration Curve
plt.figure(figsize=(10, 6))
plt.plot(hours, sorted_load_demand, color='blue', label='Load Duration Curve')
plt.xlabel('Hours (Sorted by Load)')
plt.ylabel('Load Demand (MW)')
plt.title('Load Duration Curve')

# Calculate the indices where the load demand is below P_lim
below_limit_indices = np.where(sorted_load_demand < P_lim)[0]

# Create a mask for the area to be shaded in green
green_fill_mask = np.zeros(len(hours))
green_fill_mask[:below_limit_indices[-1] + 1] = 1

# Add a black horizontal bar indicating the power flow limit (P_lim)
plt.axhline(y=P_lim, color='black', linewidth=2, label=f'P_lim = {P_lim} MW')

# Create a mask for the area above P_lim to be unshaded
above_limit_mask = (sorted_load_demand >= P_lim)

# Apply the mask to prevent shading above P_lim
green_fill_mask[above_limit_mask] = 0

# Shade the area below the Load Duration Curve and the black bar in light green
plt.fill_between(hours, 0, sorted_load_demand, where=green_fill_mask.astype(bool), color='lightgreen', alpha=0.7)
plt.grid(True)
plt.legend()
plt.show()

# %% TASK 6

# Task 6: Check if the maximum load demand is greater than the power flow limit (P_lim)
if max_load_demand < P_lim:
    # Calculate the capacity margin
    capacity_margin = P_lim - max_load_demand 
    print(f"Capacity Margin: {capacity_margin:.3f} MW")
else:
    print("The current load demand exceed the power flow limit. Additional capacity margin is needed.")

# %% TASK 7

# Task 7: Assume the new load has been connected to bus 95 with a maximum load demand of 0.4 MW
# Create a new Pandas DataFrame for load time series before and after adding the new load
load_data = pd.DataFrame({
    'Before New Load (MW)': load_time_series_mapped[bus_i_subset].sum(axis=1),
    'After New Load (MW)': load_time_series_mapped[bus_i_subset].sum(axis=1) + new_load_time_series
})
# Create a load duration curve plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(load_data) + 1), load_data['Before New Load (MW)'], label='Before New Load')
plt.plot(range(1, len(load_data) + 1), load_data['After New Load (MW)'], label='After New Load')
plt.xlabel('Hours')
plt.ylabel('Cumulative Load (MW)')
plt.title('Aggregated load demand series')
plt.legend()
plt.grid()
plt.show()


#FINDING THE LOAD DURATION CURVE OF THE AGGREGATED LOAD DEMAND SERIES
new_sorted_load_demand = np.sort(load_data['After New Load (MW)'])[::-1]
hours = range(1, len(new_sorted_load_demand) + 1)

# Plot the Load Duration Curve
plt.figure(figsize=(10, 6))
plt.plot(hours, new_sorted_load_demand, color='orange', label='Load Duration Curve')
plt.xlabel('Hours (Sorted by Load)')
plt.ylabel('Load Demand (MW)')
plt.title('Load Duration Curve')

# Calculate the indices where the load demand is below P_lim
below_limit_indices = np.where(new_sorted_load_demand < P_lim)[0]

# Create a mask for the area to be shaded in green
green_fill_mask = np.zeros(len(hours))
green_fill_mask[:below_limit_indices[-1] + 1] = 1

# Add a black horizontal bar indicating the power flow limit (P_lim)
plt.axhline(y=P_lim, color='black', linewidth=2, label=f'P_lim = {P_lim} MW')

# Create a mask for the area above P_lim to be unshaded
above_limit_mask = (new_sorted_load_demand >= P_lim)

# Apply the mask to prevent shading above P_lim
green_fill_mask[above_limit_mask] = 0

# Shade the area below the Load Duration Curve and the black bar in light green
plt.fill_between(hours, 0, new_sorted_load_demand, where=green_fill_mask.astype(bool), color='lightgreen', alpha=0.7)
plt.grid(True)
plt.legend()
plt.show()

# %% TASK 8

# Task 8: Finding the negative capacity margin
#Finding new max load demand:
new_max_load_demand = new_sorted_load_demand.max()

print(f"Maximum aggregated load demand: {new_max_load_demand} MW")
#print(f"Time Index of Maximum Load Demand: {max_load_demand.index}")


if max_load_demand < P_lim:
    # Calculate the capacity margin
    capacity_margin = P_lim - new_max_load_demand 
    print(f"Capacity Margin: {capacity_margin:.3f} MW")
else:
    capacity_margin = P_lim - new_max_load_demand 
    print(f"Capacity Margin: {capacity_margin:.3f} MW")
    print("The current load demand exceed the power flow limit. Additional capacity margin is needed.")

# %% TASK 9

# Taks 9: Constant new load demand as opposed to time dependent
constant_new_load_demand = 0.4  # MW

# Estimate the maximum overloading by adding the constant load demand to the existing maximum load demand
estimated_max_overloading = max_load_demand + constant_new_load_demand

# Print the estimate
print(f"Estimated maximum overloading (without time dependence): {estimated_max_overloading:.3f} MW")

# %% TASK 10

#Task 10: Coincidence factor
existing_load_profiles = load_time_series_mapped[bus_i_subset] + new_load_time_series[bus_i_subset]
max_values = existing_load_profiles.max()

# Calculate the coincidence factor
coincidence = new_max_load_demand / max_values.sum()#(max_load + 0.4) 
print("The coincidence factor with the existing load profile is:", coincidence)

# %% TASK 11
# Task 11: Number of hours to be cut per year
new_sorted_load_demand = np.sort(load_data['After New Load (MW)'].values)[::-1] 

# Initialize a counter for hours exceeding the power flow limit
hours_congested = 0

# Iterate through the load duration curve
for load_value in new_sorted_load_demand:
    if load_value > P_lim:
        hours_congested += 1

# Print the number of hours per year that would require load reduction to avoid congestion
print(f"Number of hours per year requiring load reduction to avoid congestion (using new load data): {hours_congested} hours")

# %% TASK 14
# Task 14: Constant load demand of the new load
constant_new_load_demand = 0.4  # MW

#Updating the new load demand to consume 0.4 constant instead of the new_load_time_series
new_sorted_load_demand = np.sort(load_data['Before New Load (MW)'].values)[::-1] + constant_new_load_demand

# Create a load duration curve for the constant load demand
constant_load_duration_curve = np.full(len(sorted_load_demand), P_lim)
#new_sorted_load_demand 
# Plot both load duration curves
plt.figure(figsize=(10, 6))
plt.plot(hours, new_sorted_load_demand, color='orange', label='Task 7 Load Duration Curve')
plt.plot(hours, constant_load_duration_curve, color='red', linestyle='--', label='P_lim (0.637 MW)')
plt.xlabel('Hours (Sorted by Load)')
plt.ylabel('Load Demand (MW)')
plt.title('Load Duration Curve Comparison')
plt.legend()
plt.grid(True)
plt.show()

#The old load:
plt.figure(figsize=(10, 6))
plt.plot(hours, sorted_load_demand, color='blue', label='Task 5 Load Duration Curve')
plt.plot(hours, constant_load_duration_curve, color='red', linestyle='--', label='P_lim (0.637 MW)')
plt.xlabel('Hours (Sorted by Load)')
plt.ylabel('Load Demand (MW)')
plt.title('Load Duration Curve Comparison')
plt.legend()
plt.grid(True)
plt.show()


# %% TASK 15
# Task 15: Number of hours to be cut per year with P_lim = 0.4 MW

# Initialize a counter for hours exceeding the power flow limit
hours_congested = 0

# Iterate through the load duration curve
for load_value in new_sorted_load_demand:
    if load_value > P_lim:
        hours_congested += 1

# Print the number of hours per year that would require load reduction to avoid congestion
print(f"Number of hours per year requiring load reduction to avoid congestion (using new load data): {hours_congested} hours")

# Calculate the new coincidence factor
coincidence = estimated_max_overloading / (max_load + 0.4)
print("The coincidence factor with the new overloading load profile is:",coincidence)
# %%
