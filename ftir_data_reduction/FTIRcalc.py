# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 19:02:03 2015
@author: Dan

*REQUIREMENTS*
-Spectra files must start with @
-All MI names must be in input data file and the format of the MI name must be the same as the format used for the file name AND spectra name must be first column (e.g., SH63MI02c)
-MI input data MUST have FEO and FE2O3 columns (if you don't have iron speciation, just input FeOt in FeO and 0 in Fe2O3)
-Spectra saved as csv from OMNIC (all spectra must have the same wn)
-There must be a column labeled "Thickness" exactly (without quotes)
-Must have "Sample" as first column

***Make sure to check MI input data against output after peak measurements


Updates:

12/1/15 - Switched from cyclinging through all wn during peak height and background interpolation to find method used for c_left2
        - wn are now in a single list
12/10/15 - Switched from performing B-L calculations on each indiv spectra to peak averaging and calculating once per MI (12/1/15 was up-to date before this change)

11/16/16 - Change carbonate doublet peak finding routine. Now subtracting background to deal with sloping peaks

1/24/17 - Added thickness normalization to plots

11/29/17 -  Overhaul of carbonate peak measurement, Changed how spectra are accessed

3/1/18 - Changed 3500 peak

5/23/18 - Three backgrounds for CO2 selected (take std)

"""
###############################################################################
#%% IMPORT LIBRARIES
###############################################################################

import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.interpolate as interp
from scipy.optimize import leastsq
import scipy.ndimage.filters
import os
import pandas as pd
from numpy import sqrt

###############################################################################
# DEFINE VARIABLES AND FUNCTIONS
###############################################################################

mi_input_file = 'mi_inputdata\\test.csv'
spectra_folder = 'test'
filter_background = 1 # 0:no, 1: yes - apply a low-pass gaussian filter to background for background interpolation (currently total water and carbonate)


# VARIABLES
###########

mole_mass = {'SIO2': 60.08, 'TIO2': 79.866, 'AL2O3': 101.96, 'FE2O3': 159.69, 'FEO': 71.844, 'MNO': 70.9374, 'MGO': 40.3044, 'CAO': 56.0774, 'NA2O': 61.9789, 'K2O': 94.2, 'P2O5': 283.89, 'H2O': 18.01528, 'CO2': 44.01}
cation_count = {'SIO2': 1, 'TIO2': 1, 'AL2O3': 2, 'FE2O3': 2, 'FEO': 1, 'MNO': 1, 'MGO': 1, 'CAO': 1, 'NA2O': 2, 'K2O': 2, 'P2O5': 2}
oxygen_count = {'SIO2': 2, 'TIO2': 2, 'AL2O3': 3, 'FE2O3': 3, 'FEO': 1, 'MNO': 1, 'MGO': 1, 'CAO': 1, 'NA2O': 1, 'K2O': 1, 'P2O5': 5, 'H2O': 1}

# Partial molar volumes from R. Lange pers. commun. (1999) as cited in Luhr (2001) except TiO2 and Fe2O3, which are temp. dep. values from Lange and Carmichael MSA review paper
room_temp = 25 #during analysis (deg. C)
room_pressure = 1 #during analysis (bars)
par_molar_vol = {'SIO2': 27.01, 'TIO2': (23.16+7.24*(room_temp+273-1673)/1000-0.231*(room_pressure-1)/1000), 'AL2O3': 37.76, 'FE2O3': (42.13+9.09*(room_temp+273-1673)/1000-0.253*(room_pressure-1)/1000), 'FEO': 10.50,'MGO': 8.81, 'CAO': 13.03, 'NA2O': 19.88, 'K2O': 33.63, 'H2O': 13.93}
# Values left out of density calculation to consider adding: MnO, P2O5


# FUNCTIONS
###########

def silicateliqdensity(maj_order, mole_numbers, par_molar_vol, mole_mass):
    # Inputs
    #h20_moles - number of moles of h2o
    #mole_numbers - list containing major oxides mole numbers
    #mole_mass - dictionary of molecular masses
    #par_molar_vol - dictionary of partial molar volumes
    #mole_mass - dictionary of the molecular masses

    # Return
    #density in kg/m^3
    
    # Calculate mole fractions
    mole_frac = []
    for i in mole_numbers:
        mole_frac.append(i/sum(mole_numbers))

    # Calculate volume and gfw
    vol = []
    gfw = []
    count = 0
    for i in mole_frac:
        if maj_ele_order[count] in par_molar_vol:
            vol.append(i*par_molar_vol[maj_order[count]])
            gfw.append(i*mole_mass[maj_order[count]])
        else:
            vol.append(0)
            gfw.append(0)
        count += 1

    density = 1000*sum(gfw)/sum(vol)
    
    return density

def beer_lambert_calc(mol_mass, absorbance, density, thickness, abs_coef):
    # Inputs
    #mol_mass - molecular mass
    #absorbance - absorbance
    #density - density in kg/m^3
    #thickness - thickness in microns
    #abs_coef - absorption coefficient

    concen = 1000000*mol_mass*absorbance/(density*thickness*abs_coef)
    return concen

puff = 0 #go to minimum between 1430 and 1515 in bkgd interp, 0 = no, 1 = yes

def norm(x, mean, sd):
    norm = []
    for i in range(x.size):
        norm += [1.0/(sd*np.sqrt(2*np.pi))*np.exp(-(x[i] - mean)**2/(2*sd**2))]
    return np.array(norm)

def res(p, y, x):
    m, dm, sd1, sd2 = p
    m1 = m
    m2 = m1 + dm
    y_fit = norm(x, m1, sd1) + norm(x, m2, sd2)
    err = y - y_fit
    return err

def gaussian(x, height, center, width, offset):
    return height*np.exp(-(x - center)**2.0/(2*width**2)) + offset

def three_gaussians(x, h1, c1, w1, h2, c2, w2, h3, c3, w3, offset):
    return (gaussian(x, h1, c1, w1, offset=0) +
        gaussian(x, h2, c2, w2, offset=0) +
        gaussian(x, h3, c3, w3, offset=0) + offset)

errfunc3 = lambda p, x, y: (three_gaussians(x, *p) - y)**2.0

def lorentzian(x,p):
    min_step = x[1] - x[0]
    ind1 = (x <= p[1])
    x1 = x[ind1]
    ind2 = (x > p[1])
    x2 = x[ind2]
    numerator_left = (p[0]**2 )
    denominator_left = ( x1 - (p[1]) )**2 + p[0]**2
    numerator_right = (p[2]**2 )
    denominator_right = ( x2 - (p[1]) )**2 + p[2]**2
    y_left = p[3]*(numerator_left/denominator_left)
    y_right = p[3]*(numerator_right/denominator_right)
    lin_comb = np.hstack((y_left,y_right))
    return lin_comb

def three_lorentzians(x, p):
    return (lorentzian(x, p[0:4]) +
        lorentzian(x, p[4:8]) +
        lorentzian(x, p[8:12]))

def residuals(p,y,x):
    err = y - three_lorentzians(x,p)
    return err

###############################################################################
#%% IMPORT AND SORT DATA
###############################################################################

# Import spectra

spectra_names = [] #names of csv files
wn_list = 0 #wavenumbers of all samples in a single list
ab_list = [] #absorbance of all samples
interp_wn = []
interp_ab = [] #background interpolation
interp_results = [] #carbonate interpolation function
ab_min = [] #list of background absorbance measurements under spectra peaks
mi_peaks = [] #peak measurements
mi_peaklocations = [] #peak locations
mi_peakheights = [] #peak heights
mi_inputdata = [] #contains information on MI thickness and composition
mi_id_headers = [] #MI input data column headers
mi_list = [] #List of melt inclusions
directories = [] #List of all spectra file names

# Import thickness and composition data

with open (mi_input_file, 'r') as f:
    reader = csv.reader(f)
    count = 0
    for row in reader:
        if count == 0:
            mi_id_headers = row
        else:
            temp = [row[0]]
            temp.extend(map(float, row[1:]))
            mi_inputdata.append(temp)
        count += 1

mi_inputdata = sorted(mi_inputdata)

for i in mi_inputdata:
    mi_list.append(i[0].upper())

for i in mi_list:
    ab_list.append([])
    mi_peaks.append([])
    mi_peaklocations.append([])
    mi_peakheights.append([])
    interp_wn.append([])
    interp_ab.append([])
    interp_results.append([])
    ab_min.append([])

# Import spectra

for i in os.listdir('spectra_csv/'+spectra_folder):
    directories.append(i)

directories = sorted(directories)

count = 0
for i in directories:
    for j in range(len(i)):
        if i[1:-j].upper() in mi_list:
            spectra_names.append(i[1:-j+1])
            mi_index = mi_list.index(i[1:-j].upper())
            break
    if i[1:-j] != '':
        d = pd.read_csv('spectra_csv\\%s\\%s' %(spectra_folder,i), names=['wavenumber','absorb'])
        if count == 0:
            wn_list = d.wavenumber.tolist()
            
        #THIS PART IS MEANT TO IDENTIFY SPECTRA WITH UNEXPECTED WAVE NUMBERS
        if count != 0 and len(d.absorb.tolist()) != 4176:
            print(i)
            print('Length not 4176')
            abs_list = []
            wnl = d.wavenumber.tolist()
            ab = d.absorb.tolist()
            for wn in wn_list:
                abs_pull = ab[min(range(len(wnl)), key=lambda i: abs(wnl[i]-wn))]
                abs_list.append(abs_pull)
        else:
            abs_list = d.absorb.tolist()
        ab_list[mi_index].append(abs_list)
        if len(abs_list) != 4176:
            print(i)
            print('Abs list is ' + str(len(abs_list)) + ' entries. Expecting 4176.')
        count += 1
    #print str(count)+' of '+str(len(directories))+' uploaded'

count = 0
missing = []
for i in ab_list:
    if len(i) == 0:
        missing.append(mi_list[count])
    count += 1

if len(missing) > 0:
    print('Missing spectra for the following MI')
    print(missing)

###############################################################################
# PEAK HEIGHTS
###############################################################################

wn_arms = [] #Regions of FTIR spectra used for background interpolation (ordered list: 0 - total H2O, 1 - carbonate, 2 - mole/hydroxyl)
peak_region = [] # Regions of FTIR spectra to search for peaks (ordered list: 0 - total H2O, 1 - carbonate, 2 - mole/hydroxyl)

# Total water peak (3550)
##################################################

# Peak region
tw_peakleft = 3420
tw_peakright = 3590

# Select arms
tw_left1 = 2000 #temp changed from 2250
tw_left2 = 2500
tw_right1 = 3850 
tw_right2 = 4350 #temp changed from 4200

# Carbonate doublet (1430 and 1515)
##################################################

# Peak regions
c_1430left = 1410
c_1430right = 1450
c_1515left = 1485
c_1515right = 1535

# Select arms
c_left1 = 1240
c_left2 = 1360
c_right1 = 1755
c_right2 = 2000

# Molecular water and OH peaks (5200 and 4500)
##################################################

# Peak region
w_4500left = 4465
w_4500right = 4515
w_5200left = 5180
w_5200right = 5220

# Select arms
w_left1 = 4070
w_left2 = 4270
w_right1 = 5360
w_right2 = 5560

# Lists et al.
##################################################

peak_leftbounds = [tw_peakleft, c_1430left, c_1515left, w_4500left, w_5200left]
peak_rightbounds = [tw_peakright, c_1430right, c_1515right, w_4500right, w_5200right]

#For background subtraction...
x_ind = [[],[min(range(len(wn_list)), key=lambda i: abs(wn_list[i]-peak_leftbounds[1])), min(range(len(wn_list)), key=lambda i: abs(wn_list[i]-peak_rightbounds[1]))],[min(range(len(wn_list)), key=lambda i: abs(wn_list[i]-peak_leftbounds[2])), min(range(len(wn_list)), key=lambda i: abs(wn_list[i]-peak_rightbounds[2]))]]

tw_list_wn_leftin1 = min(range(len(wn_list)), key=lambda i: abs(wn_list[i]-tw_left1))
tw_list_wn_rightin1 = min(range(len(wn_list)), key=lambda i: abs(wn_list[i]-tw_left2))
tw_list_wn_leftin2 = min(range(len(wn_list)), key=lambda i: abs(wn_list[i]-tw_right1))
tw_list_wn_rightin2 = min(range(len(wn_list)), key=lambda i: abs(wn_list[i]-tw_right2))
w_list_wn_leftin1 = min(range(len(wn_list)), key=lambda i: abs(wn_list[i]-w_left1))
w_list_wn_rightin1 = min(range(len(wn_list)), key=lambda i: abs(wn_list[i]-w_left2))
w_list_wn_leftin2 = min(range(len(wn_list)), key=lambda i: abs(wn_list[i]-w_right1))
w_list_wn_rightin2 = min(range(len(wn_list)), key=lambda i: abs(wn_list[i]-w_right2))

tw_list_wn = wn_list[tw_list_wn_leftin1:tw_list_wn_rightin1] + wn_list[tw_list_wn_leftin2:tw_list_wn_rightin2]
w_list_wn = wn_list[w_list_wn_leftin1:w_list_wn_rightin1] + wn_list[w_list_wn_leftin2:w_list_wn_rightin2]

tw_x = wn_list[tw_list_wn_rightin1:tw_list_wn_leftin2]
w_x = wn_list[w_list_wn_rightin1:w_list_wn_leftin2]

# Find peaks
##################################################
testsol = []
for sample in range(len(mi_list)):
    count_replicate = 0
    for replicate in ab_list[sample]:
        temp_peak = []
        temp_loc = []
        for j in range(5):
            if j in [1,2]:
                x = [wn_list[x_ind[j][0]],wn_list[x_ind[j][1]]]
                y = [replicate[x_ind[j][0]], replicate[x_ind[j][1]]]
                coef = np.polyfit(x,y,1)
                poly = np.poly1d(coef)
                new_abs = 0
                temp_abs = 0
                for jj in wn_list[x_ind[j][0]:x_ind[j][1]+1]:
                    absorb = replicate[wn_list.index(jj)]-poly(jj)
                    if absorb > new_abs:
                        new_abs = absorb
                        temp_abs = replicate[wn_list.index(jj)]
                if temp_abs > 0:
                    temp_peak.append(temp_abs)
                    testsol.append(temp_abs)
                else:
                    temp_peak.append(max(replicate[min(range(len(wn_list)), key=lambda i: abs(wn_list[i]-peak_leftbounds[j])):min(range(len(wn_list)), key=lambda i: abs(wn_list[i]-peak_rightbounds[j]))]))
            else:
                temp_peak.append(max(replicate[min(range(len(wn_list)), key=lambda i: abs(wn_list[i]-peak_leftbounds[j])):min(range(len(wn_list)), key=lambda i: abs(wn_list[i]-peak_rightbounds[j]))]))
            temp_loc.append(ab_list[sample][count_replicate].index(temp_peak[j]))
        temp_peak[0] = np.mean(replicate[temp_loc[0]-1:temp_loc[0]+2])#3-1 addtion
        mi_peaks[sample].append(temp_peak)
        mi_peaklocations[sample].append(temp_loc) #Location is the index of absorbance maximum
        count_replicate += 1

print('Peak finding complete.')

# Interpolate background
##################################################

for sample in range(len(mi_list)):
    for count_replicate in range(len(ab_list[sample])):

        #Choose left bound for carbonate doublet (if not entered above)
        if c_left2 == 0:
            c_left2 = wn_list[ab_list[sample][count_replicate].index(min(ab_list[sample][count_replicate][min(range(len(wn_list)), key=lambda i: abs(wn_list[i]-c_left1)):mi_peaklocations[sample][count_replicate][1]]))]
        
        #Find bounds
        c_list_wn_leftin1 = min(range(len(wn_list)), key=lambda i: abs(wn_list[i]-c_left1))
        c_list_wn_rightin1 = min(range(len(wn_list)), key=lambda i: abs(wn_list[i]-c_left2))
        c_list_wn_leftin2 = min(range(len(wn_list)), key=lambda i: abs(wn_list[i]-c_right1))
        c_list_wn_rightin2 = min(range(len(wn_list)), key=lambda i: abs(wn_list[i]-c_right2))
        c_list_wn = wn_list[c_list_wn_leftin1:c_list_wn_rightin1] + wn_list[c_list_wn_leftin2:c_list_wn_rightin2]
        c_x = wn_list[c_list_wn_rightin1:c_list_wn_leftin2]
        
        #Apply Gaussian filter to background for total water interp.
        if filter_background == 1:
            d_filt = scipy.ndimage.filters.gaussian_filter(ab_list[sample][count_replicate], 17, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
            d_filt2 = scipy.ndimage.filters.gaussian_filter(ab_list[sample][count_replicate], 4, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
            d_filt = d_filt.tolist()
            d_filt2 = d_filt2.tolist()
        else:
            d_filt = ab_list[sample][count_replicate]
            d_filt2 = ab_list[sample][count_replicate]
        
        #Find background for interpolation
        tw_list_ab = d_filt[tw_list_wn_leftin1:tw_list_wn_rightin1]+d_filt[tw_list_wn_leftin2:tw_list_wn_rightin2]
        c_list_ab = d_filt2[c_list_wn_leftin1:c_list_wn_rightin1]+d_filt2[c_list_wn_leftin2:c_list_wn_rightin2]
        w_list_ab = ab_list[sample][count_replicate][w_list_wn_leftin1:w_list_wn_rightin1]+ab_list[sample][count_replicate][w_list_wn_leftin2:w_list_wn_rightin2]
        
        min_4500_5200_ab = min(ab_list[sample][count_replicate][mi_peaklocations[sample][count_replicate][3]:mi_peaklocations[sample][count_replicate][4]])
        min_4500_5200_wn = wn_list[ab_list[sample][count_replicate].index(min_4500_5200_ab)]
        
        min_1515_1630_ab = min(ab_list[sample][count_replicate][mi_peaklocations[sample][count_replicate][2]:min(range(len(wn_list)), key=lambda i: abs(wn_list[i]-1630))])
        min_1515_1630_wn = wn_list[ab_list[sample][count_replicate].index(min_1515_1630_ab)]
        
        x_interp = []
        y_interp = []
        min_interp = []
        interp_f = []

        #Total water interpolation
        f = interp.interp1d(tw_list_wn,tw_list_ab, kind = 'cubic')
        x_interp.append(tw_x)
        y_interp.append(f(tw_x))
        min_interp.append(f(wn_list[mi_peaklocations[sample][count_replicate][0]]))
        interp_f.append(f)
        
        #Carbonate interpolation
        f = interp.interp1d(c_list_wn,c_list_ab, kind = 'cubic') # +[min_1515_1630_wn] +[min_1515_1630_ab]
        x_interp.append(c_x)
        y_interp.append(f(c_x))
        min_interp.append(f(wn_list[mi_peaklocations[sample][count_replicate][1]]))
        min_interp.append(f(wn_list[mi_peaklocations[sample][count_replicate][2]]))
        interp_f.append(f)
        
        #Mole. water and hydroxyl interpolation
        f = interp.interp1d(w_list_wn+[min_4500_5200_wn],w_list_ab+[min_4500_5200_ab], kind = 'cubic')
        x_interp.append(w_x)
        y_interp.append(f(w_x))
        min_interp.append(f(wn_list[mi_peaklocations[sample][count_replicate][3]]))
        min_interp.append(f(wn_list[mi_peaklocations[sample][count_replicate][4]]))
        interp_f.append(f)
        
        #Add results of interpolation to lists
        interp_wn[sample].append(x_interp)
        interp_ab[sample].append(y_interp)
        ab_min[sample].append(min_interp)
        interp_results.append(interp_f)
            
print('Background interpolation complete.')

co2_peaks = []
carb_plot = []
max_loc = []
shifts = [[-6,0,0,0,0,0],[0,6,0,0,0,0],[0,0,-3,-6,-9,6],[0,0,0,0,0,0]]


# Calculate peak heights
##################################################

for sample in range(len(mi_list)):
    print('Peak height calculation for: '+ mi_list[sample])
    print(str(sample+1)+' of '+str(len(mi_list)))
    co2_peaks.append([])
    carb_plot.append([])
    max_loc.append([])
    for replicate in range(len(ab_list[sample])):
        co2_peaks[sample].append([[],[]]) #First list is 1430, second is 1515, 0 = Method 1, 1 = Method 2, 2 = Method 3
        carb_plot[sample].append([])
        max_loc[sample].append([])
        temp_heights = []
        for i in range(5):
            if i == 1:
                temp = []
                temp1 = []
                temp2 = []
                temp3 = []
                for j in range(len(shifts[0])):
                    #Get data, subtract background
                    L1 = c_list_wn_leftin1 + shifts[0][j]
                    L2 =  c_list_wn_leftin2 + shifts[1][j]
                    R1 =  c_list_wn_rightin1 + shifts[2][j]
                    R2 =  c_list_wn_rightin2 + shifts[3][j]
                    x = np.array(wn_list[L1:R2+1])
                    ab = ab_list[sample][replicate][L1:R2+1]
                    
                    c_list_wn = wn_list[L1:R1+1] + wn_list[L2:R2+1]
                    c_list_ab = ab_list[sample][replicate][L1:R1+1]+ab_list[sample][replicate][L2:R2+1]
                    min_1515_1630_ab = min(ab_list[sample][replicate][mi_peaklocations[sample][replicate][2]:min(range(len(wn_list)), key=lambda i: abs(wn_list[i]-1630))])
                    min_1515_1630_wn = wn_list[ab_list[sample][replicate].index(min_1515_1630_ab)]
                    
                    if puff == 1:
                        f = interp.interp1d(c_list_wn+[min_1515_1630_wn],c_list_ab+[min_1515_1630_ab], kind = 'cubic')
                    else:
                        f = interp.interp1d(c_list_wn,c_list_ab, kind = 'cubic')
                    bkgd = f(x)
                    
                    ab_nobkgd = np.array(ab)-bkgd
                    
                    #Method 1: Simple maximum
                    select1 = max(ab_nobkgd[np.where(np.logical_and(x>=c_1430left, x<=c_1430right))])
                    temp1.append(select1)
                    select2 = max(ab_nobkgd[np.where(np.logical_and(x>=c_1515left, x<=c_1515right))])
                    
                    temp2.append(select2) #ab_list[sample][replicate][mi_peaklocations[sample][replicate][2]]-bkgd[(np.abs(x-wn_list[mi_peaklocations[sample][replicate][2]])).argmin()]
                        #co2_peaks[sample][replicate][ii].append(mi_peaks[sample][replicate][i+ii]-ab_min[sample][replicate][i+ii])
                    if j == 0:
                        temp_heights.append(select1)
                        temp_heights.append(select2)
                    temp3.append([x[np.abs(ab_nobkgd-select1).argmin()],select1,x[np.abs(ab_nobkgd-select2).argmin()],select2])
                    
                    #Other methods at https://stackoverflow.com/questions/10143905/python-two-curve-gaussian-fitting-with-non-linear-least-squares
                    
                    #Gaussian peak fitting
                    
                    #Method 2: Simple Gaussian 3 Peak - https://stackoverflow.com/questions/26936094/python-load-data-and-do-multi-gaussian-fit
                    c1 = wn_list[mi_peaklocations[sample][replicate][1]]
                    c2 = wn_list[mi_peaklocations[sample][replicate][2]]
                    h1 = mi_peaks[sample][replicate][i]-ab_min[sample][replicate][i]
                    h2 = mi_peaks[sample][replicate][i+1]-ab_min[sample][replicate][i+1]
                    guess3 = [h1, c1, 20, h2, c2, 20, 0.1, 1630, 20, 0] #h1, c1, w1, etc, etc, offset
                    optim3, success = leastsq(errfunc3, guess3[:], args=(x, ab_nobkgd))
    
                    temp1.append(optim3[0])
                    temp2.append(optim3[3])
                    
                    #Method 3: Simple Lorentzian 3 Peak - http://mesa.ac.nz/2012/03/python-workshop-ii-fancy-background-and-asymmetric-peakfitting/
                    hw = 30
                    p = [hw, c1, hw, h1, hw, c2, hw, h2, hw, 1630, hw, 0.1]  # [hwhm1, peak center, hwhm2, intensity] 
                    pbest = leastsq(residuals,p,args=(ab_nobkgd,x),full_output=1)
                    best_parameters = pbest[0]
                    fit = three_lorentzians(x,best_parameters)
                    
                    temp1.append(best_parameters[3])
                    temp2.append(best_parameters[7])
                    temp.append([x,ab,bkgd,optim3,best_parameters]) #x, ab, background, optim3, best_parameters
                co2_peaks[sample][replicate][0] += temp1
                co2_peaks[sample][replicate][1] += temp2
                carb_plot[sample][replicate] += temp
                max_loc[sample][replicate] += temp3
            elif i != 2:
                height = mi_peaks[sample][replicate][i]-ab_min[sample][replicate][i]
                if height > 0:
                    temp_heights.append(height)
                else:
                    temp_heights.append(0)
        if temp_heights[1] < 0.0 or temp_heights[2] < 0.0: #11-29-17 was 0.005, don't know why, changed to 0.0
            temp_heights[1] = 0
            temp_heights[2] = 0
        mi_peakheights[sample].append(temp_heights)



###############################################################################
#%% CREATE PDF OUTPUT OF SPECTRA - NORMALIZED
###############################################################################

import matplotlib as mpl
import matplotlib.cm as cm

mpl.style.use('classic')

norm_thick = 25.0

colors = cm.gist_rainbow(np.linspace(0, 1, len(shifts[0])))

ax_limits = [[1250,4500,0,2],[1300,1800,0,2]]
ax_limits_low_bkgd = [[-1,2], [-1,1]]
ax_limits_high_water =[[0,4], [0,2]]
with PdfPages('output//test.pdf') as pdf:
    start = 0 #0
    end = len(mi_list) #len(mi_list)
    for sample in range(start,end):
        print(str(sample-start+1)+' of '+str(end-start))
        n_fact = norm_thick/mi_inputdata[sample][mi_id_headers.index('Thickness')]
        if len(ab_list[sample])+1 > 2+len(shifts[0]):
            grid_size = (len(ab_list[sample])+1, 2+len(shifts[0]))
        else:
            grid_size = (4,2+len(shifts[0]))
        for replicate in range(len(ab_list[sample])):
            if replicate == 4:
                break
            a = np.array(carb_plot[sample][replicate])
            loc_max = max_loc[sample][replicate]
            xA = a[:,0]
            abA = a[:,1]
            bkgdA = a[:,2]
            optim3A = a[:,3]
            best_parametersA = a[:,4]
            for j in range(2+len(shifts[0])):
                plt.subplot2grid(grid_size, (replicate,j), rowspan=1, colspan=1)
                if j == 0:
                    # Plot original spectrum
                    plt.plot(wn_list,np.array(ab_list[sample][replicate])*n_fact, lw = 0.5, c = 'k')
                    # Plot background interpolation
                    for n in range(3):
                        plt.plot(interp_wn[sample][replicate][n], np.array(interp_ab[sample][replicate][n])*n_fact, lw = 0.5, c = 'b')
                    for n in range(5):
                        x_temp = [wn_list[mi_peaklocations[sample][replicate][n]], wn_list[mi_peaklocations[sample][replicate][n]]]
                        y_temp = np.array([mi_peaks[sample][replicate][n], ab_min[sample][replicate][n]])*n_fact
                        plt.plot(x_temp, y_temp, lw = 0.5, c = 'r')
                    plt.xlim(ax_limits[j][1], ax_limits[j][0])
                    if ab_list[sample][replicate][min(range(len(wn_list)), key=lambda i: abs(wn_list[i]-4000))] < 0:
                        plt.ylim(ax_limits_low_bkgd[j][0], ax_limits_low_bkgd[j][1])
                    elif mi_peaks[sample][replicate][0] > 2:
                        plt.ylim(ax_limits_high_water[j][0], ax_limits_high_water[j][1])
                    else:
                        plt.ylim(ax_limits[j][2], ax_limits[j][3])
                    if replicate == 0:
                        plt.title('Normalized spectra')
                if j == 1: #plot doublet with background interp
                    plt.plot(xA[0],abA[0],c = 'k',lw =2.5)
                    plt.plot(interp_wn[sample][replicate][1],interp_ab[sample][replicate][1], lw = 2, c = 'k', label = 'Filt_Bkgd',linestyle='--')
                    #plt.plot(x,bkgd,c='k', lw = 2, label='NoFilt_Bkgd',linestyle='--')
                    plt.xlim([1900,1200])
                    #plt.gca().invert_xaxis()
                    for i in range(len(shifts[0])):
                        plt.plot(xA[i],bkgdA[i],color = colors[i],label=str(i))
                    if replicate == 0:
                        plt.legend(loc = 2, fontsize = 10)
                        plt.title('Carbonate doublet')
                elif j > 1:
                    plt.scatter(xA[j-2],abA[j-2]-bkgdA[j-2],c='darkgray',marker='o',linewidth=0, s=35)
                    #plt.gca().invert_xaxis()
                    plt.xlim([1800,1300])
                    if j > 1 : #== 2
                        plt.plot(xA[j-2], three_gaussians(xA[j-2], *optim3A[j-2]), c = 'k', lw = 2)
                        for i in range(2):
                            plt.plot([loc_max[j-2][i*2],loc_max[j-2][i*2]],[0,loc_max[j-2][i*2+1]], c = 'r', lw = 1.5) #[wn_list[mi_peaklocations[sample][replicate][1+i]],wn_list[mi_peaklocations[sample][replicate][1+i]]],[0,co2_peaks[sample][replicate][i][0]]
                        for i in range(3):
                            plt.plot(xA[j-2], gaussian(xA[j-2], *np.append(optim3A[j-2][3*i:3*i+3],optim3A[j-2][-1])), c = 'k', lw = 1)
                        if replicate == 0:
                            plt.title('Gaussian n='+str(j-2))
#                    else:
#                        plt.plot(x, three_lorentzians(x,best_parameters), c = 'k', lw = 2)
#                        for i in range(3):
#                            plt.plot(x, lorentzian(x, best_parameters[4*i:4*i+4]), c = 'k', lw = 1)
#                        if replicate == 0:
#                            plt.title('Asym. Lorenzian')
                plt.xlabel('Wavenumber', fontweight='bold')
                plt.ylabel('Absorbance', fontweight='bold')
        
                
        plt.suptitle(mi_list[sample], size = 30)
        fig = plt.gcf()
        fig.set_size_inches(36,12, forward = True)
        if len(ab_list[sample]) == 1:
            y_loc = 0.4
        elif len(ab_list[sample]) == 2:
            y_loc = 0.3
        elif len(ab_list[sample]) == 3:
            y_loc = 0.05
        elif len(ab_list[sample]) == 4:
            y_loc = 0.0
        fig.text(0,y_loc,'Wafer thickness is normalized to 25 microns')
        plt.tight_layout(rect = [0,-0.15,1,0.98], pad = 1, w_pad = 1, h_pad = 0)
        
        plt.gcf().set_facecolor('white') 
        pdf.savefig(facecolor = 'w', orientation = 'portrait', papertype = 'letter')
        #plt.show()
        plt.close()
plt.clf()

#%% CSV with samples and subsamples

with open('output\\test.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    flag = 0
    for i in spectra_names:
        if flag == 0:
            writer.writerow(['Spectrum','Bkgd1','Bkgd2','Bkgd3','GAUorMAX1','GAUorMAX2','GAUorMAX3','Delete','Abs=0'])
            writer.writerow([i,0,1,2,0,0,0,0,0])
            flag = 1
        else:
            writer.writerow([i,0,1,2,0,0,0,0,0])

#%% CSV with samples and subsamples
with open('output\\test.csv', 'r') as f:
    reader = csv.reader(f)
    flag = 0
    for row in reader:
        if flag == 0:
            flag = 1
#        elif mi_list.index(row[0][:-1].upper()) >= 125:
        else:
            sample = mi_list.index(row[0][:-1].upper())
            if row[0][-1:] == 'a':
                replicate = 0
            elif row[0][-1:] == 'b':
                replicate = 1
            elif row[0][-1] == 'c':
                replicate = 2
            elif row[0][-1] == 'd':
                replicate = 3
            elif row[0][-1] == 'e':
                replicate = 4
            else:
                print('error')
                print(row[0])
                break
            bkgd1 = int(row[1]) #which background
            bkgd2 = int(row[2])
            bkgd3 = int(row[3])
            G_M1 = int(row[4]) #Gaussian (0) or local max (1)
            G_M1 = abs(1-G_M1)
            G_M2 = int(row[5]) #Gaussian (0) or local max (1)
            G_M2 = abs(1-G_M2)
            G_M3 = int(row[6]) #Gaussian (0) or local max (1)
            G_M3 = abs(1-G_M3)
            D = int(row[7]) #Delete? - currently not in use
            A = int(row[8]) #Set abs to zero
            #co2_peaks [sample] [replicate] [1430,1515] [18 = 6x 0 = max, 1 = Gaus, 2 = Loren]
            if A == 0:
#                mi_peakheights[sample][replicate][1] = co2_peaks[sample][replicate][0][bkgd*3+G_M]
#                mi_peakheights[sample][replicate][2] = co2_peaks[sample][replicate][1][bkgd*3+G_M]
                mi_peakheights[sample][replicate][1] = [co2_peaks[sample][replicate][0][bkgd1*3+G_M1],co2_peaks[sample][replicate][0][bkgd2*3+G_M2],co2_peaks[sample][replicate][0][bkgd3*3+G_M3]]
                mi_peakheights[sample][replicate][2] = [co2_peaks[sample][replicate][1][bkgd1*3+G_M1],co2_peaks[sample][replicate][1][bkgd2*3+G_M2],co2_peaks[sample][replicate][1][bkgd3*3+G_M3]]
            elif A == 1:
                mi_peakheights[sample][replicate][1] = [0,0,0]
                mi_peakheights[sample][replicate][2] = [0,0,0]
            else:
                print('error')
                print(row[0])
                break

###############################################################################
#%% Create CSV with peak height measurements
###############################################################################

# Before preceeding to Beer-Lamber calculation, look at spectra and make necessary changes in spectra info csv

peakheight_output = []

peakheight_output.append(['Sample', 'Spectrum', '3550','1430-1','1430-2','1430-3','1515-1','1515-2','1515-3','4500','5200']+mi_id_headers[1:])

count = 0
count2 = 0
for sample in mi_peakheights:
    peakheight_output.append([mi_list[count2]]+['','','','','','','','','','']+mi_inputdata[count2][1:])
    for replicate in sample:
        peakheight_output.append(['', spectra_names[count]]+[replicate[0]]+replicate[1]+replicate[2]+replicate[3:])
        count += 1
    count2 += 1
        #Now make a single tiered list consisting of:
        #1st row is headers
        #Spectra name, 5 peak heights, thickness, composition

with open('output\\test_peakheights.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(peakheight_output)




###############################################################################
#%% IMPORT DATA FROM CSV
###############################################################################

mi_spectra_file = 'output\\test_peakheights.csv'
mi_peakheights = []
spectra_names = []
mi_inputdata = []

with open (mi_spectra_file, 'r') as f:
    reader = csv.reader(f)
    count = 0
    sample = -1
    for row in reader:
        if count == 0:
            mi_id_headers = [row[0]]+row[11:]
            #print 'skip define mi_id_headers'
        elif row[1] == '':
            sample += 1
            mi_inputdata.append( [row[0]]+[float(i) for i in row[11:]] )
            mi_peakheights.append([])
        else:
            mi_peakheights[sample].append( [float(row[2])] + [[float(i) for i in row[3:6]]]+[[float(i) for i in row[6:9]]]+[float(i) for i in row[9:11]] )
            spectra_names.append(row[1])
        count += 1


###############################################################################
#%% BEER-LAMBERT CALCUATION
###############################################################################

# SET ABSORPTION COEFFICIENTS

# Calculate mole numbers and cation numbers

mole_numbers = []
cation_numbers = []
major_ele_conc = []
maj_ele_order = []

sd_list = []
avg_peaks_list = []

count = 0
for i in mi_inputdata:
    temp = [] #Holds mole numbers for a melt inclusion
    temp2 = [] #Holds cation numbers for a melt inclusion
    temp3 = [] #List of major elements for a melt inclusion
    for j in range(len(i)):
        if mi_id_headers[j].upper() in mole_mass:
            temp.append(i[j]/mole_mass[mi_id_headers[j].upper()])
            temp2.append(i[j]/mole_mass[mi_id_headers[j].upper()]*cation_count[mi_id_headers[j].upper()])
            temp3.append(i[j])
            if count == 0:
                maj_ele_order.append(mi_id_headers[j].upper())
    temp.append(0.0) #Place holder for H2O
    mole_numbers.append(temp)
    cation_numbers.append(temp2)
    if sum(temp3) < 95:
        print('WARNING: %s has a low total' % i[0])
        print(sum(temp3))
    major_ele_conc.append(temp3)
    count += 1

maj_ele_order.append('H2O')

# Calculate absorption coefficients

# E 3550 is from Dixon et al. (1997) and Shishkina et al. (2010)
# E 1525, 4500, 5200 from Mandeville et al. (2002) and Ohlhorst et al. (2001)

abs_coef = [] #List of absorption coefficients

count = 0
for i in cation_numbers:
    temp = [] #Holds absorption coefficients for a single melt inclusion
    SiAldivTot = (i[maj_ele_order.index('SIO2')]+i[maj_ele_order.index('AL2O3')])/sum(i)
    NadivNaCa = i[maj_ele_order.index('NA2O')]/(i[maj_ele_order.index('NA2O')]+i[maj_ele_order.index('CAO')])
    
    #0 - 3550 (Dixon)- GOES TO OUTPUT  
    temp.append(63.0)
    #1 - 3550 (Shishkina)
    temp.append(68.0)
    #2,3 - 1525 (for 1430 and 1515) - Dixon and Pan (1995) - GOES TO OUTPUT
    temp.append(451.2-(341.8*NadivNaCa))
    temp.append(451.2-(341.8*NadivNaCa))
    #4 - 4500 (Mandeville)
    temp.append((4.054*SiAldivTot)-2.026)
    #5 - 5200 (Mandeville)
    temp.append((4.899*SiAldivTot)-2.463)
    #6 - 4500 (Ohlhorst)
    temp.append((0.000257*(major_ele_conc[count][maj_ele_order.index('SIO2')])**2)-0.13)
    #7 - 5200 (Ohlhorst)
    temp.append((0.000304*(major_ele_conc[count][maj_ele_order.index('SIO2')])**2)-0.15)
    
    abs_coef.append(temp)
    count += 1
    
# Iteratively calculate H2O and CO2 concentrations with changing density

H2O_CO2_concen = []
density_values = []
h2o = 0

count = 0
for sample in range(len(mi_peakheights)):
    H2O_CO2_concen.append([])
    #density_values.append([]) - used for old version (B-L calc for each replicate, averaging after)
    #h2o_values = [] - used for old version (B-L calc for each replicate, averaging after)
    #co2_values = [] - used for old version (B-L calc for each replicate, averaging after)
    thickness = mi_inputdata[sample][mi_id_headers.index('Thickness')]
    ab_coef = abs_coef[sample]
    peaks = [[],[],[],[]] #for abs averaging method
    average_peaks = [] #for abs averaging method
    peak_sd = [] #for abs averaging method
    for replicate in range(len(mi_peakheights[sample])):
    #start new way - H2O calc once per mi (abs average)
        shift = 0
        for i in range(5):
            if i not in [1,2]:
                peaks[i-shift].append(mi_peakheights[sample][replicate][i])
            else:
                peaks[i-shift] += mi_peakheights[sample][replicate][i]
                if i == 1:
                    shift = 1
    for i in peaks:
        average_peaks.append(sum(i)/len(i))
        peak_sd.append(np.std(i, ddof=1))
        print(i)
    
    concentrations = []
    
    for i in range(10):
        if i > 0:
            mole_numbers[sample][-1] = h2o/mole_mass['H2O']
        # Density calc completed using measurements of 3550 (Dixon)
        density = silicateliqdensity(maj_ele_order, mole_numbers[sample], par_molar_vol, mole_mass)
        h2o = beer_lambert_calc(mole_mass['H2O'], average_peaks[0], density, thickness, ab_coef[0])
    #3550 (Dixon) -0
    concentrations.append(h2o)
    #3550 (Shishkina) -1
    concentrations.append(beer_lambert_calc(mole_mass['H2O'], average_peaks[0], density, thickness, ab_coef[1]))
    #1430 and 1515 -2
    carb = beer_lambert_calc(mole_mass['CO2'], average_peaks[1], density, thickness, ab_coef[2])
    concentrations.append(carb)
    #1515 -3
#    c1515 = beer_lambert_calc(mole_mass['CO2'], average_peaks[2], density, thickness, ab_coef[3])
#    concentrations.append(c1515)
    #4500 (Mandeville) -3
    concentrations.append(beer_lambert_calc(mole_mass['H2O'], average_peaks[2], density, thickness, ab_coef[4]))
    #5200 (Mandeville) -4
    concentrations.append(beer_lambert_calc(mole_mass['H2O'], average_peaks[3], density, thickness, ab_coef[5]))
    #Total H2O Mandeville -5
    concentrations.append(concentrations[-1]+concentrations[-2])
    #4500 (Ohlhorst) -6
    concentrations.append(beer_lambert_calc(mole_mass['H2O'], average_peaks[2], density, thickness, ab_coef[6]))
    #5200 (Ohlhorst) - 7
    concentrations.append(beer_lambert_calc(mole_mass['H2O'], average_peaks[3], density, thickness, ab_coef[7]))
    #Total H2O Ohlhorst - 8
    concentrations.append(concentrations[-1]+concentrations[-2])
        
    H2O_CO2_concen[sample].append(concentrations)
    density_values.append(density)
    
    major_ele_conc[sample].append(h2o)
    major_ele_conc[sample].append(carb)
    sd_list.append(peak_sd)
    avg_peaks_list.append(average_peaks)



    #     #old way - H2O calc for each replicate (this goes at top of for replicate loop)
    #     concentrations = []
    #     for i in range(10):
    #         if i > 0:
    #             mole_numbers[count][-1] = h2o/mole_mass['H2O']
    #         # Density calc completed using measurements of 3550 (Dixon)
    #         density = silicateliqdensity(maj_ele_order, mole_numbers[count], par_molar_vol, mole_mass)
    #         h2o = beer_lambert_calc(mole_mass['H2O'], mi_peakheights[sample][replicate][0], density, thickness, ab_coef[0])
    #     #3550 (Dixon) 
    #     concentrations.append(h2o)
    #     #3550 (Shishkina)
    #     concentrations.append(beer_lambert_calc(mole_mass['H2O'], mi_peakheights[sample][replicate][0], density, thickness, ab_coef[1]))
    #     #1430
    #     c1430 = beer_lambert_calc(mole_mass['CO2'], mi_peakheights[sample][replicate][1], density, thickness, ab_coef[2])
    #     concentrations.append(c1430)
    #     #1515
    #     c1515 = beer_lambert_calc(mole_mass['CO2'], mi_peakheights[sample][replicate][2], density, thickness, ab_coef[3])
    #     concentrations.append(c1515)
    #     #4500 (Mandeville)
    #     concentrations.append(beer_lambert_calc(mole_mass['H2O'], mi_peakheights[sample][replicate][3], density, thickness, ab_coef[4]))
    #     #5200 (Mandeville)
    #     concentrations.append(beer_lambert_calc(mole_mass['H2O'], mi_peakheights[sample][replicate][4], density, thickness, ab_coef[5]))
    #     #4500 (Ohlhorst)
    #     concentrations.append(beer_lambert_calc(mole_mass['H2O'], mi_peakheights[sample][replicate][3], density, thickness, ab_coef[6]))
    #     #5200 (Ohlhorst)
    #     concentrations.append(beer_lambert_calc(mole_mass['H2O'], mi_peakheights[sample][replicate][4], density, thickness, ab_coef[7]))
        
    #     h2o_values.append(h2o)
    #     co2_values.append((c1430+c1515)/2)
        
    #     H2O_CO2_concen[sample].append(concentrations)
    #     density_values[sample].append(density)
        
    # major_ele_conc[sample].append(sum(h2o_values)/len(h2o_values))
    # major_ele_conc[sample].append(sum(co2_values)/len(co2_values))
    


maj_ele_order.append('CO2')


###############################################################################
#%% OUTPUT SHORT FORM RESULTS TO CSV (ready for data analysis script)
###############################################################################

headers = ['Sample']
headers.extend(maj_ele_order)

short_output = [headers+['Density']]

mi_list = []
for i in mi_inputdata:
    mi_list.append(i[0])

count = 0
for sample in major_ele_conc:
    temp = [mi_list[count]]
    temp.extend(sample)
    temp.extend([density_values[count]])
    short_output.append(temp)
    count += 1

with open('output\\test_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(short_output)

###############################################################################
#%% CALCULATE ERROR FOR H2O + CO2 MEASUREMENTS (input includes major element error)
###############################################################################

filename = 'mi_inputdata\\test_error.csv'
outfile = 'output\\test_error_output.csv'
#columns must start with sample
#first row must be column headers (labeled 'SIO2', 'AL2O3', etc.)
#column headers must be the same as input data
#should have the exact same number of MI as data file, in the same order

mi_maj_error = []
mi_maj_error_sam = []

#Import data

count = 0
with open (filename, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if count == 0:
            mi_maj_error_headers = row[1:]
        else:
            mi_maj_error.append(float(row[1]))
            mi_maj_error_sam.append(row[0].upper())
        count += 1

# if len(mi_maj_error_sam) != len(mi_list):
#     print 'error, number of samples does not match up with %s' %input_file
# for i in len(mi_list):
#     if mi_maj_error_sam[i].upper != mi_list[i].upper():
#         print 'error, samples out of order or different from %s' %input_file
# if mi_major_error != mi_id_headers:
#     print 'error, column headers are not the same as %s' %input_file
   
   
error = [['Sample', 'H2O1sig', 'CO21sig']]

for sample in range(len(major_ele_conc)):
    avg3550 = avg_peaks_list[sample][0]
    sd3550 = sd_list[sample][0]
    avgdoublet = avg_peaks_list[sample][1]
    sddoublet = sd_list[sample][1]
    h2o_avg = major_ele_conc[sample][maj_ele_order.index('H2O')]
    co2_avg = major_ele_conc[sample][maj_ele_order.index('CO2')]
    thickness_avg = mi_inputdata[sample][mi_id_headers.index('Thickness')]
    err_ind = mi_maj_error_sam.index(mi_inputdata[sample][0])
    thickness_sd = mi_maj_error[sample]
    h2o_error = h2o_avg * sqrt((sd3550/avg3550)**2 + (thickness_sd/thickness_avg)**2)
    co2_error = co2_avg * sqrt((sddoublet/avgdoublet)**2+(thickness_sd/thickness_avg)**2)
    error.append([mi_maj_error_sam[err_ind],h2o_error,co2_error])

with open(outfile, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(error)

###############################################################################
# #%% OUTPUT LONG FORM RESULTS TO CSV (incomplete)
# ###############################################################################

# headers = ['Sample']
# headers.extend(maj_ele_order)

# for i in maj_ele_order:
#     headers.extend(i + ' 1sig')
    
# headers.extend(['H2O (Dixon)', 'H2O (Shishkina)', 'CO2 1430', 'CO2 1515', 'NIR H2O (Mandeville)', 'NIR H2O (Ohlhorst)', 'E CO3', 'E 4500 (Mandeville)', 'E 5200 (Mandeville)', 'E 4500 (Ohlhorst)', 'E 5200 (Ohlhorst)', 'Thickness', 'Density', '1430 CO2 concen', '1515 CO2 concen', '3550 PH', '1430 PH', '1515 PH', '4500 PH', '5200 PH'])

# for i in maj_ele_order:
#     headers.extend(i + ' mol')

# h20 error = h2o average * sqrt ((3500 st dev/3500 avg)**2 + (thickness stdev/avg thick)**2)
# co2 error = co2 average * sqrt((1415 st dev/1415 average)**2+(1515 st dev/1515 average)**2+(thick st dev/avg thick)**2)

###############################################################################