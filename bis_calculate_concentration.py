# Importing the required packages.

import pandas as pd
import numpy as np
import os as os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.optimize as opt
import scipy.signal as sig
import scipy as sp
from uncertainties import ufloat
from uncertainties import unumpy as unp
from IPython.display import display, Math, Latex

SLOPE = ufloat(0.125877358900729, 0.001612237505520) # Slope of calibration curve generated using UV-Vis data of 5 different concentrations.
INTERCEPT = 0.0 # Intercept for the calibration curve is forced to be 0.

# Calibration concentrations (with uncertainties considering 0.5% uncertainty during measurement)

calib_concs = np.array([
    ufloat(0.6643287708558626, 0.003322292433599791),
    ufloat(3.9541374630057073, 0.019774552454058233),
    ufloat(7.243260619415341, 0.03622443328290164),
    ufloat(10.533428582268114, 0.05268050198444311),
    ufloat(13.827306665344352, 0.06915860355700629)
])

# Calibration areas
calib_areas = np.array([5.558796779313221, 30.415349581725117, 57.4909021178828,
                        83.07252924008449, 107.9141618690143])

# Dictionary containing the start and end wavelengths over which the area under curve is to be integrated. The area under curve to be integrated for the input spectrum should be the same as wavelengths over which integration was done to generate the calibration curve.

integration_params = {
    "start_wl": 315.0,
    "end_wl": 450.0,
    "method": "simpson",
    "note": "Bis-MSB sample integration range for concentration estimation of unknown sample"
}


# Data class will encapsulate the binary file info.

import struct, time, os, re, string
class Data:
    # Read the binary file on initialisation.

    def __init__(self, fname, npoints):
        '''
        Read the UVProbe binary file fname.

        # Args:
        
        - fname: the name of the file (if not in current directory should be
          whole path).

        - npoints: Number of points in the UV-VIS spectrum. eg. for a spectrum
          taken between 200 and 700 nm at 0.5 nm intervals, there will be 901
          points. Must be integer.

        '''
        inputsOK = True
        self.binfile = fname
        self.npoints = npoints
        try:
            fh = open(fname, 'rb')
        except FileNotFoundError:
            print('File not found! Bailing...')
            inputsOK = False
       
        if not (type(npoints) is int):
            print('npoints must be integer, but type is {}! Bailing...'.format(type(npoints)))
            inputsOK = False

        if inputsOK:
            self._parsefile(fh)

    def _parsefile(self,fh):

        # The bits to read have been determined empirically, if something is wrong after a version change, or a change of settings, these lines are likely suspects.
        
        if self.npoints == 901:
            # Values for 250 -- 700 nm measurements:
            junk = fh.read(0x26c0)
            hdrblock = fh.read(0x2930 - 0x26c0)# Header info
            junk = fh.read(0x2a00 - 0x2930)
            firstblock = fh.read(0x4628 - 0x2a00) #contains the absorbance
            junk = fh.read(0x4800 - 0x4628)
            secondblock = fh.read(0x6428 - 0x4800)#contains the wavelengths
            fh.close()

        # Values for 300 -- 700 nm measurements:
        else:
            junk = fh.read(0x2680)
            hdrblock = fh.read(0x2950 - 0x2680)# Header info
            junk = fh.read(0x2a00 - 0x2950)
            firstblock = fh.read(self.npoints*8) #contains the absorbance
            junk = fh.read(30*8)
            secondblock = fh.read(self.npoints*8)#contains the wavelengths
            fh.close()

            # Getting data is easy, fortunately.
        
        try:
            self.abs = struct.unpack('<' + 'd'*self.npoints, firstblock)
            self.wl  = struct.unpack('<' + 'd'*self.npoints, secondblock)
        except:
            print('Error: len(firstblock) = {}, len(secondblock) = {}'.format(
                    len(firstblock), len(secondblock)))
            raise

        # Parsing the header is tricky. This may not work in the most general case.

        self.txthdr = hdrblock.decode('cp850')#1252')
        # Hdr info will be stored in a dictionary.
        self.hdrdir = {'Attachment Properties': {'Attachment': None}, 
                       'Instrument Properties': {'Instrument Type': None, 'Measuring Mode': None, 
                                                 'Slit Width': None, 'Accumulation time': None, 
                                                 'Light Source Change Wavelength': None,
                                                 'Detector Unit': None, 'S/R Exchange': None,
                                                 'Stair Correction': None}, 
                       'Sample Preparation Properties': {'Weight': None, 'Volume': None, 
                                                         'Dilution': None, 'Path Length': None,
                                                         'Additional Information': None}, 
                       'Measurement Properties': {'Wavelength Range (nm.)': None, 'Scan Speed': None,
                                                  'Sampling Interval': None, 
                                                  'Auto Sampling Interval': None, 'Scan Mode': None}}
        # Now fill the header info.

        for key, item in self.hdrdir.items():
            #Grab the chunk of string corresponding to the outer level header info
            startidx = self.txthdr.find('['+key+']') + len('[' + key + ']')
            tmphdr2 = self.txthdr[startidx:]
            endidx = tmphdr2.find('[')
            tmphdr3 = tmphdr2[:endidx]
            datainds_start = []
            datainds_end = []
            for innerkey, inneritem in item.items():
                datainds_start += [tmphdr3.find(innerkey)]
                datainds_end += [tmphdr3.find(innerkey) + len(innerkey)]
            datainds_start = np.array(datainds_start)
            datainds_end = np.array(datainds_end)
            for i, (innerkey, inneritem) in enumerate(item.items()):

            #Need to read between the end of this property and the start of the next.
                diffs = datainds_start - datainds_end[i]
                diffs = diffs[diffs>0]
                if len(diffs) == 0:
                    tmphdr4 = tmphdr3[datainds_end[i]:]
                else:
                    tmphdr4 = tmphdr3[datainds_end[i]:(datainds_end[i] +  min(diffs))]
                item[innerkey] = ''.join([j for j in tmphdr4 if j in string.printable 
                                        and not j in string.whitespace])[1:]

# Function which is used for fitting the calibration curve with forced zero intercept. Here it is used to find the residuals.  
def lin_zero(x, m):
    return m * x

# Function used to calculate uncertainty from calibration residuals.

def compute_sigma_res(areas, concentrations, slope, intercept=0.0):
    fitted_vals = lin_zero(areas, slope) # To calculate the concentrations of the original samples used to geenrate calibration curve by interpolating the area under curve to the generated calibration curve to estimate the concentration.
    residuals = concentrations - fitted_vals # actual concentrations
    sigma_res = np.sqrt(np.sum(residuals**2) / (len(areas) - 2)) #calculating the uncertainty due to calibration residuals
    for xi, yi, fi in zip(areas, concentrations, fitted_vals):
        print(f"x = {xi:.5f}, observed y = {yi:.5f}, fitted y = {fi:.5f}, residual = {yi-fi:.5f}")
    print(f"Sum of squared residuals = {np.sum(residuals**2):.5f}")
    return sigma_res

# Function to normalize the input spectrum at the desired wavlength taken as input from user.

def normalize_and_extract(wavelengths, absorbance, norm_wl):
    norm_idx = (np.abs(wavelengths - norm_wl)).argmin() # Normalization wavelength is taken as input from user.
    norm_val = absorbance[norm_idx] 
    return absorbance - norm_val # Subtracting the absorbance of entire spectrum with the absorbance at normalization wavelength.

# Function to integrate the area under the absorbance curve given as input in the form of a .spc file

def compute_area(wavelengths, absorbance, start_wl, end_wl):
    mask = (wavelengths >= start_wl) & (wavelengths <= end_wl) 
    x = wavelengths[mask]
    y = absorbance[mask]

    # Ensure ascending order of wavelengths (else the wavelengths are in descending order and it leads to negative sign during area calculation)
    if x[0] > x[-1]:
        x = x[::-1]
        y = y[::-1]

    # Compute area by simpson method
    area_simpson = sp.integrate.simpson(y, x)
    return area_simpson

# Main function that takes input from user in the order - .spc file containing UV-Vis Data, start and stop wavlengths of the input spectrum plot, interval at which the data is taken, normalization wavelength.
def main():
    if len(sys.argv) != 6:
        print("Usage: python bis_calculate_concentration.py <file.spc> <start_wl> <end_wl> <norm_wl>")
        sys.exit(1)

    spc_file = sys.argv[1]
    spectrum_start = float(sys.argv[2])
    spectrum_stop = float(sys.argv[3])
    interval = float(sys.argv[4])
    norm_wl = float(sys.argv[5])

    # Determine number of points based on start and stop wavelengths and interval

    if spectrum_start == 200 and spectrum_stop == 700 and interval == 0.5:
        npoints = 998
    elif spectrum_start == 250 and spectrum_stop == 700 and interval == 0.5:
        npoints = 901
    elif spectrum_start == 300 and spectrum_stop == 700 and interval == 0.5:
        npoints = 802
    else:
        print("Unsupported wavelength or interval range. Please verify.")
        sys.exit(1)

    # Read the data file

    spec = Data(spc_file, npoints)

    print(f"Data read: {npoints} points")
    print(f"Attributes found: absorbance → {hasattr(spec, 'abs')}, wavelength → {hasattr(spec, 'wl')}")

    absorbance = np.array(spec.abs) # Read absorbance values
    wavelengths = np.array(spec.wl) # Read wavelength values

    # Remove invalid wavelength values

    valid = (wavelengths >= 300) & (wavelengths <= 700)
    wavelengths = wavelengths[valid]
    absorbance = absorbance[valid]

    # Normalize and compute area

    absorbance = normalize_and_extract(wavelengths, absorbance, norm_wl)
    area = compute_area(wavelengths, absorbance, integration_params["start_wl"], integration_params["end_wl"])
    
    # ====== Uncertainty breakdown ======
    m = SLOPE.n
    sigma_m = SLOPE.s

    # Instrument noise (estimate)

    sigma_abs = 0.001  # per point (adjust if known)
    num_points = len(wavelengths[(wavelengths >= integration_params["start_wl"]) &
                                 (wavelengths <= integration_params["end_wl"])])
    sigma_A = sigma_abs * (integration_params["end_wl"] - integration_params["start_wl"]) / num_points

    # Calibration residual scatter

    sigma_res = compute_sigma_res(calib_areas, unp.nominal_values(calib_concs), m, INTERCEPT)

    # Final propagated uncertainty

    A = area
    C_nominal = m * A
    sigma_C = np.sqrt((A*sigma_m)**2 + (m*sigma_A)**2 + sigma_res**2)
    C_with_uncertainty = ufloat(C_nominal, sigma_C)

    # Contribution breakdown

    contrib_slope = (A*sigma_m)**2
    contrib_area  = (m*sigma_A)**2
    contrib_res   = sigma_res**2

    total_sigma = sigma_C
    perc_slope = 100 * contrib_slope / total_sigma
    perc_area  = 100 * contrib_area / total_sigma
    perc_res   = 100 * contrib_res / total_sigma

    # Plot the input data and save it as a spectrum plot

    plt.figure(figsize=(10,8))
    plt.plot(wavelengths, absorbance, label='Normalized Spectrum')
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorbance (normalized)")
    plt.title(f"Spectrum: {spc_file}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("spectrum_plot.png")

    # Print results

    print(f"Area under curve (from {integration_params['start_wl']} to {integration_params['end_wl']} nm) = {area:.5f}")
    print(f"Estimated Concentration = {C_with_uncertainty:.5uP} mg/L")
    print(f"Uncertainty Breakdown (variance contributions):")
    print(f"  From slope: {contrib_slope:.5f} mg/L ({perc_slope:.1f}%)")
    print(f"  From area integration: {contrib_area:.5f} mg/L ({perc_area:.1f}%)")
    print(f"  From calibration residuals: {contrib_res:.5f} mg/L ({perc_res:.1f}%)")
    print(f"Total combined uncertainty: {total_sigma:.5f} mg/L")
    print("\nPlot saved as spectrum_plot.png")

if __name__ == "__main__":
    main()
