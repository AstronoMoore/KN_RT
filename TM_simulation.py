#!/usr/bin/env python
# MIT License
#
# Copyright (c) 2018 Ulrich Noebauer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Python module containing tools to perform simple MCRT simulations for the
line profile test in a homologously expanding spherical flow. This test is
presented in the MCRT review.
There is the possibility to compare the MCRT results to analytic predictions
obtained from a formal integration of the radiative transfer problem, following
the procedure outlined by Jeffery & Branch 1990. For this, a external module
has to be imported which can be obtained from the github repository
https://github.com/unoebauer/public-astro-tools.git
References
----------
Jeffery, D. J. & Branch,  Analysis of Supernova Spectra in
    Analysis of Supernova Spectra Supernovae,
    Jerusalem Winter School for Theoretical Physics, 1990, 149
Abbreviations used:
    CMF: co-moving frame
    LF: lab frame
    MC: Monte Carlo

Varaibles that need updated for each run:
    line 74 = t_default = time since merger
    line 530 = Blackbody temperature
    line 556 = 'Day' flag (time since merger in days: used in JitterAndHistograms.py)
"""
from __future__ import print_function
import os
import sys
import numpy as np
from astropy import units, constants
from astropy.modeling.models import BlackBody
import matplotlib
import pandas as pd
from itertools import product
from tqdm import tqdm


if "DISPLAY" not in os.environ:
    # backend that works without an X-server
    matplotlib.use("agg")
import matplotlib.pyplot as plt
try:
    # Available from https://github.com/unoebauer/public-astro-tools.git
    import pcygni_profile as pcyg
    analytic_prediction_available = True
except ImportError:
    analytic_prediction_available = False
    pass


# Set RNG seed for reproducibility
np.random.seed(42)

# Parameters used for test calculation shown in the review
''' gamma_default must be a negative number as optical depth decreases with 
    distance from the centre of the event
    tau_sobolev_default = 0.5 and gamma_default = -2
    found to be the best fit values for the data'''

tau_sobolev_default = 1.0
gamma_default = -3
t_default = 5.000 * units.d
lam_min_default = 2000 * units.AA
lam_max_default = 18000 * units.AA
lam_line_default = 10500 * units.AA
vmin_default = 0.1 * constants.c
vmax_default = 0.35 * constants.c
Rmin_default = vmin_default * t_default
Rmax_default = vmax_default * t_default


class PropagationError(Exception):
    pass


class mc_packet(object):
    """Monte Carlo packet class
    Class describing a Monte Carlo packet propagating in a spherical homologous
    flow, bounded by the radii Rmin and Rmax. With the class methods, the
    propagation of the packet, including resonant line interactions within the
    Sobolev approximation can be performed.
    Parameters
    ----------
    Rmin : float
        inner radius of the spherical homologous flow;
        must be in cm (default 3.5e12)
    Rmax : float
        outer radius of the spherical homologous flow;
        must be in cm (default 3.5e14)
    nu_min : units.quantity.Quantity object
        minimum fequency of the spectral range considered, must be in Hz
        (default 2.4e15)
    nu_max : units.quantity.Quantity object
        maximum frequency of the spectral range considered, must be in Hz
        (default 2.5e15)
    lam_line : units.quantity.Quantity object
        rest frequency of the line transition, must be in Hz
        (default 2.47e15)
    tau_sobolev : float
        Sobolev optical depth of the line transition; assumed to constant
        throughout the domain, must be dimensionless
        (default 1)
    t : units.quantity.Quantity object
        time since explosion, must be in s (default 1.1e6)
    verbose : boolean
        flag controlling the output to stdout (default False)
    """
    # gamma = correction factor in power law for optical depth calculation

    def __init__(self, Rmin=Rmin_default.value, Rmax=Rmax_default.value,
                 nu_min=constants.c / lam_max_default, nu_max=constants.c / lam_min_default,
                 nu_line=constants.c / lam_line_default, tau_sobolev=tau_sobolev_default, gamma=gamma_default,
                 t=t_default.to("s").value, verbose=False, temperature=3000.0):

        self.verbose = verbose

        self.gamma = gamma
        self.nu_min = nu_min
        self.nu_max = nu_max
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.temperature = temperature

        self.nu_line = nu_line
        self.tau_sob = tau_sobolev

        # consistency check
        assert (self.Rmin < self.Rmax)
        assert (self.nu_max > self.nu_min)

        # initializing the packets at the inner boundary; no limb darkening,
        # flat SED between nu_min and nu_max
        # clock tracks the time from packets being released to escaping,
        # t = 0 when packets are initialised at inner boundary
        # self.l_track tracks the total distance travelled by the packet before escaping,
        # minimum value = Rmax as a packet that undergoes no collisions travels,
        #                 a distance = radius of the kilonova
        self.r = self.Rmin
        self.mu = np.sqrt(np.random.rand(1)[0])
        self.l_track = -99e30  # self.Rmax - self.Rmin
        self.nu = self.nu_min + \
            (self.nu_max - self.nu_min) * np.random.rand(1)[0]
        self.t = t
        self.emergent_weight = 0.0
        self.clock = None  # (self.Rmax - self.Rmin) / constants.c.cgs.value

        # LF frequency of packet when it emerges from the surface of the
        # homologous sphere
        self.emergent_nu = None
        # Time difference when packets emerge from sphere
        self.relative_time = 0
        # Total time taken for packets to travel to observer
        # = time taken for packets to escape + relative time delay
        # (self.Rmax - self.Rmin) / constants.c.cgs.value
        self.total_time = None
        # distance to next interaction in optical depth space
        self.tau_int = None
        # distance to nearest boundary
        self.lbound = None
        # flag describing which boundary is intersected first on current path
        # either 'inner' or 'outer'
        self.boundint = None
        # flag describing the ultimate fate of the packet, whether it escaped
        # or was absorbed
        self.fate = None
        # flag describing whether packet has been propagated or not
        self.propagated = False
        # flag describing whether packet interacts and is scattered or not
        self.scattered = False

        self.draw_new_tau()
        self.check_for_boundary_intersection()
        self.calc_distance_to_sobolev_point()

    def draw_new_tau(self):
        """Draw new distance to next interaction based on Beer-Lambert law"""

        self.tau_int = -np.log(np.random.rand(1)[0])

    def update_position_direction(self, l):
        """Update the packet state during propagation
        Calculate the new radial position and propagation direction after
        having covered the distance l along the current trajectory.
        Parameters
        ----------
        l : units.quantity.Quantity object
            distance the packet travelled along the current trajectory, must
            dimension of length
        """

        ri = self.r
        self.r = np.sqrt(self.r**2 + l**2 + 2 * l * self.r * self.mu)
        self.mu = ((l + self.mu * ri) / self.r)

    def check_for_boundary_intersection(self):
        """Check which boundary of the spherical domain is intersected first
        Checks whether the inner or the outer boundary of the spherical domain
        is intersected first on the current trajectory. Sets the flag
        self.boundint' accordingly and calculates the physical distance to the
        nearest boundary and stores it in self.lbound.71.5	2573	2.979

        """

        if self.mu <= -np.sqrt(1 - (self.Rmin / self.r)**2):
            # packet will intersect inner boundary if not interrupted
            sgn = -1.
            rbound = self.Rmin
            self.boundint = "inner"
        else:
            # packet will intersect outer boundary if not interrupted
            sgn = 1.
            rbound = self.Rmax
            self.boundint = "outer"

        self.lbound = (
            -self.mu * self.r + sgn * np.sqrt((self.mu * self.r)**2 -
                                              self.r**2 + rbound**2))

    def perform_interaction(self):
        """Performs line interaction
        Updates the LF frequency of the packet according to the first order
        Doppler shift formula and assuming resonant scattering. A new
        propagation LF direction is drawn assuming isotropy in the CMF.
        """

        beta = self.r / self.t / constants.c.cgs.value

        self.mu = 2. * np.random.rand(1)[0] - 1.
        self.mu = (self.mu + beta) / (1 + beta * self.mu)

        self.nu = self.nu_line / (1. - beta * self.mu)

    def calc_distance_to_sobolev_point(self):
        """Calculated physical distance to Sobolev point"""

        self.lsob = (constants.c.cgs.value * self.t *
                     (1 - self.nu_line / self.nu) -
                     self.r * self.mu)

    def print_info(self, message):
        if self.verbose:
            print(message)

    # adding blackbody continum

    # function to calculate blackbody flux with output per angstrom
    # input wavelength should be in units of m, input temperature in units of K
    # inputs must be dimensionless to avoid issues with astropy.units and np.exp
    # astropy's inbuilt bb func NOT used to avoid issues arising
    # in working with wavelength vs frequency

    def bb_lam(self, wavelength, temperature):

        h = 6.62607015e-34
        c = 299792458
        k_B = 1.380649e-23
        a = 2.0 * h * (c ** 2)
        b = (h * c) / (wavelength * k_B * temperature)
        flux = a / ((pow(wavelength, 5)) * (np.exp(b) - 1.0))

        return (flux)

    def bb_nu(self, freq, temperature):

        h = 6.62607015e-34
        c = 299792458
        k_B = 1.380649e-23
        a = 2.0 * h / (c ** 2)
        b = (h * freq) / (k_B * temperature)
        flux = a * pow(freq, 2) / (np.exp(b) - 1.0)

        return (flux)

    def propagate(self):
        """Perform packet propagation
        The packet is propagated through the spherical domain until it either
        escapes through the outer boundary and contributes to the spectrum or
        until it intersects the inner boundary and is discarded.  The
        implementation of the propagation routine is specific to the problem at
        hand and makes use of the fact that a packet can at most interact once.
        """

        """intialise the weight"""
        beta = self.r / self.t / constants.c.cgs.value
        nu_cmf = self.nu*(1. - beta * self.mu)
#        lam_cmf = constants.c.cgs.value/self.nu
        self.emergent_weight = self.bb_nu(
            nu_cmf, self.temperature)*self.r*self.r/nu_cmf

        if self.propagated:
            raise PropagationError(
                "Packet has already been propagated!"
            )

        if self.lbound < self.lsob or self.lsob < 0:
            # self.print_info("Reaching outer boundary")
            self.fate = "escaped"

            '''calculate the relative extra distance a packet has to travel
            given the trajectory it escapes with'''

            self.l_track = self.lbound
            self.update_position_direction(self.lbound)
#            self.relative_distance = self.Rmax - (self.Rmax * (self.mu))
            self.relative_distance = -1.*(self.Rmax * (self.mu))
            self.relative_time = self.relative_distance / constants.c.cgs.value

        else:
            # self.print_info("Reaching Sobolev point")
            self.update_position_direction(self.lsob)

            '''Apply correction factor so optical depth is not considered constant
            and instead changes with radius
            Gamma is the exponential term in a power law relationship'''

#            self.tau_corrected = self.tau_sob * ((self.r / (self.Rmax - self.Rmin)) ** self.gamma)
            self.tau_corrected = self.tau_sob * \
                ((self.r / self.Rmin) ** self.gamma)

            if self.tau_corrected >= self.tau_int:
                # self.print_info("Line Interaction")
                self.perform_interaction()
                self.check_for_boundary_intersection()
                self.scattered = True
                if self.boundint == "inner":
                    # self.print_info("Intersecting inner boundary")
                    self.fate = "absorbed"
                else:
                    # self.print_info("Reaching outer boundary")
                    self.fate = "escaped"

                    '''calculate the relative extra distance a packet has to travel
                    given the trajectory it escapes with'''
                    self.l_track = self.lsob + self.lbound
                    self.update_position_direction(self.lbound)
#                    self.relative_distance = self.Rmax - (self.Rmax * (self.mu))
                    self.relative_distance = -1.*(self.Rmax * (self.mu))
                    self.relative_time = self.relative_distance / constants.c.cgs.value
            else:
                self.fate = "escaped"

                '''calculate the relative extra distance a packet has to travel
                given the trajectory it escapes with'''

                self.l_track = self.lbound
# CHECK
                self.update_position_direction(self.lbound - self.lsob)
#                self.relative_distance = self.Rmax - (self.Rmax * (self.mu))
                self.relative_distance = -1.*(self.Rmax * (self.mu))
                self.relative_time = self.relative_distance / constants.c.cgs.value

        self.emergent_mu = self.mu
        self.emergent_nu = self.nu
        self.emergent_weight = self.emergent_weight*self.nu
        self.l_store = self.l_track
        self.emergent_clock = self.l_store / constants.c.cgs.value
        self.time_delay = self.relative_time
        self.propagated = True
        self.scattered = self.scattered


class homologous_sphere(object):
    """
    Class describing the sphere in homologous expansion in which the MCRT
    simulation is performed
    The specified number of MC packets are initialized. Their propagation is
    followed in the main routine of this class. As a result, the emergent
    frequencies of all escaping packets are recorded in self.emergent_nu.
    Parameters
    ----------
    Rmin : units.quantity.Quantity object
        inner radius of the spherical homologous flow;
        must have a length dimension (default vmin_default * t_default)
    Rmax : units.quantity.Quantity object
        outer radius of the spherical homologous flow;
        must have a length dimension (default vmax_default * t_default)
    lam_min : units.quantity.Quantity object
        minimum wavelength of the spectral range considered, must have
        a length dimension (default lam_min_default)
    lam_max : units.quantity.Quantity object
        maximum wavelength of the spectral range considered, must have
        a length dimension (default lam_max_default)
    lam_line : units.quantity.Quantity object
        rest wavelength of the line transition, must have a length
        dimension (default lam_line_default)
    tau_sobolev : float
        Sobolev optical depth of the line transition; assumed to constant
        throughout the domain (default tau_sobolev_default)
    t : units.quantity.Quantity object
        time since explosion, must have dimension of time (default t_default)
    verbose : boolean
        flag controlling the output to stdout (default False)
    npacks : int
        number of packets in the MCRT simulation (default 10000)
    """

    def __init__(self, Rmin=Rmin_default, Rmax=Rmax_default,
                 lam_min=lam_min_default, lam_max=lam_max_default,
                 lam_line=lam_line_default, tau_sobolev=tau_sobolev_default,
                 t=t_default, verbose=False, npacks=10000, temperature=3000.0):

        t = t.to("s").value
        Rmin = Rmin.to("cm").value
        Rmax = Rmax.to("cm").value

        nu_min = lam_max.to("Hz", equivalencies=units.spectral()).value
        nu_max = lam_min.to("Hz", equivalencies=units.spectral()).value
        nu_line = lam_line.to("Hz", equivalencies=units.spectral()).value

        self.npacks = npacks
        self.temperature = temperature
        self.packets = [mc_packet(Rmin=Rmin, Rmax=Rmax, nu_min=nu_min,
                                  nu_max=nu_max, nu_line=nu_line,
                                  tau_sobolev=tau_sobolev_default, t=t,
                                  verbose=verbose, temperature=temperature) for i in range(npacks)]

        self.emergent_mu = []
        self.emergent_nu = []
        self.emergent_weight = []
        self.time_delay = []
        self.emergent_clock = []
        self.l_store = []
        self.scattered_check = []

    def perform_simulation(self):
        """Perform MCRT simulation in the homologous flow
        All packets are propagated until they either escape from the sphere or
        intersect the photosphere and are discarded.
        """

        for i, pack in enumerate(self.packets):
            pack.propagate()
            if pack.fate == "escaped":
                self.emergent_mu.append(pack.emergent_mu)
                self.emergent_nu.append(pack.emergent_nu)
                self.emergent_weight.append(pack.emergent_weight)
                self.emergent_clock.append(pack.emergent_clock)
                self.time_delay.append(pack.time_delay)
                self.l_store.append(pack.l_store)
                self.scattered_check.append(pack.scattered)
            # if (i % 10000) == 0:
                # print("{:d} of {:d} packets done".format(i, self.npacks))

        self.emergent_nu = (np.array(self.emergent_nu) * units.Hz)
        self.emergent_clock = (np.array(self.emergent_clock) * units.second)
        self.time_delay = (np.array(self.time_delay) * units.second)
        self.l_store = (np.array(self.l_store) * units.cm)
        self.emergent_nu_list = self.emergent_nu.value.tolist()
        self.emergent_clock_list = self.emergent_clock.value.tolist()
        self.time_delay_list = self.time_delay.value.tolist()
        self.emergent_mu_list = self.emergent_mu
        self.scattered_check_list = self.scattered_check


def perform_line_profile_calculation(temp_use, Rmin=Rmin_default, Rmax=Rmax_default,
                                     lam_min=lam_min_default,
                                     lam_max=lam_max_default,
                                     lam_line=lam_line_default,
                                     tau_sobolev=tau_sobolev_default,
                                     t=t_default, verbose=False, npacks=10000,
                                     nbins=100, npoints=500, save_to_pdf=True,
                                     include_analytic_solution=False):
    """
    Class describing the sphere in homologous expansion in which the MCRT
    simulation is performed
    The specified number of MC packets are initialized. Their propagation is
    followed in the main routine of this class. As a result, the emergent
    frequencies of all escaping packets are recorded in self.emergent_nu.
    Parameters
    ----------
    Rmin : units.quantity.Quantity object
        inner radius of the spherical homologous flow;
        must have a length dimension (default vmin_default * t_default)
    Rmax : units.quantity.Quantity object
        outer radius of the spherical homologous flow;
        must have a length dimension (default vmax_default * t_default)
    lam_min : units.quantity.Quantity object
        minimum wavelength of the spectral range considered, must have
        a length dimension (default lam_min_default)
    lam_max : units.quantity.Quantity object
        maximum wavelength of the spectral range considered, must have
        a length dimension (default lam_max_default)
    lam_line : units.quantity.Quantity object
        rest wavelength of the line transition, must have a length
        dimension (default lam_line_default)
    tau_sobolev : float
        Sobolev optical depth of the line transition; assumed to constant
        throughout the domain (default tau_sobolev_default)
    t : units.quantity.Quantity object
        time since explosion, must have dimension of time (default t_default)
    verbose : boolean
        flag controlling the output to stdout (default False)
    npacks : int
        number of packets in the MCRT simulation (default 10000)
    nbins : int
        number of bins used for the histogram when plotting the emergent
        spectrum (default 100)
    npoints : int
        number of points used in the formal integration when calculating the
        analytic solution, provided that the module is available and that
        include_analytic_solution is set to True (default 500)
    save_to_pdf : bool
        flag controlling whether the comparison plot is saved to pdf (default
        True)
    include_analytic_solution : bool
        flag controlling whether the analytic solution is included in the plot;
        this requires that the appropriate module is available (default True)
    """

    vmin = (Rmin / t).to("cm/s")
    vmax = (Rmax / t).to("cm/s")

    nu_min = lam_max.to("Hz", equivalencies=units.spectral())
    nu_max = lam_min.to("Hz", equivalencies=units.spectral())

    npoints = 500

    # print("Using Rmin: ", Rmin, " Rmax: ", Rmax, " t: ", t)
    # print("Using Tbb: ", temp_use)

    sphere = homologous_sphere(
        Rmin=Rmin, Rmax=Rmax, lam_min=lam_min, lam_max=lam_max,
        lam_line=lam_line, tau_sobolev=tau_sobolev_default, t=t, npacks=npacks,
        verbose=verbose, temperature=temp_use)
    sphere.perform_simulation()

    fig = plt.figure(1)
    ax = fig.add_subplot(111)

    if include_analytic_solution:
        if analytic_prediction_available:
            # WARNING: untested
            ve = 1e40 * units.cm / units.s
            vref = 1e8 * units.cm / units.s
            solver = pcyg.PcygniCalculator(t=t, vmax=vmax, vphot=vmin,
                                           tauref=tau_sobolev, vref=vref,
                                           ve=ve, lam0=lam_line)
            nu_tmp, Fnu_normed_tmp = solver.calc_profile_Fnu(npoints=npoints)
            Fnu_normed = np.append(np.insert(Fnu_normed_tmp, 0, 1), 1)

            # numpy append has difficulties with astropy quantities
            nu = np.zeros(len(nu_tmp) + 2) * nu_tmp.unit
            nu[1:-1] = nu_tmp[::]
            nu[0] = nu_min
            nu[-1] = nu_max

            ax.plot(nu.to("1e15 Hz"), Fnu_normed,
                    label=r"formal integration")
        # else:
            # print("Warning: module for analytic solution not available")

    total_time = sphere.emergent_clock + sphere.time_delay
    total_time = np.array(total_time)
    total_time_list = total_time.tolist()
    emergent_wavelength_convert = constants.c.value / sphere.emergent_nu.value
    emergent_wavelength_list = emergent_wavelength_convert.tolist()
    # bb_wavelength_list = sphere.emergent_weight.tolist()

    # create pandas dataframe
    record = {
        'Arrival time': total_time_list,
        'Frequency': sphere.emergent_nu_list,
        'Wavelength': emergent_wavelength_list,
        'BB Flux':  sphere.emergent_weight,
        'Time inside kilonova': sphere.emergent_clock,
        'Relative time to observer': sphere.time_delay_list,
        'Direction cosine': sphere.emergent_mu_list,
        'Scattered': sphere.scattered_check_list,
        'Day': t.to(units.d).value
    }
    return record
    # dataframe = pd.DataFrame(record, columns=['Arrival time', 'Frequency', 'Wavelength', 'BB Flux',
    #                                          'Time inside kilonova', 'Relative time to observer', 'Direction cosine', 'Scattered', 'Day'])
    # print(dataframe)
    # fname=sys.argv[7]+'/' +sys.argv[6]+'Day.csv'
    # dataframe.to_csv(fname, index = False)

# section used to plot single histogram from individual run of the simulation


def example(temp_use, t_use):
    """Perform the MCRT test simulation from the review"""

    """Use Gillanders fit for inner boundary"""
    inner_v = 4.33246154e-01 - 1.17547436e-01*t_use.value + 9.93589744e-05*t_use.value*t_use.value+4.83974359e-03 * \
        t_use.value*t_use.value*t_use.value-5.60897436e-04 * \
        t_use.value*t_use.value*t_use.value*t_use.value
    if (inner_v < 0.05):
        inner_v = 0.05
    if (inner_v > 0.20):
        inner_v = 0.2

    # print(temp_use, inner_v, t_use)

    record = perform_line_profile_calculation(temp_use,
                                              Rmin=Rmin_default*t_use/t_default*inner_v/0.1, Rmax=Rmax_default*t_use/t_default, lam_min=lam_min_default,
                                              lam_max=lam_max_default, lam_line=lam_line_default,
                                              tau_sobolev=tau_sobolev_default, t=t_use, verbose=False,
                                              npacks=100000, nbins=100, npoints=500, save_to_pdf=True)
    return record


def is_decreasing(combination):
    for i in range(1, len(combination)):
        if combination[i] > combination[i - 1]:
            return False
    return True


def main():
    """Main routine; performs the example calculation"""

    # for i in range(1, len(sys.argv)):
    # print('argument:', i, 'value:', sys.argv[i])

    # t_fit = float(sys.argv[6])
    # print(t_fit)

    """get fit coefficients - input are temps at 0.5,1.4, 2.4, 3.4 and 4.4"""
    fitting_times = [0.5, 1.4, 2.4, 3.4, 4.4]

    array1 = np.arange(6300, 6300 + 4 * 250, 250)
    array2 = np.arange(4700, 4700 + 4 * 250, 250)
    array3 = np.arange(4200, 4200 + 4 * 250, 250)
    array4 = np.arange(3500, 3500 + 4 * 250, 250)

    all_combinations = product(*[array1, array2, array3, array4])

    decreasing_combinations = [
        comb for comb in all_combinations if is_decreasing(comb)]
    decreasing_combinations = []
    for comb in product(*[array1, array2, array3, array4]):
        if is_decreasing(comb):
            decreasing_combinations.append(comb)

    times = np.arange(0.5, 8, 0.1)

    for comb in tqdm(decreasing_combinations, desc=" outer", position=0):
        fitting_temps = [10000, comb[0], comb[1], comb[2], comb[3]]
        output_df = pd.DataFrame(columns=['Arrival time', 'Frequency', 'Wavelength', 'BB Flux',
                                 'Time inside kilonova', 'Relative time to observer', 'Direction cosine', 'Scattered', 'Day'])
        coeffs = np.polyfit(fitting_times, fitting_temps, 4)

        # if (t_fit > 4.4): # what is this?
        #    t_fit = sys.argv[5]

        for time in tqdm(times, desc=" inner loop", position=1, leave=False):
            temp_use = coeffs[4] + coeffs[3]*time + coeffs[2]*time * \
                time + coeffs[1]*time*time*time+coeffs[0]*time*time*time*time

            if (temp_use < 2200):
                temp_use = 2200
            record = example(temp_use, time*units.d)
            # output_df.append(record)
            output_df.loc[len(output_df)] = record

        output_df.to_csv('single_temp_conbination.csv')


if __name__ == "__main__":

    main()
    # plt.show()
