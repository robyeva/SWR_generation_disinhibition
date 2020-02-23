# SWR_generation_disinhibition
Code to reproduce figures of the manuscript 'Generation of sharp wave-ripple events by disinhibition' by Evangelista, Cano, Cooper, Maier, Schmitz, and Kempter (2020).

## Content

The code is organized in three folders: 
 - `spiking_model`: code to generate the spiking model and associated Figures (2, 6-10, 2-1, 6-1)
 - `rate_model`: code to derive a rate model from a spiking network and create Figures 2-2 and 6-2
 - `bifurcation_analysis`: code to perform bifurcation analysis of the rate model and create Figures 3-5
 
 ### `Spiking_model`: 
 - `run_spiking.py`: generates spiking network and all plots
 - `utils_spiking.py`: supporting functions to simulate, analyze, and plot results
 - `detect_peaks.py`: supporting function to detect peaks in spiking simulations [1]
 - `construct_spiking_network.py`: functions to create the 3d spiking network in steps 
 - `simulate_spiking.py`: run simulations of the spiking network and save results for plotting
 - `figures_spiking.py`: generates Figures 2, 6-10, 2-1, 6-1 using saved simulations
 
 ### `Rate_model`:
 - `run_rate.py`: finds parameters to define the rate model and creates all plots
 - `utils_rate.py`: supporting functions to simulate, analyze and plot results 
 - `simulate_and_plot_rate.py`: stored optimization results to find rate model parameters, simulate rate network and create Figures 2-2 and 6-2
 
 ### `Bifurcation_analysis`: 
 - coming soon
 
 ## How to run
 
 - Run `python spiking_model/run_spiking.py` to generate spiking network and all plots of spiking network
 - Run `python rate_model/run_rate.py` to generate rate network and all related plots 
 
 Code has been tested using python=3.6.10, brian2=2.3, matplotlib=3.1.3, numpy=1.18.1, and scipy=1.4.1.
 
 
 
 [1]: Marcos Duarte, https://github.com/demotu/BMC
 
