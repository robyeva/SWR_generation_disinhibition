# SWR_generation_disinhibition
Code to reproduce figures of the manuscript 'Generation of sharp wave-ripple events by disinhibition' by Evangelista, Cano, Cooper, Maier, Schmitz, and Kempter (2020), JNeuroscience, https://doi.org/10.1523/JNEUROSCI.2174-19.2020.

## Content

The code is organized in two folders:
 - `network_simulations`: code to generate the spiking model and associated Figures (2-4, 9, 11, 13-15); to derive a rate model from a spiking network and create Figures 5 and 10; and to simulate the rate model with noise, create Figure 12, and create data for Figures 13E and 15D.
 - `bifurcation_analysis`: code to perform bifurcation analysis of the rate model, create Figures 6-8, and provide bifurcation diagrams for Figures 13E and 15D.

### `Network_simulations`:
 - `run_spiking.py`: constructs the spiking network, runs network simulations and creates and Figures 2-4, 9, 11, 13-15.
 - `run_rate.py`: finds parameters to define the rate model and creates Figures 5 and 10.
 - `run_noisy_rate.py`: runs noisy rate model simulations, creates Figure 12, and creates data for Figures 13E and 15D.
 - `/helper_functions` folder: contains supporting functions:
    - `utils_spiking.py`: supporting functions to simulate, analyze, and plot results
    - `detect_peaks.py`: supporting function to detect peaks in spiking and noisy rate model simulations [1]
    - `construct_spiking_network.py`: functions to create the 3d spiking network in steps
    - `simulate_spiking.py`: functions to simulate the spiking network and save results for plotting
    - `figures_spiking.py`: functions to generate Figures 2-4, 9, 11, 13-15 using saved simulations
    - `utils_rate.py`: supporting functions to simulate, analyze and plot results of the rate model
    - `simulate_and_plot_rate.py`: functions to store optimization results to find rate model parameters, simulate rate network and create Figures 5 and 10
    - `params_noisy_rate.py`: parameters for the noisy rate model simulations
    - `utils_noisy_rate.py`: supporting functions to analyze results of the noisy rate model and plot Figure 12
    - `simulate_noisy_rate.py`: functions to perform and store results of noisy rate model simulations
 - `/results` folder: default destination for all simulations' .npz files and figures

### `Bifurcation_analysis`:
 - `run_all.sh` automatically generates all figures

 - `clean_all.sh` cleans all auto-generated files

 - `bifurcation_diagrams/` **XPPAUT** files

   - `rate_model.ode` **XPPAUT** input file. Has to be tweaked in order to replicate each bifurcation diagram.

   - `1param/` **XPPAUT** outputs: 1D bifurcation diagram *.dat* files

     - `auto_xxx_line.dat` and `auto_xxx_fold.dat` each bifurcation diagram has been saved in two separate components: the line branch and the fold branch.

   - `2param` XPPAUT outputs: 2D bifurcation diagram *.dat* files

 - `figures_code/` code to generate figures

   - `rate_model.svg` rate model *svg* diagram with connection strengths

   - `svg_to_png.sh` script to convert *svg* diagram to *png* using **Inkscape** (must be run before *fig_rate_overview.py*)

   - `helper_functions/`: `model.py`, `params.py`, `aux.py`, `bifurcations.py`, and `nullclines.py` auxiliary python code used across different figures

   - `fig_rate_overview.py` creates *fig_rate_overview.eps* (Fig 6)

   - `fig_bifurcations_1d.py` creates *fig_bifurcations_1d.eps* (Fig 7)

   - `fig_bifurcations_2d.py` creates *fig_bifurcations_2d.eps* (Fig 8)

   - `pseudo_nullclines/` folder where the auto-generated pseudo-nullcline *.npy* files, created by *fig_rate_overview.py*, are saved.

 - `figures_output/` folder where generated *.eps* figures are saved.

## How to run

### `Spiking_model`:
- Run `python run_spiking.py` to generate spiking network and all plots of spiking network. *Note*: to generate Figures 13 and 15, `run_noisy_rate.py` must be run beforehand.

### `Rate_model`:
- Run `python run_rate.py` to generate rate network and all related plots.  *Note*: the rate model is an approximation of the spiking model. Thus, `run_rate.py` should be run after the creation of the spiking model (all scripts in `construct_spiking_model.py` and f-I curves calculation).

### `Noisy_rate_model`:
- Run `python run_noisy_rate.py` to simulate noisy rate model, export data used in Figures 13E and 15D, and generate Figure 12. *Note*: unlike the simulations in `run_rate.py` (where the rate model is derived from the randomly generated spiking network), the noisy rate model uses fixed parameters (in manuscript) and can be run before the creation of spiking network.

### `Bifurcation_analysis`:
All figures are saved in `figures_output/`. The script `run_all.sh` automatically generates all figures described above. The script `clean_all.sh` cleans all auto-generated files. To generate individual figures run the following python scripts in `figures_code/`:

- `fig_rate_overview.py` generates *fig_rate_overview.eps* (Fig 6)

  **Note**: to run this script it is first necessary to convert the *rate_model* diagram from *svg* to *png*. If you have **Inkscape** you can do this automatically by running the script `svg_to_png.sh`.

- `fig_bifurcations_1d.py` generates *fig_bifurcations_1d.eps* (Fig 7)

- `fig_bifurcations_2d.py` generates *fig_bifurcations_2d.eps* (Fig 8)


Code has been tested using python=3.6.10, brian2=2.3, matplotlib=3.1.3, numpy=1.18.1, and scipy=1.4.1.

## License
This repository is licensed under the
GNU General Public License v3.0 (see `LICENSE.txt` for details)


 [1]: Marcos Duarte, https://github.com/demotu/BMC
