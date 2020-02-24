# SWR_generation_disinhibition
Code to reproduce figures of the manuscript 'Generation of sharp wave-ripple events by disinhibition' by Evangelista, Cano, Cooper, Maier, Schmitz, and Kempter (2020).

## Content

The code is organized in three folders:
 - `spiking_model`: code to generate the spiking model and associated Figures (2, 6-10, 2-1, 6-1)
 - `rate_model`: code to derive a rate model from a spiking network and create Figures 2-2 and 6-2
 - `bifurcation_analysis`: code to perform bifurcation analysis of the rate model and create Figures 3-5, 8-1, and 10-1

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

   - `model.py`, `params.py`, `aux.py`, `bifurcations.py`, and `nullclines.py` auxiliary python code used across different figures

   - `fig_rate_overview.py` creates *fig_rate_overview.eps* (Fig 3)

   - `fig_bifurcations_1d.py` creates *fig_bifurcations_1d.eps* (Fig 4)

   - `fig_bifurcations_2d.py` creates *fig_bifurcations_2d.eps* (Fig 5)

   - `fig_extra_synapses.py` creates *fig_bif_extra_depr.eps* (Fig 8-1) and *fig_bif_extra_facil.eps* (Fig 10-1)

   - `pseudo_nullclines/` folder where the auto-generated pseudo-nullcline *.npy* files, created by *fig_rate_overview.py*, are saved.

 - `figures_output/` folder where generated *.eps* figures are saved.

## How to run

### `Spiking_model`:
- Run `python spiking_model/run_spiking.py` to generate spiking network and all plots of spiking network

### `Rate_model`:
- Run `python rate_model/run_rate.py` to generate rate network and all related plots

### `Bifurcation_analysis`:
All figures are saved in `figures_output/`. The script `run_all.sh` automatically generates all figures described above. The script `clean_all.sh` cleans all auto-generated files. To generate individual figures run the following python scripts in `figures_code/`:

- `fig_rate_overview.py` generates *fig_rate_overview.eps* (Fig 3)

  **Note**: to run this script it is first necessary to convert the *rate_model* diagram from *svg* to *png*. If you have **Inkscape** you can do this automatically by running the script `svg_to_png.sh`.

- `fig_bifurcations_1d.py` generates *fig_bifurcations_1d.eps* (Fig 4)

- `fig_bifurcations_2d.py` generates *fig_bifurcations_2d.eps* (Fig 5)

- `fig_extra_synapses.py` generates

  - *fig_bif_extra_depr.eps* (Fig 8-1)

  - *fig_bif_extra_facil.eps* (Fig 10-1)



Code has been tested using python=3.6.10, brian2=2.3, matplotlib=3.1.3, numpy=1.18.1, and scipy=1.4.1.

## License
This repository is licensed under the
GNU General Public License v3.0 (see `LICENSE.txt` for details)


 [1]: Marcos Duarte, https://github.com/demotu/BMC
