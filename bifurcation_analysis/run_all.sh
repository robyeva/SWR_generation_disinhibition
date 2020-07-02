# Script to generate all bifurcation analysis figures

# Convert .svg rate model diagram to .png
echo "Converting svg diagram to png ..."
pushd figures_code
./svg_to_png.sh

# Create Rate Overview Figure
echo "Creating fig_rate_overview.eps ..."
python fig_rate_overview.py

# Create 1D Bifurcations Figure
echo "Creating fig_bifurcations_1d.eps ..."
python fig_bifurcations_1d.py

# Create 2D Bifurcations Figure
echo "Creating fig_bifurcations_2d.eps ..."
python fig_bifurcations_2d.py

popd
