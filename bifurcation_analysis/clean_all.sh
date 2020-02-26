# Cleans all automatically generated files

# Delete pseudo nullclines
rm -r figures_code/pseudo_nullclines/*.npy

# Delete png diagram
rm -r figures_code/*.png

# Delete python cache
rm -rf figures_code/__pycache__ figures_code/*.pyc figures_code/helper_functions/__pycache__

# Delete figure outputs
rm -r figures_output/*.eps
