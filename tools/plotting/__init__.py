from .stat_plot import line_plot
from .field_plot import (profile_plot,
                         slice_plot)
import matplotlib.pyplot as plt
from .plotting import (create_image_grid,
                       remove_axis_ticks)
import shutil

plt.rcParams['font.family'] = 'serif'

# (16 May 2025)
# https://docs.python.org/3/library/shutil.html#shutil.which
plt.rcParams['text.usetex'] = (shutil.which(cmd='pdflatex') != None or
                               shutil.which(cmd='latex') != None)
