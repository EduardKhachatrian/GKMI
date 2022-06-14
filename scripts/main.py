from src.gkmi import GKMI, normalize
from src.performance_analysis import perform_classification

import random
import timeit
import logging
import argparse
import matplotlib
import numpy as np
import scipy.io as io
from skimage import exposure
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

start = timeit.default_timer()

# Creating the log file
logging.basicConfig(filename = 'gkmilogfile.log')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

parser   = argparse.ArgumentParser()
parser.add_argument(
    '--dataset', '-d',
    help ='name of the dataset (Pavia, KSC, Salinas)',
    )
args     = parser.parse_args()
dataset  = args.dataset


if dataset=='Pavia':
	data         = normalize(io.loadmat('test_datasets/PaviaU.mat')['paviaU'])
	ground_truth = io.loadmat('test_datasets/PaviaU_gt.mat')['paviaU_gt']
	classes = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets', 
			   'Bare Soil', 'Bitumen', 'Self-Blocking Bricks', 'Shadows']

if dataset=='KSC':
	data         = normalize(io.loadmat('test_datasets/KSC.mat')['KSC'])
	ground_truth = io.loadmat('test_datasets/KSC_gt.mat')['KSC_gt']
	classes = ['Shrub', 'Willow swamp', 'CP hammrock', 'CP/Oak', 'Slash pine', 
			   'Oak/Broadleaf', 'Hardwood swamp', 'Graminoid marsh', 'Spartina marsh', 
			   'Cattail marsh', 'Salt marsh', 'Mud flats', 'Water']

if dataset=='Salinas':
	data         = normalize(io.loadmat('test_datasets/Salinas.mat')['salinas'])
	ground_truth = io.loadmat('test_datasets/Salinas_gt.mat')['salinas_gt']
	classes = ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 
			   'Fallow_rough_plow', 'Fallow_smooth', 'Stubble', 'Celery',
			   'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds',
			   'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk',
			   'Lettuce_romaine_7wk', 'Vinyard_untrained', 'Vinyard_vertical_trellis']

rows, cols = ground_truth.shape

# Running GKMI attribute selection
pixel_idx, idx_GKMI = GKMI(data, n_clusters='auto', attribute_n=0, 
						   n_superpixels=20, segmentation_algorithm='slic')

# Running parallel classification
OA, Kappa, AA, cm, classified_map = perform_classification(data, ground_truth, pixel_idx, idx_GKMI, 
                                                           test_size=0.8, classifier='RF')

print  ('Performance Evaluation')
print  ('%s : %s' % ('Overall Accuracy', np.round(OA,1)))
print  ('%s : %s' % ('Average Accuracy', np.round(AA,1)))
print  ('%s : %s' % ('Kappa', np.round(Kappa,1)))

logger.info('Performance Evaluation')
logger.info('%s : %s' % ('Overall Accuracy', np.round(OA,1)))
logger.info('%s : %s' % ('Average Accuracy', np.round(AA,1)))
logger.info('%s : %s' % ('Kappa', np.round(Kappa,1)))
logger.info('%s : %s' % ('Confusion Matrix', np.round(cm,2)))

number_of_colors = len(np.unique(classified_map))
n_bins = np.arange(1,number_of_colors+1)

colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors)]
    
#classes = ['Class ' + str(i+1) for i in range(number_of_colors)]

values = np.unique(classified_map.ravel())
patches = [ mpatches.Patch(color=colors[i], label=classes[i].format(l=values[i]) ) for i in range(len(values)) ]

plt.figure(figsize=(20,15))
for n_bin in n_bins:
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('', colors, N=n_bin)
    plt.imshow(np.reshape(classified_map, (rows, cols)), cmap=cmap)
plt.axis('off')
plt.legend(handles=patches, bbox_to_anchor=(1.01,1), loc=2, borderaxespad=0., fontsize=20)
plt.savefig('/Users/ekh011/Desktop/script/classified_maps/' + str(dataset) + '_classified.eps', dpi=200, 
	transparent=True, bbox_inches='tight')

stop = timeit.default_timer()

print  ('%s : %s' % ('Execution Time [sec]', stop - start))
logger.info('%s : %s' % ('Execution Time [sec]', stop - start))


