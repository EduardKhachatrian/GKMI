from src.gkmi import GKMI, normalize
from src.performance_analysis import perform_classification

import os
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
	from config_pavia import (
		data_dir,
		classes,
		output_dir,
		n_clusters,
		n_superpixels,
		attribute_n,
		segmentation_algorithm
	)
	data         = normalize(io.loadmat(os.path.join(data_dir,'PaviaU.mat'))['paviaU'])
	ground_truth = io.loadmat(os.path.join(data_dir,'PaviaU_gt.mat'))['paviaU_gt']
	

if dataset=='KSC':
	from config_ksc import (
		data_dir,
		classes,
		output_dir,
		n_clusters,
		n_superpixels,
		attribute_n,
		segmentation_algorithm
	)
	data         = normalize(io.loadmat(os.path.join(data_dir,'KSC.mat'))['KSC'])
	ground_truth = io.loadmat(os.path.join(data_dir,'KSC_gt.mat'))['KSC_gt']

if dataset=='Salinas':
	from config_salinas import (
		data_dir,
		classes,
		output_dir,
		n_clusters,
		n_superpixels,
		attribute_n,
		segmentation_algorithm
	)
	data         = normalize(io.loadmat(os.path.join(data_dir,'Salinas.mat'))['salinas'])
	ground_truth = io.loadmat(os.path.join(data_dir,'Salinas_gt.mat'))['salinas_gt']

rows, cols = ground_truth.shape

# Running GKMI attribute selection
pixel_idx, idx_GKMI = GKMI( 
	data,
	n_clusters=n_clusters, 
	attribute_n=attribute_n, 
	n_superpixels=n_superpixels, 
	segmentation_algorithm=segmentation_algorithm
	)

# Running parallel classification
OA, Kappa, AA, cm, classified_map = perform_classification(
	data, 
	ground_truth, 
	pixel_idx, 
	idx_GKMI, 
	test_size=0.8, 
	classifier='RF'
)

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
    
values = np.unique(classified_map.ravel())
patches = [ mpatches.Patch(color=colors[i], label=classes[i].format(l=values[i]) ) for i in range(len(values)) ]

plt.figure(figsize=(20,15))
for n_bin in n_bins:
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('', colors, N=n_bin)
    plt.imshow(np.reshape(classified_map, (rows, cols)), cmap=cmap)
plt.axis('off')
plt.legend(handles=patches, bbox_to_anchor=(1.01,1), loc=2, borderaxespad=0., fontsize=20)
plt.savefig(os.path.join(output_dir, str(dataset) + '_classified.eps'), dpi=200, 
	transparent=True, bbox_inches='tight')

io.savemat(os.path.join(output_dir, str(dataset) + '_results.mat'), {
										'pixel indexes' : pixel_idx, 
										'selected indexes' : idx_GKMI, 
										'OA' : OA, 
										'AA' : AA, 
										'Kappa' : Kappa, 
										'map' : classified_map
})

stop = timeit.default_timer()

print  ('%s : %s' % ('Execution Time [sec]', stop - start))
logger.info('%s : %s' % ('Execution Time [sec]', stop - start))


