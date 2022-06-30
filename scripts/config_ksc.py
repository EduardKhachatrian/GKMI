import os
import logging

dataset = "KSC"
data_dir = "/Users/ekh011/Desktop/script/test_datasets/"
classes = ['Shrub', 'Willow swamp', 'CP hammrock', 'CP/Oak', 'Slash pine', 
			   'Oak/Broadleaf', 'Hardwood swamp', 'Graminoid marsh', 'Spartina marsh', 
			   'Cattail marsh', 'Salt marsh', 'Mud flats', 'Water']
output_dir = "classification_outputs/KSC"

n_superpixels = 150
attribute_n   = 70
n_clusters    = 'auto'
segmentation_algorithm = 'slic'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

logging.basicConfig(filename = os.path.join(output_dir,'gkmilogfile.log'))
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logger.info('%s : %s' % ('Number of superpixels to create', n_superpixels))
logger.info('%s : %s' % ('Attribute idx to use for superpixel segmentation', attribute_n))
logger.info('%s : %s' % ('Number of relevant attributes to select', n_clusters))
logger.info('%s : %s' % ('Segmentation algorithm', segmentation_algorithm))

logger.info('%s : %s' % ('Dataset Classes', classes))


if not os.path.exists(data_dir):
	print(f"Directory {data_dir} doesn't exist!")
	logger.info(f"Directory {data_dir} doesn't exist!")