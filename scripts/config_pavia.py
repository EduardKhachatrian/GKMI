import os
import logging

dataset = "Pavia"
data_dir = "/Users/ekh011/Desktop/script/test_datasets/"
classes = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets', 
			   'Bare Soil', 'Bitumen', 'Self-Blocking Bricks', 'Shadows']
output_dir = "classification_outputs/Pavia"

n_superpixels = 50
attribute_n   = 50
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