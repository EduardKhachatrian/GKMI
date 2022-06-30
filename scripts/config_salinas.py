import os
import logging

dataset = "Pavia"
data_dir = "/Users/ekh011/Desktop/script/test_datasets/"
classes = ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 
			   'Fallow_rough_plow', 'Fallow_smooth', 'Stubble', 'Celery',
			   'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds',
			   'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk',
			   'Lettuce_romaine_7wk', 'Vinyard_untrained', 'Vinyard_vertical_trellis']
output_dir = "classification_outputs/Salinas"

n_superpixels = 25
attribute_n   = 110
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