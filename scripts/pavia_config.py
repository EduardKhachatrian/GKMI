import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

dataset = "Pavia"
data_dir = "/Users/ekh011/Desktop/script/test_datasets/"
classes = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets', 
			   'Bare Soil', 'Bitumen', 'Self-Blocking Bricks', 'Shadows']
output_dir = "classification_outputs/Pavia"

n_superpixels = 50
attribute_n   = 50
n_clusters    = 'auto'
segmentation_algorithm = 'slic'

if not os.path.exists(data_dir):
	print(f"Directory {data_dir} doesn't exist!")
	logger.info(f"Directory {data_dir} doesn't exist!") 

if not os.path.exists(output_dir):
    os.makedirs(output_dir)