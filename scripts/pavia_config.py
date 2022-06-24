import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

dataset_name = "pavia"
data_dir = "/test_datasets/pavia"
classes = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets', 
			   'Bare Soil', 'Bitumen', 'Self-Blocking Bricks', 'Shadows']
class_color = [] # maybe?
output_dir = "/output_stuff/pavia"


if not os.path.exists(data_dir):
	print("no data, sorry")
	logger.info("no data, sorry") 

if not os.path.exists(output_dir):
    os.makedirs(output_dir)