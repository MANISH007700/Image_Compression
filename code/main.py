import os 
import sys 
import time 

from config_parser import cwd, original_image_path, compressed_image_path
from compression import get_image_compressed

start_time = time.time()

image_name = input("Enter the Image name to be compressed without <quotes> -> ")

(get_image_compressed(image_name, original_image_path, compressed_image_path))

end_time = time.time()
print("Total Time taken in seconds: ", round(end_time - start_time,2))