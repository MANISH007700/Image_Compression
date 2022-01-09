import configparser
global cwd, original_image_path

config = configparser.ConfigParser()
config.read('config.ini')
cwd = config['PATHS']['cwd']
original_image_path = config['PATHS']['original_image_path']
compressed_image_path = config['PATHS']['compressed_image_path']