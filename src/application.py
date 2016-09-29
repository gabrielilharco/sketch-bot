from data_processing import *

if __name__ == '__main__':
	folders = maybe_extract('data.tar.gz', 250)
	display_random_image('data')