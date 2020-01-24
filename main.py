from data import *

def main():
	x_data = np.array(load_x('data/X_train.h5'))
	plot_eeg_sample(x_data[0, 0])

if __name__=='__main__':
	main()