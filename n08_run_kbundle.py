import exp_setting

import kbundle
import numpy as np 
import math
import profit 
import json
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import time

import time

def run_qp(mean_vec, cov_mat, alpha, k, separate_profit_vec):
	print ('Testando algoritmo de relaxation QP')

	start_time = time.time()
	bundle_set, profit_bundle = kbundle.Kbundle_QP_relaxation(mean_vec, cov_mat, alpha, k, separate_profit_vec)
	running_time = time.time() - start_time

	return profit_bundle, bundle_set, running_time

if __name__ == '__main__':
	print ('Script 8: Gerando um bundle de tamanho K')
	json_data = json.load(open('data/drv_mean_std.json'))
	mean_vec = json_data['mean_vec']
	std_vec = json_data['std_vec']
	correlation_mat = pd.read_csv('data/drv_cov_psd.csv', delimiter=',', header=None).as_matrix()

	std_array = np.asarray(std_vec).reshape((len(std_vec), 1))
	std_scale_mat = np.dot(std_array, np.transpose(std_array))
	cov_mat = np.multiply(correlation_mat, std_scale_mat).astype(np.double)

	alpha = 0 # dummy

	# precompute the separate sale profit
	separate_profit_vec = profit.separate_sale_profit(mean_vec, cov_mat, alpha)

	profit_vec = []
	running_time_vec = []
	bundle_size_vec = [2, 3, 4]
	for bundle_size in bundle_size_vec:
		profit_bundle, bundle_set, running_time = run_qp(mean_vec, cov_mat, alpha, bundle_size, separate_profit_vec)
		profit_vec.append(profit_bundle)
		running_time_vec.append(running_time)
		with open('data/result_kbundle_' + str(bundle_size) + '.json', 'w') as output_file:
			json.dump({"profit": profit_bundle, "set": list(bundle_set)}, output_file, indent=4)

	with open('data/result_kbundle_profits.json', 'w') as output_file:
		json.dump({"bundle_size_vec": bundle_size_vec, "profit_vec": profit_vec, "running_time_vec": running_time_vec}, output_file, indent=4)

	plt.plot(bundle_size_vec, profit_vec, color='black')
	plt.xlabel('bundle size')
	plt.ylabel('profit after bundling')
	plt.title('bundle size and profit after bundling')
	plt.savefig('images/large-size-profit.png', format='png', dpi=1000)