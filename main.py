import numpy as np
import pandas as pd
import json, math, time, random
import pi, profit, kbundle
import matplotlib.pyplot as plt
from scipy.stats import norm, mvn
import scipy.stats as stat
from multiprocessing import Pool


# Dataset: https://www.kaggle.com/olistbr/brazilian-ecommerce/data

def average_delta(a, sales_count_vec):
    max_sales_count = max(sales_count_vec)
    N_product = len(sales_count_vec)
    delta_vec = [None] * N_product
    for i in range(N_product):
        delta_vec[i] = np.log((sales_count_vec[i] / max_sales_count) * (a - 1) + 1) / np.log(a)

    return (sum(delta_vec) / N_product)


def search(sales_count_vec):
    epsilon = 0.0001
    a_left = 0
    a_right = 20000
    while a_right - a_left > epsilon:
        a = (a_left + a_right) / 2
        average = average_delta(a, sales_count_vec)
        if average > 0.5:  # should decrease a
            a_right = a
        elif average < 0.5:
            a_left = a
        else:
            break
    return a


def get_copurchase_prob_mat_raw(sales_vec, copurchase_mat):
    a = 16674.27
    N_product = len(sales_vec)
    copurchase_prob_mat = np.zeros(shape=(N_product, N_product))
    sales_max = max(sales_vec)
    for i in range(N_product):
        for j in range(N_product):
            # f^{-1} function
            copurchase_prob_mat[i][j] = np.log((copurchase_mat[i][j] / sales_max) * (a - 1) + 1) / np.log(a)
    return copurchase_prob_mat


def get_mean(delta, price, std):
    delta = min(0.99, delta)
    delta = max(0.01, delta)
    return norm.ppf(delta) * std + price


def empirical_mean_variance(products):
    N_product = len(products)
    mean_vec = [None] * N_product
    std_vec = [None] * N_product

    sales_count_vec = []
    price_vec = []
    for product in products:
        sales_count_vec.append(product['sold_units'])
        price_vec.append(product['mean_price'])

    # print ('average price:', sum(price_vec)/len(price_vec))

    max_sales_count = max(sales_count_vec)
    # print (max_sales_count)

    delta_vec = [None] * N_product
    # f^{-1} function
    a = 16674.27
    for i in range(N_product):
        delta_vec[i] = np.log((sales_count_vec[i] / max_sales_count) * (a - 1) + 1) / np.log(a)

    # print ('mean delta', sum(delta_vec)/N_product)

    std = 15  # fixed variance, you may change it to other values

    for i in range(N_product):
        mean_vec[i] = get_mean(delta_vec[i], price_vec[i], std)
        std_vec[i] = std

    # print ('mean of valuations and mean of price', sum(mean_vec)/N_product, sum(price_vec)/N_product)

    return mean_vec, std_vec, delta_vec


def smf_pgd(Cov, W, f):
    N_product = Cov.shape[0]
    # initialize
    X_mat = []
    for i in range(N_product):
        X_vec = []
        for j in range(f):
            X_vec.append(random.random())
        norm2 = math.sqrt(sum(list(map(lambda x: x ** 2, X_vec))))
        X_vec_normalized = list(map(lambda x: x / norm2, X_vec))
        X_mat.append(X_vec_normalized)

    X = np.array(X_mat).T

    for i in range(N_product):
        W[i][i] = 0

    N_iteration = 100
    eta = 0.00005

    grad_sum = np.zeros(N_product)

    cost_vec = []
    for ite in range(N_iteration):
        # print('iteration', ite)
        cost = np.multiply(np.square(np.dot(X.T, X) - Cov), W).sum() / (N_product ** 2)
        cost_vec.append(cost)
        # print('Average cost:', cost)
        right_part = np.multiply(np.dot(X.T, X) - Cov, W)
        grad = np.dot(X, right_part)

        Y = X - grad * (eta / math.sqrt(ite + 1))
        Y_2norm_col = np.linalg.norm(Y, axis=0)
        # print('Average 2-norm:', Y_2norm_col.sum() / N_product)
        X = Y / Y_2norm_col

    predicted_cov_mat = np.dot(X.T, X)
    plt.plot(cost_vec)
    return predicted_cov_mat, X


def prob_to_correlation(prob, s, t):  # s = mu_1/sigma_1, t = mu_2/sigma_2
    epsilon = 0.00001
    # search the unique correlation coefficient corresponding to the desired probability
    lower = np.array([-s, -t])
    upper = np.array([0, 0])  # dummy
    infin = np.array([1, 1])
    rho_left = -1
    rho_right = 1
    while rho_right - rho_left > epsilon:
        rho = (rho_left + rho_right) / 2
        error, value, inform = mvn.mvndst(lower, upper, infin, np.array([rho]))
        if value < prob:
            rho_left = rho
        elif value > prob:
            rho_right = rho
        else:
            break

    return rho_left


def empirical_cor(mean_vec, price_vec, std_vec, copurchase_prob_mat):
    N_product = len(mean_vec)
    pool = Pool(processes=8)  # parallel, set as number of cores
    cor_mat = np.zeros(shape=(N_product, N_product))
    for i in range(N_product):
        arg_vec = []
        for j in range(N_product):
            prob = copurchase_prob_mat[i][j]
            prob = min(1, prob)

            s = (mean_vec[i] - price_vec[i]) / std_vec[i]
            t = (mean_vec[j] - price_vec[j]) / std_vec[j]
            arg_vec.append((prob, s, t))
        cor_vec = pool.starmap(prob_to_correlation, arg_vec)

        for j in range(N_product):
            cor_mat[i][j] = cor_vec[j]

    pool.close()
    pool.join()
    return cor_mat


def correlation_to_prob(rho, s, t):
    lower = np.array([-s, -t])
    upper = np.array([0, 0])  # no use
    infin = np.array([1, 1])
    error, value, inform = mvn.mvndst(lower, upper, infin, np.array([rho]))
    return value


def opt_enumerate(mean_vec, cov_mat, k, separate_profit_vec):
    alpha = 0  # dummy
    N = len(mean_vec)
    products = list(range(N))
    max_profit = -math.inf
    opt_bundle = None
    for bundle_set in itertools.combinations(products, k):
        bundle_vec = [0] * N
        for i in bundle_set:
            bundle_vec[i] = 1
        revenue = profit.profit_bundle(bundle_set, bundle_vec, mean_vec, cov_mat, alpha, separate_profit_vec)
        if revenue > max_profit:
            max_profit = revenue
            opt_bundle = bundle_set

    return float(max_profit)


def evaluate_prediction(predicted_cov_mat, copurchase_mat, mean_vec, price_vec, std_vec, sales_vec, delta_vec, a):
    print(predicted_cov_mat)
    # print('evaluating the predicted covariance matrix...')
    N_product = len(mean_vec)

    # predicted probability that two products are co-purchased
    predicted_copurchase_mat = np.zeros((N_product, N_product))

    max_sales = max(sales_vec)

    f_delta_vec = list(map(lambda x: pi.mapping(x, a) * 100000, delta_vec))

    pool = Pool(processes=8)
    for i in range(N_product):

        arg_vec = []
        for j in range(i, N_product):
            s = (mean_vec[i] - price_vec[i]) / std_vec[i]
            t = (mean_vec[j] - price_vec[j]) / std_vec[j]
            rho = predicted_cov_mat[i][j]
            arg_vec.append((rho, s, t))
        value_vec = pool.starmap(correlation_to_prob, arg_vec)

        for j in range(i, N_product):
            value = value_vec[j - i]  # minus the index

            predicted_copurchase_mat[i][j] = value  # we do not do the mapping because the rank doesn't change
            predicted_copurchase_mat[j][i] = predicted_copurchase_mat[i][j]

    pool.close()
    pool.join()

    rank_vec = [None] * N_product
    for i in range(N_product):
        predicted_copurchase_vec = predicted_copurchase_mat[i]
        rank_vec[i] = list(stat.rankdata(predicted_copurchase_vec))

    rank_mat = np.array(rank_vec)

    total_rank = np.multiply(rank_mat, copurchase_mat).sum()
    copurchase_sum = copurchase_mat.sum()

    average_rank = total_rank / copurchase_sum / N_product
    return 1 - average_rank  # because we sort reversely


def productId2productIndex(products, list_of_ids):
    productsIndexes = []
    for id in list_of_ids:
        for index, product in enumerate(products):
            if id == product['product_id']:
                productsIndexes.append(index)
    return productsIndexes


if __name__ == '__main__':

    tempos_execucao = [0, 0, 0, 0, 0]

    ###############################################################################
    ###############################################################################
    ################################## Parte 1 ####################################
    ############################# Extração dos dados ##############################
    ###############################################################################
    ###############################################################################
    print("Parte 1")
    start = time.time()

    # Import
    customers = pd.read_csv('data/olist_customers_dataset.csv', header=0, usecols=['customer_id', 'customer_unique_id'])
    orders = pd.read_csv('data/olist_orders_dataset.csv', header=0, usecols=['order_id', 'customer_id'])
    df_products = pd.read_csv('data/olist_order_items_dataset.csv', header=0,
                              usecols=['order_id', 'product_id', 'price'])

    # Pre-processing
    df_products['mean_price'] = df_products.groupby('product_id')['price'].transform('mean')
    customers_orders = pd.merge(customers, orders, on="customer_id")
    customer_product = pd.merge(df_products, customers_orders, how='inner', on='order_id')
    customer_product.drop(['order_id', 'customer_id', 'price'], axis=1, inplace=True)
    del [[customers, orders, customers_orders]]
    customer_product_reduced = customer_product[:1000]

    # Transaction matrix
    transaction_df = pd.get_dummies(customer_product_reduced.customer_unique_id).groupby(
        customer_product_reduced.product_id).apply(max)

    # Copurchase matrix
    pre_matrix = transaction_df.transpose()
    matrix_int = pre_matrix.astype(int)
    matrix = matrix_int.T.dot(matrix_int)
    np.fill_diagonal(matrix.values, 0)

    # Lista de Produtos
    sold_units = customer_product_reduced['product_id'].value_counts()
    customer_product_reduced.assign(sold_units=0)

    for product, quantity in sold_units.iteritems():
        customer_product_reduced.loc[customer_product_reduced['product_id'] == product, 'sold_units'] = quantity

    customer_product_reduced.drop_duplicates('product_id', inplace=True)
    del customer_product_reduced['customer_unique_id']

    #   Matriz de correlação
    np.savetxt('data/drv_copurchase_selected.csv', matrix, delimiter=",", fmt="%i")

    #   Lista de produtos
    customer_product_reduced.to_json('data/drv_products_selected.json', orient='records')

    tempos_execucao[0] = time.time() - start

    ###############################################################################
    ###############################################################################
    ################################## Parte 2 ####################################
    ########################### Elaboração dos inputs #############################
    ###############################################################################
    ###############################################################################
    print("Parte 2")
    start = time.time()

    products = json.load(open('data/drv_products_selected.json'))

    # Produtos obrigatórios
    mandatory_products_ids = ['4244733e06e7ecb4970a6e2683c13e61', '368c6c730842d78016ad823897a372db']
    mandatory_products_indexes = productId2productIndex(products, mandatory_products_ids)
    N_mandatory_products = len(mandatory_products_indexes)
    N_product = len(products)
    K = 2
    if not K >= N_mandatory_products:
        raise SystemExit()
    sales_vec = []
    price_vec = []
    for product in products:
        price_vec.append(product['mean_price'])
        sales_vec.append(product['sold_units'])

    copurchase_mat = pd.read_csv('data/drv_copurchase_selected.csv', delimiter=',', header=None).to_numpy()
    copurchase_prob_mat = get_copurchase_prob_mat_raw(sales_vec, copurchase_mat)

    tempos_execucao[1] = time.time() - start
    ###############################################################################
    ###############################################################################
    ################################## Parte 3 ####################################
    ############################ Parâmetros do modelo #############################
    ###############################################################################
    ###############################################################################
    print("Parte 3")
    start = time.time()

    ########## Média u
    mean_vec, std_vec, delta_vec = empirical_mean_variance(products)

    ########## Correlação empírica
    cov_mat = empirical_cor(mean_vec, price_vec, std_vec, copurchase_prob_mat)

    # ########## Correlação otimizada
    f = 20  # number of features of each product
    alpha = 0.1  # baseline weight for 0 co-purcahse
    W = copurchase_mat + alpha * np.ones((N_product, N_product))
    predicted_cov_mat, X = smf_pgd(cov_mat, W, f)
    tempos_execucao[2] = time.time() - start

    ###############################################################################
    ###############################################################################
    ################################## Parte 4 ####################################
    ############################# Execução do modelo ##############################
    ###############################################################################
    ###############################################################################
    print("Parte 4")
    start = time.time()

    std_array = np.asarray(std_vec).reshape((len(std_vec), 1))
    std_scale_mat = np.dot(std_array, np.transpose(std_array))
    cov_mat = np.multiply(predicted_cov_mat, std_scale_mat).astype(np.double)

    alpha = search(sales_vec)
    # precompute the separate sale profit
    separate_profit_vec = profit.separate_sale_profit(mean_vec, cov_mat, alpha)

    profit_vec = []
    bundle_size_vec = [2]
    # bundle_size_vec_norm = np.where(np.asarray(bundle_size_vec) >= N_mandatory_products)
    for bundle_size in bundle_size_vec:
        bundle_set, profit_bundle = kbundle.Kbundle_QP_relaxation(mean_vec, cov_mat, alpha, bundle_size,
                                                                  separate_profit_vec, mandatory_products_indexes)
        profit_vec.append(profit_bundle)

        with open('data/result_kbundle_' + str(bundle_size) + '.json', 'w') as output_file:
            json.dump({"profit": profit_bundle, "set": list(bundle_set)}, output_file, indent=4)

    with open('data/result_kbundle_profits.json', 'w') as output_file:
        json.dump({"bundle_size_vec": bundle_size_vec, "profit_vec": profit_vec}, output_file, indent=4)

    tempos_execucao[3] = time.time() - start

    max_products_sold = int(customer_product_reduced['sold_units'].max())
    products_sold_axis = np.linspace(1, max_products_sold, max_products_sold)

    count_sales = customer_product_reduced.groupby('sold_units').count()

    plt.bar(count_sales.index,  count_sales['mean_price'])
    plt.show()
    ###############################################################################
    ###############################################################################
    ################################## Parte 5 ####################################
    ########################## Avaliação dos resultados ###########################
    ###############################################################################
    ###############################################################################
    print("Parte 5")
    start = time.time()

    # ########## Avaliação do Bundle
    # print ('loading the files...')
    # json_data = json.load(open('data/drv_mean_std.json'))
    # mean_vec = json_data['mean_vec']
    # std_vec = json_data['std_vec']
    # correlation_mat = pd.read_csv('data/drv_cov_psd.csv', delimiter=',', header=None).as_matrix()
    # N_product = len(mean_vec)

    # std_array = np.asarray(std_vec).reshape((len(std_vec), 1))
    # std_scale_mat = np.dot(std_array, np.transpose(std_array))
    # cov_mat = np.multiply(correlation_mat, std_scale_mat).astype(np.double)

    # alpha = 0 # dummy

    # # precompute the separate sale profit
    # separate_profit_vec = profit.separate_sale_profit(mean_vec, cov_mat, alpha)

    # N_vec = [20]
    # k_vec = [2, 3, 4, 5, 6, 7, 8, 9]
    # sample_size = 24
    # result_vec = []
    # pool = Pool(processes = 8)
    # for N in N_vec:
    #     print ("N =", N)
    #     for k in k_vec:
    #         print (">> k =", k)
    #         profit_algo_vec = []

    #         arg_vec = []
    #         running_time_algo_vec = []
    #         for s in range(sample_size):
    #             rand_indices = list(np.random.choice(N_product, N))
    #             mean_vec_sub = [mean_vec[i] for i in rand_indices]
    #             cov_mat_sub = cov_mat[:, rand_indices][rand_indices]
    #             separate_profit_vec_sub = [separate_profit_vec[i] for i in rand_indices]

    #             start_time = time.time()
    #             _, profit_algo = kbundle.Kbundle_QP_relaxation(mean_vec_sub, cov_mat_sub, alpha, k, separate_profit_vec_sub)
    #             running_time_algo = time.time() - start_time
    #             running_time_algo_vec.append(running_time_algo)

    #             arg_vec.append((mean_vec_sub, cov_mat_sub, k, separate_profit_vec_sub))

    #             profit_algo_vec.append( float(profit_algo) )

    #         print ('N = ', N, 'k = ', k)

    #         start_time = time.time()
    #         profit_opt_vec = pool.starmap(opt_enumerate, arg_vec)
    #         profit_opt_vec = list(profit_opt_vec)
    #         running_time_enumerate = time.time() - start_time

    #         ratio_vec = list(map(lambda x,y:x/y, profit_algo_vec, profit_opt_vec))

    #         result = {
    #             'N': N,
    #             'k': k,
    #             'profit_algo_vec': list(profit_algo_vec),
    #             'profit_opt_vec': list(profit_opt_vec),
    #             'ratio_vec': ratio_vec,
    #             'running_time_algo': sum(running_time_algo_vec) / sample_size,
    #             'running_time_enumerate': running_time_enumerate / sample_size
    #         }
    #         print ('average ratio:', sum(ratio_vec)/sample_size)
    #         with open('data/result_heterogeneous_' + str(N) + '_' + str(k) +'.json', 'w') as output_file:
    #             json.dump(result, output_file, indent=4)
    #         result_vec.append(result)

    # with open('data/result_approximation.json', 'w') as output_file:
    #     json.dump(result_vec, output_file, indent=4)

    ######### Avaliação dos parâmetros do modelo

    accuracy = evaluate_prediction(predicted_cov_mat, copurchase_mat, mean_vec, price_vec, std_vec, sales_vec,
                                   delta_vec, alpha)
    print('Acurácia: ', accuracy)
    tempos_execucao[4] = time.time() - start
    print(tempos_execucao)
