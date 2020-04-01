# import numpy as np
# import pandas as pd
# from scipy.stats import norm
# from scipy.optimize import minimize_scalar, minimize, Bounds
# from matplotlib.pyplot import matshow, show
#
# # Dataset: https://www.kaggle.com/olistbr/brazilian-ecommerce/data
# # Algorithm input
# # k = 2
# # b = np.array([0, 0, 0, 0])
# # Sigma = np.random.rand(b.size, b.size)
# # Mu = np.random.random(b.size)
#
# # Import
# customers = pd.read_csv('data/olist_customers_dataset.csv', header=0, usecols=['customer_id', 'customer_unique_id'])
# orders = pd.read_csv('data/olist_orders_dataset.csv', header=0, usecols=['order_id', 'customer_id'])
# products = pd.read_csv('data/olist_order_items_dataset.csv', header=0, usecols=['order_id', 'product_id', 'price'])
#
# # Pre-processing
# products['mean_price'] = products.groupby('product_id')['price'].transform('mean')
# products['bundable'] = 0
# customers_orders = pd.merge(customers, orders, on="customer_id")
# customer_product = pd.merge(products, customers_orders, how='inner', on='order_id')
# customer_product.drop(['order_id', 'customer_id', 'price'], axis=1, inplace=True)
# del [[customers, orders, products, customers_orders]]
# customer_product_reduced = customer_product[:1000]
#
# # Transaction matrix
# transaction_df = pd.get_dummies(customer_product_reduced.customer_unique_id).groupby(customer_product_reduced.product_id).apply(max)
# matrix = transaction_df.to_numpy()

# def variance_minimization(Sigma, b, k):
#     def Constraint(b):
#         return b.sum() - k
#
#     def ObjectiveFunction(b):
#         return b.dot(Sigma).dot(b)
#
#     constraint = {'type': 'eq', 'fun': Constraint}
#     bounds = Bounds(lb=0, ub=1)
#
#     minimized = minimize(ObjectiveFunction, b, bounds=bounds, method='SLSQP', constraints=constraint, options={'disp': True})
#     b_hat = np.random.rand(len(minimized.x)) <= minimized.x
#     return b_hat
# def maximum_profit(variance):
#   def real_buyers(delta):
#     return delta
#
#   def objective_function(delta):
#     return (-1) * (norm.ppf(1 - delta) * variance + 1) * real_buyers(delta)
#
#   results = minimize_scalar(objective_function, method='bounded', bounds=(0,1))
#
#   return (-1) * results.fun
# def potential_profit_gain(mean, variance, cost):
#   return (mean - cost) * (maximum_profit(0) - maximum_profit(variance/(mean - cost)))