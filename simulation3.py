import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import joblib

def odchylenie_reszt(real, pred, m):
    return (sum((real-pred)**2)*(1/(len(real)-m-1)))**0.5


df = pd.read_parquet('data/cars_clean_features.parquet')
df.info(verbose=False)

dataset_constraint = -1
n_sim = 500

x = df.iloc[:dataset_constraint,1:]
y = df.iloc[:dataset_constraint,0] 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05,stratify=x['marka_var'], random_state=34)


# rozwazane modele
lm = LinearRegression()
ridge = Ridge(alpha=0.3, solver='svd')
tree = DecisionTreeRegressor(max_depth=23, min_samples_leaf=15, min_samples_split=28, max_features=0.8)
rf = RandomForestRegressor(max_depth=18, min_samples_split=4, n_estimators=170, random_state=34)

x_gen = df.iloc[:dataset_constraint,1:]
y_gen = df.iloc[:dataset_constraint,0] 

x_gen['x0x2'] = x_gen.iloc[:,0] * x_gen.iloc[:,2]

x_gen_train, x_gen_test, y_gen_train, y_gen_test = train_test_split(x_gen, y_gen, test_size=0.05,stratify=x_gen['marka_var'], random_state=34)

# model do generowania danych
model = LinearRegression()
model.fit(x_gen,y_gen)

def simulation3(iter=1, model=model):
    df_results = pd.DataFrame()

    std_reszt = odchylenie_reszt(y_gen_train, model.predict(x_gen_train), x_gen_train.shape[1])
    
    # shifted gamma distribution
    y_mu = 0
    y_sig2 = std_reszt**2/100
    y_as = 1.2
    k = 4/(y_as**2)
    Teta = (y_sig2/k)**0.5
    EX = k*Teta
    e_train = pd.Series(np.random.gamma(shape=k, scale=Teta, size=len(x_gen_train))-EX+y_mu)
    e_test = pd.Series(np.random.gamma(shape=k, scale=Teta, size=len(x_gen_test))-EX+y_mu)

    model_pred = np.array((x_gen_train.iloc[:,:-1] * model.coef_[:-1] ).sum(axis=1) + x_gen_train.iloc[:,-1] * model.coef_[-1]*3 + model.intercept_ )
    model_pred_test = np.array((x_gen_test.iloc[:,:-1] * model.coef_[:-1] ).sum(axis=1) + x_gen_test.iloc[:,-1] * model.coef_[-1]*3 + model.intercept_ )
    rand_y_train = pd.Series(model_pred) + e_train 
    rand_y_test = pd.Series(model_pred_test) + e_test

    lm.fit(x_train, rand_y_train)
    ridge.fit(x_train, rand_y_train)
    tree.fit(x_train, rand_y_train)
    rf.fit(x_train, rand_y_train)
    
    df_results['rand_y'] = rand_y_test
    df_results['lm'] = lm.predict(x_test)
    df_results['ridge'] = ridge.predict(x_test)
    df_results['tree'] = tree.predict(x_test)
    df_results['rf'] = rf.predict(x_test)
    df_results['observation'] = list(range(0,len(rand_y_test)))
    df_results['iteration'] = iter
    
    return df_results

results_nonlinear = joblib.Parallel(n_jobs = 6, verbose = 13)(joblib.delayed(simulation3)(i) for i in range(n_sim))

full_results = pd.DataFrame()
for dfs in results_nonlinear:
    full_results = pd.concat([full_results, dfs])

full_results['lm_error'] = full_results['lm'] - full_results['rand_y']
full_results['ridge_error'] = full_results['ridge']- full_results['rand_y']
full_results['tree_error'] = full_results['tree']- full_results['rand_y']
full_results['rf_error'] = full_results['rf']- full_results['rand_y']

print('saving files...')
file_name = 'sim_results/sim3_results500_2_0mean.parquet'
full_results.to_parquet(file_name)
#full_results.to_csv(file_name)
print('finished')

print(abs(full_results.groupby('observation').mean()).mean())
