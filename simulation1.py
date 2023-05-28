# wymagane biblioteki
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import joblib

# funkcja do wyznaczenia odchylenia standardowego reszt
def odchylenie_reszt(real, pred, m):
    return (sum((real-pred)**2)*(1/(len(real)-m-1)))**0.5


df = pd.read_parquet('data/cars_clean_features.parquet')
df.info(verbose=False)

# zmienne określające wielkość zbioru (możliwość ograniczenia do szybszego testowania) oraz ilość rdzeni wykorzystywnaych do równolgłych obliczeń
dataset_constraint = -1
n_sim = 6 

# podział zbioru danych
x = df.iloc[:dataset_constraint,1:]
y = df.iloc[:dataset_constraint,0] 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05,stratify=x['marka_var'], random_state=34)


# rozwazane modele
lm = LinearRegression()
ridge = Ridge(alpha=0.3, solver='svd')
tree = DecisionTreeRegressor(max_depth=23, min_samples_leaf=15, min_samples_split=28, max_features=0.8)
rf = RandomForestRegressor(max_depth=18, min_samples_split=4, n_estimators=170, random_state=34)

x = df.iloc[:dataset_constraint,1:]
y = df.iloc[:dataset_constraint,0] 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05,stratify=x['marka_var'], random_state=34)

# model do generowania danych
model = LinearRegression()
model.fit(x,y)

def simulation(iter=1, model=model):
    df_results = pd.DataFrame()
    # generowanie danych
    yt = model.predict(x_train)
    std_reszt = odchylenie_reszt(y_train, yt, x_train.shape[1])
    
    rand_y_train =  pd.Series(model.predict(x_train)) + pd.Series(np.random.normal(0, std_reszt, size=len(x_train)))
    rand_y_test = pd.Series(model.predict(x_test)) + pd.Series(np.random.normal(0, std_reszt, size=len(x_test)))

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

results_linear = joblib.Parallel(n_jobs = 6, verbose = 13)(joblib.delayed(simulation)(i) for i in range(n_sim))

full_results = pd.DataFrame()
for dfs in results_linear:
    full_results = pd.concat([full_results, dfs])

full_results['lm_error'] =full_results['lm'] - full_results['rand_y']
full_results['ridge_error'] = full_results['ridge']- full_results['rand_y']
full_results['tree_error'] = full_results['tree']- full_results['rand_y']
full_results['rf_error'] = full_results['rf']- full_results['rand_y']

print('saving files...')
file_name = 'sim_results/sim1_TEST.parquet'
full_results.to_parquet(file_name)
full_results.to_csv(file_name)
print('finished')

print(abs(full_results.groupby('observation').mean()).mean())


