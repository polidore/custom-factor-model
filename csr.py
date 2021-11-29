# %%
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# %%
data = pd.read_csv('./data.csv').iloc[:, 1:]
data
# %%
cm = pd.read_csv('./constraints_matrix.csv')
cm.rename(columns={'Unnamed: 0': 'factor_group'}, inplace=True)
cm.set_index('factor_group', inplace=True)
cm

# %%
glm = sm.GLM(data['total_return_1d'], data.iloc[:, 2:], var_weights=data['wgt'])
# %%
res = glm.fit_constrained(constraints=(cm, np.zeros(shape=(len(cm)))))
res.summary()
# %%

