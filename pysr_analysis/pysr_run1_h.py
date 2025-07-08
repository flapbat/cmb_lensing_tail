#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pysr import PySRRegressor, TemplateExpressionSpec

import matplotlib.pyplot as plt
import numpy as np

import camb


# In[2]:


plt.rcParams['font.family'] = 'stixgeneral'


# In[3]:


# Load Data

pars     = np.load('CL_data/parameter_test.npy')  # [H0, ombh2, omch2 ] x 100
lensed   = np.load('CL_data/lensed_CL.npy')     # [C_2, ..., C_5000] x 100 (lensed)
unlensed = np.load('CL_data/unlensed_CL.npy')     # [C_2, ..., C_5000] x 100 (unlensed)


# In[4]:


past_ells = 1000
n_ells = 4998 - past_ells

# Truncate to ignore first 1000 l's
y_pysr = lensed[:, past_ells:]/unlensed[:, past_ells:]  #lensing

def moving_average(x):
    val = np.convolve(x, np.ones(500), 'valid') / 500
    return val

# Smoothing the Lensing Tail for Training
do_smoothing = True

if do_smoothing:
    y_pysr[:, 249:-250] = np.apply_along_axis(moving_average, axis = 1, arr = y_pysr[:, :])

# Reformatting data
y_pysr = y_pysr.reshape(-1)
# y_pysr : [par1_c502, par1_c503, ..., par1_c5000, par2_c502, ..., par299_c502, ..., par299_c5000]


# In[5]:


# Reformatting data
X_ells = np.array([ell for ell in range(past_ells + 2, 5000)])
pars_pysr = pars[:]
X_pysr = np.zeros((y_pysr.shape[0], 3 + 1))  #for the three cosmo parameters plus ells


# In[6]:


# Reformatting data
for i in range(100):
    X_pysr[n_ells*i:n_ells*(i+1), :3] = np.tile(pars[i], n_ells).reshape(n_ells, -1)
    X_pysr[n_ells*i:n_ells*(i+1), -1] = X_ells   #final column is ells

# make x3 = ombh2 + omch2 = om0h2
X_pysr[:, 2] = X_pysr[:, 1] + X_pysr[:, 2]  


# In[7]:


# Template Function

template = TemplateExpressionSpec(
    expressions = ["g"],
    variable_names = ["x1", "x2", "x3", "x4"],  #H0, ombh2, ombh2+omch2, ells
    parameters = {"beta": 4},  #parameters to vary in the model to create equation - index from 1
    combine = "1 + (beta[1]*(x4/beta[2])^(g(x2, x3)) - 1)*(1 + exp(-(x4-beta[3])/(100+400*Float32(0.7)^abs(beta[4]))))^-1"   #find equation of this form
)

# PySR Model

model = PySRRegressor(
    niterations = 100,
    binary_operators = ["+", "-", "*", "pow"],  #allowed operations
    constraints = {'pow': (4, 1), "*": (4, 4)},   #enforces maximum complexities on arguments of operators 
    batching = True, 
    batch_size = 10000, 
    maxsize = 30,
    populations = 20,
    expression_spec = template,
    complexity_of_variables = 2, #global complexity of variables
    procs = 4
)


# In[ ]:


# Train

model.fit(X_pysr, y_pysr)