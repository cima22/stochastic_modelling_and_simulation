#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
# add here the libraries you need


# # Discrete Time Markov Chains - Part 2
# This is an exercise notebook on DTMCs. 
# 
# Remember to revise of the lecture on DTMC simulation before attempting to solve it!
# In order to complete this notebook, you need the models implemented in Part 1 notebook on DTMC.

# ### 1. Simulation of DTMC
# Write a method that simulates a DTMC for `n` steps, where `n` is a parameter of the method, and returns the whole trajectory as output.

# In[ ]:





# ### 2. Statistical analysis
# Write methods for:
# - 2.1. computing the average of a function `f` of the state space, at time step `n`.
# - 2.2. computing the probability of reaching a target region `A` of the state space by time step `n`.
# 
# Both methods should use simulation, and return an estimate and a confidence interval at a specified confidence level `alpha` (0.95% by default).

# In[ ]:





# In[ ]:





# ### 3. Branching chain
# Consider a population, in which each individual at each
# generation independently gives birth to $k$ individuals with
# probability $p_k$. These will be the members of the next
# generation. Assume $k\in\{-1, 0,1,2\}$. The population is initial compused of two individuals Adam and Eve.

# In[ ]:





# Assume now that $p_0 = p_1 = p_2 = (1-p_{-1})/3$. Estimate the average and the confidence interval of the probability of the population to become extinct for increasing values of $p_{-1}$.

# In[ ]:




