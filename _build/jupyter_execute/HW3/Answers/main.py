#!/usr/bin/env python
# coding: utf-8

# # Homework #3 of Fuzzy Systems
# **Student name:** Seyed Mohammad Amin Dadgar <br>
# **Student Id:** 4003624016 <br>
# **Instructor:** Dr. Hossein Karshenas <br>
# 
# **Date:** June, 2022 | Khordad, 1401

# The questions are from the excersises of the books Fuzzy Set Theory — and Its Applications by H.J. Zimmerman $4^{th}$ edition $2001$ and fuzzy logic with engineering applications $4^{th}$ edition $2017$ by Dr Ross. We've specified the question numbers and the answers of it in this file.

# <center>
# <h1>Table of contents</h1>
#     
# <h2> Fuzzy Inference Systems and Fuzzy Controller </h2>
# <a href=#Q1>Question 1</a> <br>
# <a href=#Q2>Question 2</a> <br>
# <a href=#Q3>Question 3</a> <br>
# <a href=#Q4>Question 4</a> <br>
# <a href=#Q5>Question 5</a> <br>
# <a href=#Q6>Question 6</a> <br>
# 
# <h2> Data-Driven Modeling</h2>
# <a href=#Q7>Question 7</a> <br>
# <a href=#Q8>Question 8</a> <br>
# 
# <h2>Possibility Distribution</h2>
# <a href=#Q9>Question 9</a> <br>
# <a href=#Q10>Question 10</a> <br>
# <a href=#Q11>Question 11</a> <br>
# <a href=#Q12>Question 12</a> <br>
#     
# <h2>Extended operations on fuzzy numbers</h2>
# <a href=#Q13>Question 13</a> <br>
# <a href=#Q14>Question 14</a> <br>
# 
# <h2>Fuzzy Clustering</h2>
# <a href=#Q15>Question 15</a> <br>
# <a href=#Q16>Question 16</a> <br>
# <a href=#Q17>Question 17</a> <br>
# <a href=#Q18>Question 18</a> <br>
# <a href=#Q19>Question 19</a> <br>
# </center>
# 

# Libraries Used:
# - matplotlib: This library is used for visualizations.
# - numpy: Numpy library helped us with arrays and mathematical calculations.
# - pandas: Used For creating a beautiful represetive of datas.

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ## Fuzzy Inference Systems and Fuzzy Controller

# ### Q1
# Answer to the question 6 of chapter 9 of the Zimmerman book.

# Let the universe $X=\{1, 2, 3, 4, 5\}$ and "small integers" represented as a fuzzy set $\tilde{A}=\{(1, 1), (2, .5), (3, .4), (4, .2)\}.$
# 
# Having the Fuzzy relation "Almost Equal" as below
# ![image.png](attachment:image.png)
# Find the fuzzy set "rather small integers" using the composition $\tilde{B}=\tilde{A} o \tilde{R}$

# ##### Answer
# If we assume the composition as a `max-min` operation, we can find the answer of the $\tilde{A} o \tilde{R}$ as below
# \begin{equation}
# \tilde{B} = \tilde{A} o \tilde{R} = \{(1, 1), (2, .8), (3, .5), (4, .4)\}
# \end{equation}

# ### Q2
# Answer to the question 12 of chapter 5 in the Ross book.

# Considering the universe set X=[0,9]. 
# 
# Having $\tilde{A}={(0, 0.4), (1, 0.7), (2, 0.2), (3, 0.1), (4, 0.3)}$ and $\tilde{B}={(5, 0), (6, 0.5), (7, 0.6), (8, 0.3), (9, 0.7)}$.
# 
# - Part b. Construct the rule **$\tilde{R}$ = If $\tilde{A}$ then $\tilde{B}$** using Mamdani implication.
# - Part c. having the new antecedent $\tilde{A} = \{(0, .3), (1, .7), (2, .6), (3, .3), (4, .5) \}$ find the new consequent $\tilde{B}^{'}$.

# ##### Answer
# We can find the $\mu_{\tilde{R}} = min\{\mu_{\tilde{A}}, \mu_{\tilde{B}}\}$ as a matrix below
# ![image.png](attachment:image.png)

# And to find the $\tilde{B}^{'}$, we can find the answer using the Mamdani implication rule(`max-min` operation will be applied on $\tilde{R}$). We will write the computations for one of the values `0`, and the others are the same and we will calculate it without writing (using mind).
# 
# Calculation:
# \begin{equation}
# \mu_{\tilde{B}^{'}}(0) = max\{min\{0.3, 0\}, min\{0.7, 0\}, min\{0.3, 0\}, min\{0.3, 0\}, min\{0.5, 0\}\} = max\{0,0 , 0, 0, 0\} = 0
# \end{equation}
# So to answer for all values of $X$ is as below
# \begin{equation}
# \tilde{B}^{'} = \{(5, 0), (6, 0.5), (7, 0.6), (8, 0.3), (9, 0.7)\}
# \end{equation}

# ### Q3
# Answer to the question 15 of chapter 5, using the Ross book. 
# 
# \begin{equation}
# \mu_{\tilde{H}}(x) = \{(10, .7), (20, .6), (30, .3), (40, .4), (50, .2)\} \\
# \mu_{\tilde{D}}(y) = \{(5*10^4, .8), (10*10^4, .5), (15*10^4, .6), (20*10^4, .4), (15*10^4, .1)\}
# \end{equation}

# - Part a.Using the classical implication `min` operator find the asnwer to the rule **If high water level (H) THEN more discharge (D)**
# ![image.png](attachment:image.png)
# 
# - Part b. Using the `max-min` composition find the discharge associated with this level of water $\mu_{\tilde{H}}^{'} = \{(10, .7),(20, .3), (30, .5), (40, .6), (50, .4)\}$.
# 
# The answer is 
# \begin{equation}
# \mu_{\tilde{D}}^{'} = \{(5*10^4, 0.7), (10*10^4, 0.5), (15*10^4, 0.6), (20*10^4, 0.4), (25*10^4, 0.1)\}
# \end{equation}

# ### Q4
# Answer to the question 26 of chapter 5, using the Ross book.

# Having the values below as input, Apply Mamdani and a Sugeno inference using the membership functions and the rules.
# 
# **Inputs**
# \begin{equation}
# Re = 1.965 \times 10^4 \\
# Pr = 275
# \end{equation}
# **Rules**
# \begin{equation}
# A. \text{If Re is high and Pr is low Then Nu is low.} \\
# B. \text{If Re is low and Pr is low Then Nu is low.} \\
# C. \text{If Re is high and Pr is high Then Nu is medium.} \\
# D. \text{If Re is low and Pr is high Then Nu is medium.}
# \end{equation}
# **Membership Functions**
# ![image.png](attachment:image.png)

# Computing the `Low` membership value for *Re* gaves us $\mu_{RE, Low}(1.965 \times 10^4) = 0.25$ <br>
# Computing the `High` membership value for *Re* gaves us $\mu_{RE, High}(1.965 \times 10^4) = 0.25$ <br>
# 
# Computing the `Low` membership value for *Pr* gaves us $\mu_{Pr, Low}(275) = 0.75$ <br>
# Computing the `High` membership value for *Pr* gaves us $\mu_{Pr, High}(275) = 0.2$ <br>
# 
# Note: the values for each one is near accurate, because we're finding the membership values by look.

# Using the values above we can Find the membership values to each rule
# \begin{equation}
# \text{Rule A.} \mu_{Nu, Low} = min\{0.25, 0.75\} = 0.25 \\
# \text{Rule B.} \mu_{Nu, Low} = min\{0.25, 0.75\} = 0.25 \\
# \text{Rule C.} \mu_{Nu, Medium} = min\{0.25, 0.2\} = 0.2 \\
# \text{Rule D.} \mu_{Nu, Medium} = min\{0.25, 0.2\} = 0.2
# \end{equation}
# And the membership values for $\mu_{Nu, big}$ is zero.

# So the output of Mamdani equation is as below

# In[2]:


x1 = np.linspace(375, 575, 201).astype(float)
x2 = np.linspace(525, 725, 201).astype(float)


y1 = np.full((201), 0.25)
y2 = np.full((201), 0.25)

## note: values are near accurate
y1 = np.where(x1 <= 400, (1/100) * (x1 - 375), y1)
y1 = np.where(x1 >= 550, -(1/100) * (x1 - 575), y1)

y2 = np.where(x2 <= 525, (1/100) * (x2 - 525), y2)
y2 = np.where(x2 >= 700, -(1/100) * (x2 - 725), y2)


plt.fill(x1, y1, alpha=0.8)
plt.fill(x2, y2, alpha = 0.8)
plt.legend(['$Nu = Medium$', '$Nu = Low$'])
plt.show()


# The sugeno inference would gives us the equation below for the result of inference
# 
# \begin{equation}
# \frac{0.25(475) + 0.2 (475) + 0.2(625) + 0.2(625)}{0.25 + 0.2 + 0.2 + 0.2} = \frac{463.75}{0.85} = 545.59
# \end{equation}

# ### Q5
# Answer to the question 3 of chapter 11, using the Zimmerman book.

# Using the rule base system below, (a) Apply Mamdani implication on `error=2` and `change of error = 4`. (b) Then using two methods `MOM` and `COG`.
# ![image.png](attachment:image.png)

# **Answer**
# 
# First we will try to plot each fuzzy rule membership values.
# 
# Error membership function is as below

# In[3]:


X_rules = np.linspace(-8, 8, 161)
## the plot for `error` is as below
## constructing a plot for each granule

## for negative, we have `1` and the equation `-1/2 (x+1)`
negative_error = -0.5 * (X_rules+1)
negative_error = np.where(X_rules < -3, 1, negative_error)
negative_error = np.where(negative_error < 0, 0, negative_error)

## for zero granule, we have the equations `-1/3 (x-3)` for right expansion
## And `-1/3 (x+3)` for the left expansion

## right expansion
zero_error = (-1/3) * (X_rules-3)
## left expansion
zero_error = np.where(X_rules < 0, (1/3)* (X_rules+3), zero_error)
## zero ing for other values
zero_error = np.where((X_rules < -3) | (X_rules > 3), 0, zero_error)

## for positive granule, the left expansion equation is `1/2 (x-1)`
## And the right expansion equation is `1` 

positive_error = 0.5 * (X_rules-1)
positive_error = np.where(X_rules > 3, 1, positive_error)
positive_error = np.where(X_rules < 1, 0, positive_error)

plt.plot(X_rules, negative_error)
plt.plot(X_rules, zero_error)
plt.plot(X_rules, positive_error)
plt.xticks(range(-8, 9))
plt.ylabel('$\mu(error)$')
plt.xlabel('error')

plt.text(x=-8, y=0.5, s="Error Memeberhip", bbox=dict(facecolor='yellow', alpha=0.5))
plt.legend(['negative', 'zero', 'positive'])
plt.show()


# Change of error is

# In[4]:


X_rules = np.linspace(-8, 8, 161)
## the plot for `error` is as below
## constructing a plot for each granule

## for negative, we have `1` and the equation `-1/4 (x+1)`
negative_change_error = -0.25 * (X_rules+1)
negative_change_error = np.where(X_rules < -5, 1, negative_change_error)
negative_change_error = np.where(negative_change_error < 0, 0, negative_change_error)

## for zero granule, we have trapezoidal
## the equations: 
##   `(x-3)` for right expansion
##   `1` for the middle (range(-2, 2))
##   `-(x+3)` for the left expansion

## right expansion
zero_change_error = -(X_rules-3)
## left expansion
zero_change_error = np.where(X_rules < -2, (X_rules+3), zero_change_error)
## zero ing for other values
zero_change_error = np.where((X_rules < -3) | (X_rules > 3), 0, zero_change_error)
## the middle values
zero_change_error = np.where((X_rules >= -2) & (X_rules <= 2), 1, zero_change_error)


## for positive granule, the left expansion equation is `1/4 (x-1)`
## And the right expansion equation is `1` 

positive_change_error = 0.25 * (X_rules-1)
positive_change_error = np.where(X_rules > 5, 1, positive_change_error)
positive_change_error = np.where(X_rules < 1, 0, positive_change_error)

plt.plot(X_rules, negative_change_error)
plt.plot(X_rules, zero_change_error)
plt.plot(X_rules, positive_change_error)
plt.xticks(range(-8, 9))
plt.ylabel('$\mu(\Delta(error))$')
plt.xlabel('$\Delta \; error$')

plt.text(x=-8, y=0.5, s="$\Delta$Error\n Memeberhip", bbox=dict(facecolor='yellow', alpha=0.5))


plt.legend(['negative', 'zero', 'positive'])
plt.show()


# And the control Action rule is

# In[5]:


X_action = np.linspace(0, 15, 151)


## equations for small action is
##   `1/2 (x-1)` for x in range(1, 3)
##   `-1/2 (x-5)` for x in range(3, 5)
##   `0` otherwise

small_action = 0.5* (X_action-1)
small_action = np.where((X_action > 3) & (X_action < 5), -0.5 * (X_action-5), small_action)
small_action = np.where((X_action <= 1) | (X_action >= 5), 0, small_action)

## equations for medium action is
##  `1/2 (x-4)` for x in range(4,6)
##  `-1/2 (x-10)` for x in range(8,10)
##  `1` for x range(6,8)
##  `0` otherwise

medium_action = 0.5 * (X_action-4)
medium_action = np.where((X_action > 8) & (X_action < 10), -0.5*(X_action-10), medium_action)
medium_action = np.where((X_action > 6) & (X_action <= 8), 1, medium_action)
medium_action = np.where((X_action <= 4) | (X_action >= 10), 0, medium_action)

## And equations for big action is
##   `1/3 (x-7)` for x in range(7, 10)
##   `-1/2 (x-12)` for x in range(10, 12)
##   `0` otherwise
big_action = (1/3) * (X_action - 7)
big_action = np.where((X_action > 10) & (X_action < 12), -0.5*(X_action-12), big_action)
big_action = np.where((X_action <= 7) | (X_action >= 12), 0, big_action)

plt.plot(X_action, small_action)
plt.plot(X_action, medium_action)
plt.plot(X_action, big_action)
plt.xticks(range(0, 16))
plt.ylabel('$\mu(control action)$')
plt.xlabel('control action')

plt.text(x=12, y=0.5, s="Control Action\n membership", bbox=dict(facecolor='yellow', alpha=0.5))


plt.legend(['small', 'medium', 'big'])
plt.show()


# Until now we have plotted the membership values, so now we will dive into the answers. 
# 
# Find the fuzzy set of control while Having `error = 2` and `change of error=4`. 

# In[6]:


## Creating the usable format of the actions, error and change of error 
SMALL_ACTION = np.stack((X_action, small_action), axis=1)
MEDIUM_ACTION = np.stack((X_action, medium_action), axis=1)
BIG_ACTION = np.stack((X_action, big_action), axis=1)

NEGATIVE_Error = np.stack((X_rules, negative_error), axis=1)
ZERO_Error = np.stack((X_rules, zero_error), axis=1)
POSITIVE_Error = np.stack((X_rules, positive_error), axis=1)

NEGATIVE_CHANGE_Error = np.stack((X_rules, negative_change_error), axis=1)
ZERO_CHANGE_Error = np.stack((X_rules, zero_change_error), axis=1)
POSITIVE_CHANGE_Error = np.stack((X_rules, positive_change_error), axis=1)


# In[7]:


## Find the values 
def find_membership_value(fuzzy_set, x):
    """
    Find the fuzzy membership value of a fuzzy set using the input `x`
    
    Parameters:
    ------------
    fuzzy_set : array of 2D tuple
        the fuzzy set we need to use
    x : float or integer
        using the x value find the fuzzy membership value 
        
    Returns:
    ---------
    value : floating between 0 and 1
        the normalized membership value of the fuzzy set
        Note: the fuzzy_set in the input must be normalized
    """
    value = fuzzy_set[:, 1][fuzzy_set[:, 0] == x]
    
    return value

error = 2
error_change = 4

negative_error_membership_value = find_membership_value(NEGATIVE_Error, error)
zero_error_membership_value = find_membership_value(ZERO_Error, error)
positive_error_membership_value = find_membership_value(POSITIVE_Error, error)

negative_change_error_membership_value = find_membership_value(NEGATIVE_CHANGE_Error, error_change)
zero_change_error_membership_value = find_membership_value(ZERO_CHANGE_Error, error_change)
positive_change_error_membership_value = find_membership_value(POSITIVE_CHANGE_Error, error_change)


# **Checking Rules
# Using Mamdani implication**
# 
# - if error is negative and $\Delta$ Error is negative $\rightarrow$ Then action is big

# In[8]:


## Find the values of actions using the rule table

## If delta error is negative and error is negative then the action is big
## Let's see the big action fuzzy value
values_tuple =(negative_change_error_membership_value, negative_error_membership_value)

idx = np.argmin(values_tuple)
big_action_membership_value = values_tuple[idx]
big_action_membership_value


# So the fuzzy value is zero, so there is no need to have in mind the rule above
# 
# - if error is zero and $\Delta$ Error is negative $\rightarrow$ Then action is big

# In[9]:


values_tuple =(zero_error_membership_value, zero_change_error_membership_value)

idx = np.argmin(values_tuple)
big_action_membership_value = values_tuple[idx]
big_action_membership_value


# Again in this rule the action for big have zero fuzzy value, Let's see others.
# 
# - if error is zero and $\Delta$ Error is zero $\rightarrow$ Then action is medium

# In[10]:


values_tuple =(zero_error_membership_value, zero_change_error_membership_value)

idx = np.argmin(values_tuple)
medium_action_membership_value = values_tuple[idx]
medium_action_membership_value


# - if error is zero and $\Delta$ Error is positive $\rightarrow$ Then action is medium

# In[11]:


values_tuple =(zero_error_membership_value, positive_change_error_membership_value)

idx = np.argmin(values_tuple)
medium_action_membership_value = values_tuple[idx]
medium_action_membership_value


# ✅ So the medium action fuzzy value can be extracted using the last rule.

# - if error is positive and $\Delta$ Error is zero $\rightarrow$ Then action is small

# In[12]:


values_tuple =(positive_error_membership_value, zero_change_error_membership_value)

idx = np.argmin(values_tuple)
small_action_membership_value = values_tuple[idx]
small_action_membership_value


# - if error is positive and $\Delta$ Error is positive $\rightarrow$ Then action is small

# In[13]:


values_tuple =(positive_error_membership_value, positive_change_error_membership_value)

idx = np.argmin(values_tuple)
small_action_membership_value = values_tuple[idx]
small_action_membership_value


# ✅ And we found  `0.5` membership value for small action using the last rule.

# So until now what we have is
# - if *error is zero* and *$\Delta$ Error is positive* Then *Medium action* with confidence = 0.333
# - If *error is positive* and *change of error is positive* Then *Small action* with confidence = 0.5

# In[14]:


X_MEDIUM = MEDIUM_ACTION[:, 0][MEDIUM_ACTION[:, 1] <= 0.333]
Y_MEDIUM = MEDIUM_ACTION[:, 1][MEDIUM_ACTION[:, 1] <= 0.333]
X_Small = SMALL_ACTION[:, 0][SMALL_ACTION[:,1] <= 0.5]
Y_Small = SMALL_ACTION[:, 1][SMALL_ACTION[:,1] <= 0.5]

plt.fill(X_MEDIUM, Y_MEDIUM)
plt.fill(X_Small, Y_Small)
plt.legend(['Action Medium', 'Action Small'])
plt.text(x=10, y=0.4, s='fuzzy inference', bbox=dict(facecolor='yellow',alpha = 0.5))
plt.show()


# In[15]:


## The maximums are in small action
U = SMALL_ACTION[:, 0][np.isclose(SMALL_ACTION[:, 1], np.max(Y_Small))]

## Mean of Maxima
Z_star_MOM = (U[0] + U[1]) / 2
Z_star_MOM


# To find Center Of gravity, we would like to use the equation below
# \begin{equation}
# Z^{*}_{COG} = \frac{\sum \mu_x x}{\sum \mu_x}
# \end{equation}

# In[16]:


## To find Center of Gravity
## initialize

## divide equation in two parts 
## above division and below
above = np.sum(X_Small * Y_Small) + np.sum(X_MEDIUM * Y_MEDIUM)
below = np.sum(Y_Small) + np.sum(Y_MEDIUM)

Z_star_COG = above / below
Z_star_COG


# In[17]:


## Plotting the Mamdani inference Result
plt.fill(X_MEDIUM, Y_MEDIUM)
plt.fill(X_Small, Y_Small)

## plotting the Mean of Maxima
plt.scatter(Z_star_MOM, np.max(Y_Small) / 2)
plt.plot(np.ones(10) * Z_star_MOM, np.linspace(0, np.max(Y_Small), 10), '--')
plt.text(x=Z_star_MOM - 2, y=0.180, s='MOM: $Z^*$', bbox=dict(facecolor='yellow', alpha=0.5))

## plotting the COG
plt.plot(np.ones(10) * Z_star_COG, np.linspace(0, 0.4, 10), 'g--')
plt.text(x=Z_star_COG -1, y=0.1, s='COG: $Z^*$', bbox=dict(facecolor='yellow', alpha=0.5))


## Other options in plot
plt.legend(['Action Medium', 'Action Small'])
plt.text(x=10, y=0.3, s='fuzzy Mamdani \ninference', bbox=dict(facecolor='yellow',alpha = 0.5))
plt.show()


# ### Q6

# Answer to the question 9 of chapter 4 using Ross Book.

# To answer this question we first draw the fuzzified outputs and then go deep into the answer.

# In[18]:


## plotting (a) Figure P4.9
x = np.linspace(0, 10, 101)

## equations for plotting (a) is as below
##  `1/4 (x)` for x in [0, 2]
##  `0.5` for x in [2, 3]
##  `-1/2 (x-4)` for x in [3, 4]
##  `0` otherwise
mu_A = 0.25 * x
mu_A = np.where((x > 2) & (x <= 3), 0.5, mu_A)
mu_A = np.where((x >3) & (x<4), -0.5 * (x-4), mu_A)
mu_A = np.where((x <= 0) | (x >= 4), 0, mu_A)

## equations for plotting (b) is
##   `0.35 (x-2)` for x in [2, 4]
##   `-0.7 (x-5)` for x in [4, 5]
##   `0` otherwise
mu_B = 0.35 * (x-2)
mu_B = np.where((x >= 4) & (x <=5), -0.7*(x-5), mu_B)
mu_B = np.where((x > 5) | (x < 2), 0, mu_B)

## equations for plotting (c) is
##   `(x-4)` for x in [4, 5]
##   `1` for x in [5, 7]
##   `-1/2 (x-9)` for x in [7, 9]
##   `0` otherwise
mu_C = x-4
mu_C = np.where((x >= 5) & (x <= 7), 1, mu_C)
mu_C = np.where((x > 7) & (x <= 9), -0.5 * (x-9) , mu_C)
mu_C = np.where((x > 9) | (x <= 4), 0 , mu_C)


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(x, mu_A)
axes[0].set_xticks(range(0, 11))
axes[0].set_ylabel('$\mu$')
axes[0].text(x=5, y=0.4, s='$\mu_{a}$', bbox=dict(facecolor='yellow', alpha=0.5), fontsize=18)

axes[1].plot(x, mu_B)
axes[1].set_xticks(range(0, 11))
axes[1].set_ylabel('$\mu$')
axes[1].text(x=6, y=0.4, s='$\mu_{b}$', bbox=dict(facecolor='yellow', alpha=0.5), fontsize=18)

axes[2].plot(x, mu_C)
axes[2].set_xticks(range(0, 11))
axes[2].set_ylabel('$\mu$')
axes[2].text(x=1, y=0.4, s='$\mu_{c}$', bbox=dict(facecolor='yellow', alpha=0.5), fontsize=18)

plt.show()


# Now we saw the memebership functions results. Let's plot them together and then have a defuzzification process using `Weighted Average`, `Centeroid (COG)`, `Mean of Maxima (MOM)`, `max membership`, `Left of Maxima (LOM)` and `Right of Maxima (ROM)` methods.

# In[19]:


## The union of the results is as below
plt.fill(x, mu_A)
plt.fill(x, mu_B)
plt.fill(x, mu_C)

plt.legend(['$\mu_A$', '$\mu_B$', '$\mu_C$'])

plt.show()


# In[20]:


mu_union = np.maximum(np.maximum(mu_A, mu_B), mu_C)
plt.fill(x, mu_union)
plt.legend(['$\mu_A \cup \mu_B \cup \mu_C$'])
plt.show()


# In[21]:


fuzzy_union_set = np.stack((x, mu_union), axis=1)


# In[22]:


## the maximum defuzzification method
z_star_max = fuzzy_union_set[
    np.isclose(fuzzy_union_set[:, 1], np.max(fuzzy_union_set[:, 1]))
               ]
z_star_max.shape


# In this case we're having more than one value in defuzzification process, So we can say that there would be no appropriate result in defuzzification of flat fuzzy numbers.

# In[23]:


## mean of maxima defuzzification process
avg = np.average(z_star_max[:, 0])
z_star_mom = z_star_max[np.isclose(z_star_max[:, 0], avg)]

z_star_mom


# In[24]:


## weighted average defuzzification process

## first we create the fuzzy sets of membership values
fuzzy_mu_A = np.stack((x, mu_A), axis=1)
fuzzy_mu_B = np.stack((x, mu_B), axis=1)
fuzzy_mu_C = np.stack((x, mu_C), axis=1)

average_A = np.average(fuzzy_mu_A[:, 0][fuzzy_mu_A[:, 1] > 0])
average_B = np.average(fuzzy_mu_B[:, 0][fuzzy_mu_B[:, 1] > 0])
average_C = np.average(fuzzy_mu_C[:, 0][fuzzy_mu_C[:, 1] > 0])

print(f'Averages A,B,C: {average_A}, {average_B}, {average_C}')

## to average
average_mu_A = fuzzy_mu_A[:, 1][np.isclose(fuzzy_mu_A[:, 0], average_A)]
average_mu_B = fuzzy_mu_A[:, 1][np.isclose(fuzzy_mu_A[:, 0], average_B)]
average_mu_C = fuzzy_mu_A[:, 1][np.isclose(fuzzy_mu_A[:, 0], average_C)]

## devide the weigted average equation in two parts
## above division and below of the division
above = (average_mu_A * average_A) + (average_mu_B * average_B) + (average_mu_C * average_C)
below = average_mu_A + average_mu_B + average_mu_C

z_star_weigted_avg = above / below
z_star_weigted_avg


# In[25]:


## Center of gravity defuzzification process

z_star_COG = np.sum(np.multiply(x, mu_union)) / np.sum(mu_union)
z_star_COG


# In[26]:


## it's easy to find left of maxima and right of maxima

z_star_LOM = z_star_max[0,0]
z_star_ROM = z_star_max[-1, 0]

z_star_LOM, z_star_ROM


# We've now found result values using 6 defuzzification process. Let's plot them:

# In[27]:


## first we created the union fuzzy set
union_fuzzy_set = np.stack((x, mu_union), axis=1)


# In[28]:


def find_fuzzy_value(fuzzy_set, x):
    """
    Find fuzzy membership value of x
    
    Parameters:
    ------------
    fuzzy_set : array of 2D tupple
        The fuzzy set we have
    x : float
    
    Returns:
    ---------
    fuzzy_value : float
        a floating point represent the fuzzy membership value
    """
    fuzzy_value = fuzzy_set[:, 1][np.isclose(fuzzy_set[:, 0], np.round(x, 1))]
    
    return fuzzy_value


plt.figure(figsize=(15, 8))
z_star_COG_fuzzy_value = find_fuzzy_value(union_fuzzy_set, z_star_COG)
plt.plot(np.ones(10) * z_star_COG, np.linspace(0, z_star_COG_fuzzy_value, 10) , 'b--')

z_star_LOM_fuzzy_value = find_fuzzy_value(union_fuzzy_set, z_star_LOM)
plt.plot(np.ones(10) * z_star_LOM, np.linspace(0, z_star_LOM_fuzzy_value, 10) , 'g--')

z_star_ROM_fuzzy_value = find_fuzzy_value(union_fuzzy_set, z_star_ROM)
plt.plot(np.ones(10) * z_star_ROM, np.linspace(0, z_star_ROM_fuzzy_value, 10) , 'r--')

z_star_MOM_fuzzy_value = find_fuzzy_value(union_fuzzy_set, z_star_mom[:,0])
plt.plot(np.ones(10) * z_star_mom[:,0], np.linspace(0, z_star_MOM_fuzzy_value, 10) , 'k--')

# z_star_Weighted_avg_fuzzy_value = find_fuzzy_value(union_fuzzy_set, z_star_weigted_avg)
# plt.plot(np.ones(10) * z_star_weigted_avg, np.linspace(0, z_star_Weighted_avg_fuzzy_value, 10) , 'o--')

plt.fill(x, mu_union, 'c')


plt.legend([
    '$Z^{*}_{COG}$',
    '$Z^{*}_{LOM}$',
    '$Z^{*}_{ROM}$',
    '$Z^{*}_{MOM}$',
#     '$Z^{*}_{Weighted \; avg}$',
    '$\mu_A \cup \mu_B \cup \mu_C$'
])

plt.text(x=-0.4,y=0.8, s='''The union of the fuzzy values is the filled shapes
                And the results of each defuzzification processes is the dotted lines''', bbox=dict(alpha=0.05))
plt.show()


# ## Data-Driven Modeling

# ### Q7
# Answer to question 5 of chapter 7 in Ross book

# We need to apply Clustering Method on the dataset, So we need to first implement it.

# In[29]:


## initilize the dataset
dataset = pd.DataFrame(data=[
    [6800, 6834, 1.23e-2],
    [6500, 6492, 1.02e-2],
    [6900, 6950, 1.34e-2],
    [6875, 6700, 1.31e-2]
    ], columns=['x1', 'x2', 'delta_l'],)
dataset


# In[30]:


parameters = pd.DataFrame(data=[
    [7000, 6500, 175, 175],
    [6850, 3200, 135, 175],
    [6500, 4500, 121, 175],
    [6000, 5000, 85, 175],
    [6200, 5000, 83.5, 175],
    [6315, 5800, 87, 175]
    ], columns=['c1', 'c2', 'sigma_3', 'sigma_4'])
parameters


# We use gaussian function as our membership function.

# In[31]:


def gaussian(X, Sigma, Mu):
    """
    The gaussian distribution
    
    Paremeters:
    ------------
    X : array_like
        the input range, can be an array of floating points
    Sigma : float
        the variance for gaussian distribution
    Mu : float
        the mean for gaussian distribution
        
    Returns:
    ---------
    Y : array_like
        same dimension as input `X`
        the values representing gaussian distribution
    """
    p1 = 1 / np.sqrt(2 * np.pi * (Sigma**2))
    p2 = np.exp(-np.power(X - Mu, 2) / (2*(Sigma**2)))

    Y = p1 * p2
    return Y
def CM_function(x, A, B, V, sigma):
    """
    The function that is used for Clustering method 
    with parameters `theta=(A, B, V, sigma)`
    the gaussian function will be used for each rule in this function
    
    Parameters:
    ------------
    x : array_like
        the input data
    A : 1D array
        the parameters for each membership function
    B : 1D array
        this parameter shows how many data is used for a rule
    V : array_like
        the mean values of data, `V.shape[1]` must match the data features count
    sigma : 1D array
        contains values for each rule standar deviation
        length must match the data features count
        
    Note: the length of `A`, `B` and `V` arrays must be the same
        
    Returns:
    ---------
    predicted : float
        the predicted value for an input x using the parameter theta
    """
    ## the results for above and below the division
    result_above = 0
    result_below = 0
    
    for i in range(len(A)):
        ## initilize the variable with one because we are going to multiply it
        gaussian_function_result = 1 
        for j in range(len(x)):
            gaussian_function_result = gaussian_function_result * gaussian(x[j], sigma[i][j], V[i][j])
        result_above += A[i] * gaussian_function_result
        result_below += B[i] * gaussian_function_result
        
    results = result_above / result_below
    return results


# In[32]:


Y_data = dataset[['delta_l']].values


# In[33]:


X_data = dataset[['x1', 'x2']].values

sigma1 = parameters[['sigma_3']].values
sigma2 = parameters[['sigma_4']].values

Sigma = np.array([sigma1, sigma2]).reshape((6, 2))
V = parameters[['c1', 'c2']].values

f_Y = []
for i in range(len(X_data)):
    y_predicted = CM_function(X_data[i], A= Y_data[i], B=[1], V=V, sigma=Sigma)
    f_Y.append(y_predicted)


# In[34]:


f_Y - Y_data.T


# We saw that using the rules given in the question we've found the outputs of the fuzzy system same as the output value $\delta_L$ and our model is working perfectly. Now Let's plot the rules

# In[35]:


X = np.linspace(2000, 10000, 80000) 

rules_count = 6

## initilize a value for y rules plot
Y_rule_X1 = []
Y_rule_X2 = []

for i in range(0, rules_count):
    Y = gaussian(X, Sigma[i,0], V[i, 0])
    Y_rule_X1.append(Y)
    
    Y = gaussian(X, Sigma[i,1], V[i, 1])
    Y_rule_X2.append(Y)


# In[36]:


plt.figure(figsize=(10, 5))

## plot mu_X1
for i in range(0, rules_count):
    plt.plot(X, Y_rule_X1[i])
plt.legend(['Rule 1', 'Rule 2','Rule 3','Rule 4','Rule 5','Rule 6',])
plt.xlim((5000, 8000))
plt.text(5100, 0.003, '$\mu_{x_1}$', fontsize=18)
plt.show()


# In[37]:


plt.figure(figsize=(10, 5))

## plot mu_X1
for i in range(0, rules_count):
    plt.plot(X, Y_rule_X2[i])
plt.legend(['Rule 1', 'Rule 2','Rule 3','Rule 4','Rule 5','Rule 6',])
plt.text(2000, 0.002, '$\mu_{x_2}$', fontsize=18)
plt.show()


# ### Q8
# Answer the question 9, Chapter 6 of Ross book.

# First we prepare the dataset used in this question.

# In[38]:


dataset = pd.DataFrame(dict(x=[0, 0.3,0.6, 1, 100], 
                        y=[1, 0.74, 0.55, 0.37, 0]))
dataset


# And the rules defined in the question are as below
# - If X is Large then Y is Zero
# - If X is Small then Y is Small

# In[39]:


## find the C values for the equation in the cell below
## we can apply C values manually but we represent this way as an alternative
C_min_x = dataset[['x']].min().values[0]
C_min_y = dataset[['y']].min().values[0]

C_max_x = dataset[['x']].max().values[0]
C_max_y = dataset[['y']].max().values[0]


# \begin{equation}
# C_i = C_{min} + \frac{b}{2^L - 1} (C_{max_i} - C_{min_i})
# \end{equation}

# In[40]:


## the implementation of equation above
def find_C_value(b, L, c_min, c_max):
    """
    Find the value of `C` parameter for the genetic algorithm
    
    Parameters:
    ------------
    b : float
        decimal value representive of the input value of the data
    L : positive integer
        value representing the length of each string in genetic algorithm
    c_min : float or integer
        decimal value representive for the smallest value that is fitted in dataset
    c_max : float or integer
        decimal value representive for the biggest value that is fitted in dataset
    
    Returns:
    ---------
    C : float
        the parameter representive for the value of b
    """
    
    C = c_min + (b / (2**L - 1)) * (c_max - c_min)
    
    return C


# In[41]:


def get_binary_representation_array(x, binary_width):
    """
    get the binary rerpresentation of an array
    
    Parameters:
    ------------
    x : array_like
        input decimal values
    binary_width : positive integer
        the value representive the count of the zeros and ones
        Important to set this with respect to `x` values
    
    Returns:
    ---------
    x_bin : string array
        binary string representation of decimal inputs 
    """
    
    ## find the binary representation of the population
    x_bin = []
    for i in range(len(x)):
        ## binary population
        binary_x = np.binary_repr(x[i], 
                                    width=binary_width)

        x_bin.append(binary_x)
    
    return x_bin


# In[42]:


## initialize the population using random binary value

maximum_random_value = 64
minimum_random_value = 0

## we need to specify four bases for our rules
bases_count = 4 

## the width of the population strings
population_width = int(np.ceil(np.log2(maximum_random_value)))

## fix the random seed 
np.random.seed(123)

population = np.random.randint(
    minimum_random_value,
    maximum_random_value,
    bases_count)
## transfer to binary representation
population_binary_representation = get_binary_representation_array(
    population,
    population_width
)

## convert to numpy for ease of use in later
population_binary_representation = np.array(
    population_binary_representation
)

population, population_binary_representation


# In[43]:


bases_x = find_C_value(population[:2], population_width, C_min_x, C_max_x )
bases_y = find_C_value(population[2:], population_width, C_min_y, C_max_y )

bases_x, bases_y


# Now we found some base values for our rules.
# 
# Let's apply to rules and see how it works

# In[44]:


def right_triangle_rules(start_value, base_width, slope):
    """
    Create right triangle membership functions for one rule
    
    Parameters:
    -----------
    start_value : float
        the value that we are going to start the plot
    base_width : float
        the width of triangle's base
    slope : -1 or +1
        the slope of the triangle's hypotenuse
    
    Returns:
    ----------
    x : 1D array_like
        the x values representive the x coordinates of the membership function
    y : 1D array_like
        the y values representive the membership values
        Note: the membership function values are normalized.
    """
    assert abs(slope) == 1, 'Error value for slope, can be either +1 or -1!'
    
    ## the x values start at `start_value` and ends at `start_value + base_width`
    end_value = start_value + base_width
    x = np.linspace(start_value, end_value, int(np.ceil(base_width)) * 100)
    
    ## finding y values via the slope
    if slope == -1:
        y = -(1/base_width) * (x - end_value)       
    else:
        y = (1/base_width) * (x - start_value)
    
    return x, y


# In[45]:


### Rules for X

plt.figure(figsize=(10, 8))
## rule1_X_x means the X data and x axis in Rule 1
## rule1_X_y means the X data and y axis in Rule 1
rule1_X_x, rule1_X_y = right_triangle_rules(C_min_x, bases_x[0], -1)
## rule2_X_x means the X data and x axis in Rule 2
## rule2_X_y means the X data and y axis in Rule 2
rule2_X_x, rule2_X_y = right_triangle_rules(C_max_x - bases_x[1], bases_x[1], 1)    

plt.plot(rule1_X_x, rule1_X_y)
plt.plot(rule2_X_x, rule2_X_y, color='g')
plt.vlines(rule1_X_x[0], 0, 1)
plt.vlines(rule2_X_x[-1], 0, 1, colors='g')
plt.xticks(rule1_X_x[::800])
plt.legend(['Rule 1. $X$ is Large ', 'Rule 2. $X$ is Small'])
plt.ylabel('$\mu_X$')
plt.xlabel('$X$')
plt.show()


# In[46]:


### Rules for Y

plt.figure(figsize=(10, 8))

## rule1_Y_x means x axis for Y data in Rule 1
## rule1_Y_y means y axis for Y data in Rule 1
rule1_Y_x, rule1_Y_y = right_triangle_rules(C_min_y, bases_y[0], -1)

## and as above naming is applied
rule2_Y_x, rule2_Y_y = right_triangle_rules(C_max_y - bases_y[1], bases_y[1], 1)    

plt.plot(rule1_Y_x, rule1_Y_y)
plt.plot(rule2_Y_x, rule2_Y_y, color='g')
plt.vlines(rule1_Y_x[0], 0, 1)
plt.vlines(rule2_Y_x[-1], 0, 1, colors='g')
plt.xticks(rule1_Y_x[::8])
plt.legend(['Rule 1. $Y$ is Zero', 'Rule 2. $Y$ is Small'], loc='upper center')
plt.ylabel('$\mu_Y$')
plt.xlabel('$Y$')
plt.show()


# In[47]:


## using rules we are going to find the output values
def predict_FIS(x_data, rule_X, rule_Y):
    """
    predict the output of the x using rule base fuzzy system
    Tsukamoto inference system is used
    
    Parameters:
    ------------
    x_data : array_like
        the input values of the dataset
    rule_X : 2D array
        first dimension data range and for second dimension fuzzy membership values
    rule_Y : 2D array
        output data
        first dimension data range and for second dimension fuzzy membership values
        
    Returns:
    --------
    y_predicted : array
        the float values representing the predicted output of x_data
    """
    
    ## the initialization of the predicted values array
    y_predicted = []
    
    for data in x_data:
        try:
            ## find the x fuzzy value for x input
            x_fuzzy_value = rule_X[:, 1][np.isclose(data, rule_X[:, 0], atol=1e-2)]

            ## if there was more than one value in x_fuzzy_value we know that they are close to each other
            ## so just we use one of it
            ## here we check if it was an array
            if np.ndim(x_fuzzy_value) > 0:
                x_fuzzy_value = x_fuzzy_value[0]
                
            ## check which y fuzzy value is close the x fuzzy value and return the original y value
            data_y_pred = rule_Y[:, 0][np.isclose(x_fuzzy_value, rule_Y[:, 1], atol=1e-2)]
            
            ## same as the if condition above
            if np.ndim(data_y_pred) > 0:
                data_y_pred = data_y_pred[0]
            
            ## append the predicted value
            y_predicted.append(data_y_pred)
        
        ## if the input x data didn't match the rules 
        ## then insert zero as predicted value
        except:
            y_predicted.append(0)
    
    return y_predicted


# In[48]:


##### First Rule. If `X` is Large then `Y` is Zero #####

rule_A1 = np.stack((rule1_X_x, rule1_X_y), axis=1)
rule_B1 = np.stack((rule1_Y_x, rule1_Y_y), axis=1)
y_predicted_rule1 = predict_FIS(dataset.x.values, rule_A1, rule_B1)

## square error
SE_Rule1 = np.sum(np.power(y_predicted_rule1 - dataset.y, 2))
print(f'First square error: {SE_Rule1}')


# In[49]:


##### Second Rule. If `X` is Small then `Y` is Small #####

rule_A2 = np.stack((rule2_X_x, rule2_X_y), axis=1)
rule_B2 = np.stack((rule2_Y_x, rule2_Y_y), axis=1)
y_predicted_rule2 = predict_FIS(dataset.x.values, rule_A2, rule_B2)

## print the square errors
SE_Rule2 = np.sum(np.power(y_predicted_rule2 - dataset.y, 2))

print(f'Second rule square error: {SE_Rule2}')


# The We will continue the first iteration to find the fitness values. For fitness values we will go through the equation below
# \begin{equation}
# f = A - \sum_i (y_i - y_i^{'})^2
# \end{equation}
# $A$ is one of our parameters and is chosen by user.
# 
# We would like to choose $A=100$

# In[50]:


def find_fitness_SE(SE, A):
    """
    find the fitness values using squared values
    
    Parameters:
    ------------
    SE : float
        the squared value
    A : integer
        parameter chosen by user
        
    Returns:
    --------
    f : float
        the fitness value
    """
    f = A - np.sum(SE)
    
    return f
def find_fitness(y_pred, y_actual, A):
    """
    find fitness using predicted values and actual y values
    
    Parameters:
    -----------
    y_pred : 1D array
        the predicted output values 
    y : 1D array
        the actual output values
    A : integer
        parameter chosen by user
    
    Returns:
    --------
    fitness : float
        the fitness value
    """
    squared_error = np.power(y_pred - y_actual, 2)
    fitness = find_fitness_SE(squared_error, A)
    
    return fitness


# In[51]:


f_rule1 = find_fitness_SE(SE_Rule1, 10)
f_rule2 = find_fitness_SE(SE_Rule2, 10)

f_rule1, f_rule2


# We will create a row of table same as table 6.10a in Ross book. 

# In[52]:


## using pandas to create a beatifule table
population_str = str(population_binary_representation)
pd.DataFrame(
    data={
        'String' : [population_str], 
        'Base 1 binary' : [population[0]],
        'Base 2 binary' :[ population[1]],
        'Base 3 binary' : [population[2]],
        'Base 4 binary' : [population[3]],
        'Base 1' : [bases_x[0]],
        'Base 2' : [bases_x[1]],
        'Base 3' : [bases_y[0]],
        'Base 4' : [bases_y[1]],
        f'Rule 1, y\' (x={dataset.iloc[0].x})': [y_predicted_rule1[0]],
        f'Rule 1, y\' (x={dataset.iloc[1].x})': [y_predicted_rule1[1]],
        f'Rule 1, y\' (x={dataset.iloc[2].x})': [y_predicted_rule1[2]],
        f'Rule 1, y\' (x={dataset.iloc[3].x})': [y_predicted_rule1[3]],
        f'Rule 2, y\' (x={dataset.iloc[0].x})': [y_predicted_rule2[0]],
        f'Rule 2, y\' (x={dataset.iloc[1].x})': [y_predicted_rule2[1]],
        f'Rule 2, y\' (x={dataset.iloc[2].x})': [y_predicted_rule2[2]],
        f'Rule 2, y\' (x={dataset.iloc[3].x})': [y_predicted_rule2[3]],
        'Rule 1 Fitness' : [f_rule1],
        'Rule 2 Fitness' : [f_rule2],        
    }
)


# Until now we've done genetic algorithm for 4 population and one iteration. Let's Make the code as a class and run everything inside it in order to use more population count and run more iterations.

# In[53]:


class genetic_fuzzy_tsukamoto():
    """
    apply genetic algorithm on a fuzzy dataset and rules
    And also use tsukamoto inference system in it
    
    This is not a general class and is to capsulate all the works in Question 8 in order to use it in more than one iteration easily
    """
    def __init__(self,
                 bases_count = 4,
                 max_population_value = 64,
                 min_population_value = 0,
                 population_count = 4,
                 random_seed= None):
        """
        initilize the class
        
        Parameters:
        -----------
        bases_count : positive integer
            the count for rules bases
            Since we have 2 rules and each has two values the default parameter is 4
        max_population_value : positive integer
            the maximum value that the population can have
            default_value is 64
        min_population_value : positive integer
            the minimum value that the population can have
            default_value is 0
        population_count : positive integer
            the value that can be set in population
        random_seed : positive integer
            to fix the random values can be reproduced
            default value is None, meaning values cannot be reproduced
        """
        
        self.bases_count = bases_count
        self.max_population_value = max_population_value
        self.min_population_value = min_population_value
        self.population_count = population_count
        self.random_seed = random_seed
        
        self.population_binary_representation = None
        self.population = None
        
        
        
        ## two rules will be applied for this question
        ## `rule_A1` and `rlue_B1` are representive of the first rule (A1 then B1)
        self.rule_A1 = None
        self.rule_B1 = None
        
        ## `rule_A2` and `rlue_B2` are representive of the second rule (A2 then B2)
        self.rule_A2 = None
        self.rule_B2 = None
        
    
    def fit(self, X, Y, population=None):
        """
        fit the parameters of the rules in one iteration 
        
        Parameters:
        -----------
        X : 1D array_like
            input data with one feature 
        Y : 1D array_like
            the output value for each X
        population : array
            fit the data on a predefined population
            default value is None meaning we are producing a random population
        """
        ## if there was no population
        if np.ndim(population) == 0:
            population = self.__init_population()
            self.population = population
        else:
            self.population = population
        
        ## the width of the population strings
        population_width = int(np.ceil(np.log2(self.max_population_value)))
        self.population_binary_representation = self.__get_binary_representation_array(
                population, population_width
            )
        
        ## we can apply C values manually but we represent this way as an alternative
        C_min_x = dataset[['x']].min().values[0]
        C_min_y = dataset[['y']].min().values[0]

        C_max_x = dataset[['x']].max().values[0]
        C_max_y = dataset[['y']].max().values[0]
        
        bases_x = self.__get_C_value(population[:2], population_width, C_min_x, C_max_x )
        bases_y = self.__get_C_value(population[2:], population_width, C_min_y, C_max_y )
        
        ## rule1_X_x means the X data and x axis in Rule 1
        ## rule1_X_y means the X data and y axis in Rule 1
        rule1_X_x, rule1_X_y = self.__right_triangle_rules(C_min_x, bases_x[0], -1)
        ## rule2_X_x means the X data and x axis in Rule 2
        ## rule2_X_y means the X data and y axis in Rule 2
        rule2_X_x, rule2_X_y = self.__right_triangle_rules(C_max_x - bases_x[1], bases_x[1], 1)   
        
        ## rule1_Y_x means x axis for Y data in Rule 1
        ## rule1_Y_y means y axis for Y data in Rule 1
        rule1_Y_x, rule1_Y_y = self.__right_triangle_rules(C_min_y, bases_y[0], -1)

        ## and as above naming is applied
        rule2_Y_x, rule2_Y_y = self.__right_triangle_rules(C_max_y - bases_y[1], bases_y[1], 1)
        
        ## First and second rules 
        rule_A1 = np.stack((rule1_X_x, rule1_X_y), axis=1)
        rule_B1 = np.stack((rule1_Y_x, rule1_Y_y), axis=1)
        
        rule_A2 = np.stack((rule2_X_x, rule2_X_y), axis=1)
        rule_B2 = np.stack((rule2_Y_x, rule2_Y_y), axis=1)
        
        ## update the rules
        self.rule_A1 = rule_A1
        self.rule_B1 = rule_B1
        
        self.rule_A2 = rule_A2
        self.rule_B2 = rule_B2
        
        self.bases_x = bases_x
        self.bases_y = bases_y
        
        
        
    
    def __init_population(self):
        """
        initialize population function
        
        Returns:
        --------
        population : 1D array_like
            array of decimal values between max_random_value and min_random_value
        """
        ## initialize the population using random binary value

        maximum_random_value = self.max_population_value
        minimum_random_value = self.min_population_value

        ## we need to specify four bases for our rules
        bases_count = self.bases_count

        ## if random seed was given 
        ## set it
        if self.random_seed != None:
            np.random.seed(self.random_seed)

        population = np.random.randint(
            minimum_random_value,
            maximum_random_value,
            bases_count)

        return population
    
    def __get_binary_representation_array(self, x, binary_width):
        """
        get the binary rerpresentation of an array

        Parameters:
        ------------
        x : array_like
            input decimal values
        binary_width : positive integer
            the value representive the count of the zeros and ones
            Important to set this with respect to `x` values

        Returns:
        ---------
        x_bin : string array
            binary string representation of decimal inputs 
        """

        ## find the binary representation of the population
        x_bin = []
        for i in range(len(x)):
            ## binary population
            binary_x = np.binary_repr(x[i], 
                                        width=binary_width)

            x_bin.append(binary_x)

        return x_bin
    
    def __get_C_value(self, b, L, c_min, c_max):
        """
        Find the value of `C` parameter for the genetic algorithm

        Parameters:
        ------------
        b : float
            decimal value representive of the input value of the data
        L : positive integer
            value representing the length of each string in genetic algorithm
        c_min : float or integer
            decimal value representive for the smallest value that is fitted in dataset
        c_max : float or integer
            decimal value representive for the biggest value that is fitted in dataset

        Returns:
        ---------
        C : float
            the parameter representive for the value of b
        """

        C = c_min + (b / (2**L - 1)) * (c_max - c_min)

        return C
    
    def __right_triangle_rules(self, start_value, base_width, slope):
        """
        Create right triangle membership functions for one rule

        Parameters:
        -----------
        start_value : float
            the value that we are going to start the plot
        base_width : float
            the width of triangle's base
        slope : -1 or +1
            the slope of the triangle's hypotenuse

        Returns:
        ----------
        x : 1D array_like
            the x values representive the x coordinates of the membership function
        y : 1D array_like
            the y values representive the membership values
            Note: the membership function values are normalized.
        """
        assert abs(slope) == 1, 'Error value for slope, can be either +1 or -1!'

        ## the x values start at `start_value` and ends at `start_value + base_width`
        end_value = start_value + base_width
        x = np.linspace(start_value, end_value, int(np.ceil(base_width)) * 100)

        ## finding y values via the slope
        if slope == -1:
            y = -(1/base_width) * (x - end_value)       
        else:
            y = (1/base_width) * (x - start_value)

        return x, y


    def predict(self, x_data):
        """
        predict the output of the x using rule base fuzzy system
        Tsukamoto inference system is used
        The results will be for both 2 rules

        Parameters:
        ------------
        x_data : array_like
            the input values of the dataset

        Returns:
        --------
        y1_predicted : array
            the float values representing the predicted output of x_data
            predicted values for rule number 1
        y2_predicted : array
            the float values representing the predicted output of x_data
            predicted values for rule number 2
        """

        assert np.ndim(self.rule_A1) != 0, "Error! First fit the model with data"
        
        ## predictions for both rules
        y1_predicted = self.__prediction(x_data, self.rule_A1, self.rule_B1)
        y2_predicted = self.__prediction(x_data, self.rule_A2, self.rule_B2)

            
        return y1_predicted, y2_predicted
    
    def __prediction(self, x_data, rule_X, rule_Y):
        """
        predict the output of the x using rule base fuzzy system
        Tsukamoto inference system is used

        Parameters:
        ------------
        x_data : array_like
            the input values of the dataset
        rule_X : 2D array
            first dimension data range and for second dimension fuzzy membership values
        rule_Y : 2D array
            output data
            first dimension data range and for second dimension fuzzy membership values

        Returns:
        --------
        y_predicted : array
            the float values representing the predicted output of x_data
        """
        ## the initialization of the predicted values array
        y_predicted = []
        
        for data in x_data:
            try:
                ## find the x fuzzy value for x input
                x_fuzzy_value = rule_X[:, 1][np.isclose(data, rule_X[:, 0], atol=1e-2)]

                ## if there was more than one value in x_fuzzy_value we know that they are close to each other
                ## so just we use one of it
                ## here we check if it was an array
                if np.ndim(x_fuzzy_value) > 0:
                    x_fuzzy_value = x_fuzzy_value[0]

                ## check which y fuzzy value is close the x fuzzy value and return the original y value
                data_y_pred = rule_Y[:, 0][np.isclose(x_fuzzy_value, rule_Y[:, 1], atol=1e-2)]

                ## same as the if condition above
                if np.ndim(data_y_pred) > 0:
                    data_y_pred = data_y_pred[0]

                ## append the predicted value
                y_predicted.append(data_y_pred)

            ## if the input x data didn't match the rules 
            ## then insert zero as predicted value
            except:
                y_predicted.append(0)
        return y_predicted
    


# In[54]:


GA_fuzzy = genetic_fuzzy_tsukamoto(random_seed=123)

GA_fuzzy.fit(dataset.x.values, dataset.y.values)

y_pred1, y_pred2 = GA_fuzzy.predict(dataset.x.values) 

rule1_fitness = find_fitness(y_pred1, dataset.y.values, 10)
rule2_fitness = find_fitness(y_pred2, dataset.y.values, 10)

rule1_fitness, rule2_fitness


# In[55]:


y_pred1


# In[56]:


def create_table(class_instance, dataset, y_pred1, y_pred2, rule1_fitness, rule2_fitness):
    """
    Create a table of the data 
    
    Parameters:
    ------------
    class_instance : class instance
        an instance of `genetic_fuzzy_tsukamoto` class representing the values
    dataset : pandas dataframe
        contains input dataset
    y_pred1 : 1D array
        predicted values of input dataset using rule1
    y_pred2 : 1D array
        predicted values of input dataset using rule2
        
    Returns:
    ---------
    table : pandas dataframe
        the table of data
    """
    ## using pandas to create a beatifule table
    population_str = str(population_binary_representation)
    table = pd.DataFrame(
        data={
            'String' : [str(class_instance.population_binary_representation)], 
            'Base 1 binary' : [class_instance.population[0]],
            'Base 2 binary' :[class_instance.population[1]],
            'Base 3 binary' : [class_instance.population[2]],
            'Base 4 binary' : [class_instance.population[3]],
            'Base 1' : [class_instance.bases_x[0]],
            'Base 2' : [class_instance.bases_x[1]],
            'Base 3' : [class_instance.bases_y[0]],
            'Base 4' : [class_instance.bases_y[1]],
            f'Rule 1, y\' (x={dataset.iloc[0].x})': [y_pred1[0]],
            f'Rule 1, y\' (x={dataset.iloc[1].x})': [y_pred1[1]],
            f'Rule 1, y\' (x={dataset.iloc[2].x})': [y_pred1[2]],
            f'Rule 1, y\' (x={dataset.iloc[3].x})': [y_pred1[3]],
            f'Rule 2, y\' (x={dataset.iloc[0].x})': [y_pred2[0]],
            f'Rule 2, y\' (x={dataset.iloc[1].x})': [y_pred2[1]],
            f'Rule 2, y\' (x={dataset.iloc[2].x})': [y_pred2[2]],
            f'Rule 2, y\' (x={dataset.iloc[3].x})': [y_pred2[3]],
            'Rule 1 Fitness' : [rule1_fitness],
            'Rule 2 Fitness' : [rule2_fitness],        
        }
    )
    
    return table


# In[57]:


table_row1 = create_table(GA_fuzzy, dataset, y_pred1, y_pred2, rule1_fitness, rule2_fitness)
table_row1


# In[58]:


class genetic_reproduction():
    def __init__(self, population):
        """
        reproduce samples class for genetic algorithm
        
        Parameters:
        ------------
        population : array of strings
            the population for now
        """
        
        self.population = population
    
    def reproduce(self, mutation_rate=0.1):
        """
        produce new population using the given population
        the production is using cross over and mutation 
        
        Parameters:
        -----------
        mutation_rate : float between 0 and 1
            this value represents that how many times the mutation is applied
            default value is 0.1 meaning for ten precent of the time mutation is applied
            
        Returns:
        --------
        new_population : array of string
            the returned population is as length as the given population as input
        """
        
        population = self.population
        ## initialize new population
        new_population = []
        
        population_count = len(population)
        
        i = 0
        ## while we didn't produce enough population continue the procedure
        while i < population_count:
            ## find a value to use mutation or cross over 
            choice = np.random.random()

            ## apply mutation if choice is less than 0.1
            if choice <= 0.1:
                produced_sample = self.__mutation(population[i])
                new_population.append(produced_sample)
                ## one new sample is produced so we increase the index with 1 value
                i += 1
            else:
                ## apply a random division point
                division_point = np.random.randint(1, len(population[0]))
                
                ## if there was just one sample available try mutation on one sample with itself
                if i+1 == population_count:
                    produced_samples = self.__cross_over(population[i], population[i], division_point )
                    ## just append one of the produced samples
                    new_population.append(produced_samples[0])
                else:
                    produced_samples = self.__cross_over(population[i], population[i+1], division_point)
                    new_population.append(produced_samples[0])
                    new_population.append(produced_samples[1])
                
                ## two new samples are produced so increase the index with 2 value
                i += 2
        
        return new_population 
                
        
    def __cross_over(self, sample1, sample2, division_point):
        """
        Apply the cross over method on two samples of population 

        Parameters:
        ------------
        sample1 : string 
            a sample from the population in genetic algorithm
            having binary representation for example: `111000`
        sample2 : string 
            a sample from the population in genetic algorithm
            having binary representation for example: `111000`  
        division_point : positive integer
            the point that the samples are splitted

        Returns:
        ---------
        new_sample1 : string
            the newly created sample using cross over
        new_sample2 : string
            the newly created sample using cross over
        """

        new_sample1 = sample1[:division_point] + sample2[division_point:]
        new_sample2 = sample1[division_point:] + sample2[:division_point]

        return new_sample1, new_sample2

    def __mutation(self, sample1):
        """
        Apply the mutation method on a samples of population to produce new samples 

        Parameters:
        ------------
        sample1 : string 
            a sample from the population in genetic algorithm
            having binary representation for example: `111000`

        Returns:
        ---------
        new_sample : string
            the newly created sample using cross over
        """
        ## the point that mutation will be applied
        mutation_point = np.random.randint(0, len(sample1))
        
        new_sample = ''
        new_sample += sample1[:mutation_point] + str(int(not int(sample1[mutation_point]))) + sample1[mutation_point+1:]

        return new_sample
    


# In[59]:


new_population_str = genetic_reproduction(GA_fuzzy.population_binary_representation).reproduce()


# In[60]:


def bin_to_decimal(binary_array):
    """
    convert binary string array values to a decimal array
    
    Parameters:
    ------------
    binary_array : 1D array_like
        an array that each value is representive of an binary string value
    
    Returns:
    ---------
    decimal_array : 1D array_like
        array of decimal values representive of each binary input values
    """
    decimal_array = []
    for binary_value in binary_array:
        decimal_array.append(int(binary_value, 2))
        
    return decimal_array


# In[61]:


new_population = bin_to_decimal(new_population_str)
new_population = np.array(new_population)
new_population


# In[62]:


GA_fuzzy1 = genetic_fuzzy_tsukamoto()
GA_fuzzy1.fit(dataset.x.values, dataset.y.values, population=np.array(new_population))

y_pred1, y_pred2 = GA_fuzzy1.predict(dataset.x.values) 

rule1_fitness = find_fitness(y_pred1, dataset.y.values, 10)
rule2_fitness = find_fitness(y_pred2, dataset.y.values, 10)

rule1_fitness, rule2_fitness


# In[63]:


table_row2 = create_table(GA_fuzzy1, dataset, y_pred1, y_pred2, rule1_fitness, rule2_fitness)


# In[64]:


pd.concat((table_row1, table_row2),ignore_index=True)


# Let's do all the works and return the iteration one results.

# In[65]:


def iterate_genetic_algorithm(new_population_str, iteration_count=4):
    """
    Iterate over all methods we've been created
    
    Parameters:
    -----------
    new_population_str : string array
        the initialized population
    iteration_count : positive integer
        
        
    Returns:
    --------
    results_table : pandas dataframe
        the results for all values
    """
    table_iteration = []

    for i in range(0, iteration_count):
        GA_fuzzy = genetic_fuzzy_tsukamoto()

        ## Reproduction phase
        ## the last population is passed to create new population
        new_population_str = genetic_reproduction(new_population_str).reproduce()
        new_population = bin_to_decimal(new_population_str)
        new_population = np.array(new_population)

        GA_fuzzy.fit(dataset.x.values, dataset.y.values, population=np.array(new_population))

        y_pred1, y_pred2 = GA_fuzzy.predict(dataset.x.values) 

        rule1_fitness = find_fitness(y_pred1, dataset.y.values, 10)
        rule2_fitness = find_fitness(y_pred2, dataset.y.values, 10)

        table = create_table(GA_fuzzy, dataset, y_pred1, y_pred2, rule1_fitness, rule2_fitness)   

        table_iteration.append(table)
    
    results_table = pd.concat(table_iteration, ignore_index=True)
    
    return results_table


# In[66]:


## iteration count
## Note all the values of random!!!
## Can be different in the next run
n = 10

print(f"Found values in {n} iterations")
iterate_genetic_algorithm(new_population_str, n)


# ## Possibility Distribution

# ### Q9
# Answer to the question 2 of chapter 8 Zimmerman book. 

# Define a probability distribution for "cars drive X mph on American freeways."
# 
# If we define the mean to $80$, meaning that the most cars in freeways are driven with the speed of the $80 \; mph$, We can define a gaussian function with $\mu=80$ for the probability and if the highest and lowest speeds are $100$ and $60$ miles per hour the $\sigma$ can be $20$. So the Probability can be defined as
# \begin{equation}
# P(x) = \frac{1}{\sqrt{2\pi}\sigma} exp\Big(-\frac{1}{2}\big(\frac{x-80}{20}\big)^2\Big)
# \end{equation}
# 
# And having the facts we've defined above for probability distribution we can define a triangular fuzzy number for the possiblity distribution.
# \begin{equation}
# \pi(x) = Triangular(80, 60, 100)
# \end{equation}
# Note that if we want to describe possiblity distribution with strong consistency principal we shouldn't use the triangular density function above, and instead we could defind the function below
# \begin{equation}
# \pi(x) = \begin{cases}
# 0 \text{ For  }abs(x)=\inf  \\
# 1 \text{ Otherwise}  
# \end{cases}
# \end{equation}

# In[67]:


### Let's just plot the distirbutions

##### Parameters
Mu = 80
sigma = 20

############### Probability
X = np.linspace(10, 160, 100)
Y = gaussian(X, sigma, Mu)
plt.plot(X, Y)
plt.show()


# In[68]:


############# Possiblity distribution
def triangular_fuzzy(X, Mu, left, right):
    """
    Triangular fuzzy number
    
    Parameters:
    -----------
    X : array
        the input range, can be an array of floating points
    Mu : integer or float
        the mean of the fuzzy number
    left : integer or float
        left expansion of the fuzzy number
    right : integer or float
        right expansion of the fuzzy number
    """
    ## apply the left expansion on all the input
    ## then using np.where apply the right expansion to the data
    
    ## Apply the right expansion in the first condition
    ## and apply the left expansion in the left condition
    Y = np.where(X > Mu, (right - X) / (right - Mu) ,(X - left) / (Mu - left))
    ## zero the negative numbers
    Y = np.where(Y < 0, 0, Y)
    
    return Y

#### Parameters
Mu = 80
right = 100
left = 60

#### Plot the possiblity distribution
X = np.linspace(10, 160, 100)
Y = triangular_fuzzy(X, Mu, left, right)
plt.plot(X, Y)
plt.show()


# ### Q10
# Answer to the question 3 of chapter 8 of Zimmerman book.
# 
# To find the possibility measures we should find the suprimum of the minimum values of $\mu_{\tilde{A}}(u), \pi_x{u}$  

# So because values for $A$ given in range(6, 14) the answers can be
# \begin{equation}
# \pi_{\tilde{A}} = \{(8, .6), (9, .8), (10, 1),(1 1, .8), (12, .6)\}
# \end{equation}
# suprimum of the minimum values will be 
# \begin{equation}
# \text{possiblity} = max(.6, 0.8, 1, 0.8, 0.6) = \color{green}1
# \end{equation}
# \begin{equation}
# \pi_{\tilde{B}} = {(6, .4), (7, .5), (8, .6), (9, .8), (10,1),(11,.8), (12, .6), (13,.5), (14, .4)}
# \end{equation}
# And for $\tilde{B}$:
# \begin{equation}
# \text{possiblity} = max(0.4, 0.5, 0.6, 0.8, 1, 0.8, 0.6, 0.5, 0.4) = \color{green}1
# \end{equation}
# 
# To discuss the values we can say that both possibilities for $\tilde{A}$ and $\tilde{B}$ is 1, meaning that they can happen with a probability.

# ### Q11

# Answer to the question 4 of chapter 4 of the Zimmerman book.

# Considering Example 4-1, we can answer find the probabilities of $A_1, A_2$ and $A_3$ via the equation below
# \begin{equation}
# \text{for }A \subseteq X \rightarrow \pi(A) = sup(\{x\})
# \end{equation}

# And given $A_1, A_2$ and $A_3$ are as below:
# \begin{equation}
# A_1 = \{1, 2, 3,4,5, 6\} \\
# A_2=\{1, 5, 8, 9\} \\ 
# A3=\{7, 9\}
# \end{equation}
# We can find the possiblity of them in the equations below

# \begin{equation}
# \text{possiblity}(A_1) = \pi(A_1) = sup(\pi(1), \pi(2), \pi(3), \pi(4), \pi(5), \pi(6)) = sup\{0, 0, 0, 0, 0.1, 0.5\} = \color{green}{0.5} \\
# \text{possiblity}(A_2) = \pi(A_1) = sup(\pi(1), \pi(5), \pi(8), \pi(9)) = sup\{0, 0.1, 1, 0.8\} = \color{green}1 \\
# \text{possiblity}(A_3) = \pi(A_1) = sup(\pi(7), \pi(9)) = sup\{0.8, 0.8\} = \color{green}{0.8}
# \end{equation}

# ### Q12
# Answer the question 5 of chapter 8 in Zimmerman book.

# To define the Yager's probablity of a fuzzy event for question 10 here, we need to first have a look at what it was.
# \begin{equation}
# P(A_\alpha) = \frac{\sum_{x \in A_{\alpha}} P(x)}{\sum_{x \in A} P(x)}
# \end{equation}
# So we can find out that for any $\alpha$ cut of the set $A$, we can compute its probability with summing the probabilities of each item in $\alpha$ cut.
# 
# first to find the probability of each item having the universe set $A$ in mind we can assign $\frac{1}{9}$ for each item probability.
# \begin{equation}
# A = \{6, 7, . . . , 13, 14\}
# \end{equation}
# And having $\pi_{\tilde{A}}$ in mind the question's answer will goes as follows
# ![image.png](attachment:image.png)
# 
# The complement of $\tilde{A}$ will be
# \begin{equation}
# \pi_{C\tilde{A}} = \{(6,1), (7,1), (8,0.4),(9, 0.2),(11, 0.2),(12, 0.4), (13, 1), (14, 1)\}
# \end{equation}
# And so for $\pi_{C\tilde{A}}$
# ![image-2.png](attachment:image-2.png)
# 
# So using yager's probability definition, our answer will be the intersection of $\pi_{\tilde{A}}$ and $\pi_{C\tilde{A}}$ 
# 
# \begin{equation}
# probability = \begin{cases}
# 0 \text{ for } w \in [0, \frac{4}{9}] \\
# 0.6 \text{ for } w \in [\frac{4}{9}, \frac{5}{9}] \\
# 0 \text{ for } w \in [\frac{5}{9}, 1] \\
# \end{cases}
# \end{equation}

# ## Extended operations on fuzzy numbers

# ### Q13
# Answer to the question 5, chapter 5 of Zimmerman book.

# In this question using the values of $\tilde{M}, L(x), R(x)$ in example 5-8, and having $\tilde{N} = (-4, .1, .6)_{LR}$, find the answer of $\tilde{M} \ominus \tilde{N}$.

# First we recall the values in example 5-8.
# \begin{equation}
# L(x) = R(x) = \frac{1}{1+x^2} \\
# \tilde{M} = (1, .5, .8)_{LR} 
# \end{equation}

# Then it's easy to answer the question. We know from the extended subtraction that we first need to apply opposite of $\tilde{N}$, then use extended sum to answer the question. So
# \begin{equation}
# \tilde{M} \ominus \tilde{N} = \tilde{M} \oplus (\ominus \tilde{N}) \\ (1, 0.5, 0.8)_{LR} \oplus (\ominus (-4, 0.1, 0.6)_{LR} \\
# (1, 0.5, 0.8)_{LR} \oplus (4, 0.6, 0.1)_{LR} \\
# \color{green}{(5, 1.1, 0.9)_{LR}}
# \end{equation}

# ### Q14
# Answer to the question 6, chapter 5 of Zimmerman book.

# For this question, We need to find $\tilde{M} \odot \tilde{N}$ and values for $\tilde{M}$ and $\tilde{N}$ are given in example 5-8 as below
# \begin{equation}
# L(x) = R(x) = \frac{1}{1+x^2} \\
# \tilde{M} = (1, .5, .8)_{LR} \\
# \tilde{N} = (2, .6, .2)_{LR}
# \end{equation}

# First we recall the equation for extended product
# \begin{equation}
# (n, \alpha, \beta) \odot (m, \gamma, \delta) = (mn, m\alpha + n\gamma, m\beta + n\delta)
# \end{equation}
# So using the answer can be written as
# \begin{equation}
# \tilde{M} \odot \tilde{N} = (1, 0.5, 0.8) \odot (2, 0.6, 0.2) = (2, 0.6 + 2 \times 0.5, 0.2 + 2 \times 0.8 ) = \color{green}{(2, 1.6, 1.8)} 
# \end{equation}

# ## Fuzzy Clustering

# ### Q15
# Answer to the question 7, chapter 13 of Zimmerman book.

# In this question, three possible fuzzy three-partitions for dataset $X=\{x_1, x_2, x_3, x_4, x_5\}$ is requested.
# 
# The answer can be a matrix whose sum each column must be 1, So we can write 3 possible matrixes for fuzzy 3-partition clustering
# 
# \begin{equation}
# U_1 = \begin{bmatrix}
# 0.3 & 0.5 & 0.1 & 0.25 & 0.07\\
# 0.2 & 0.1 & 0.3 & 0.35 & 0.08\\
# 0.5 & 0.4 & 0.6 & 0.4 & 0.85
# \end{bmatrix},
# U_2 = \begin{bmatrix}
# 0.1 & 0.35 & 0.15 & 0.9 & 0.45\\
# 0.4 & 0.15 & 0.25 & 0.05 & 0.05\\
# 0.5 & 0.5 & 0.6 & 0.05 & 0.05
# \end{bmatrix}, \\
# U_3 = \begin{bmatrix}
# 0.4 & 0.01 & 0.12 & 0.18 & 0.55\\
# 0.6 & 0.9 & 0.28 & 0.17 & 0.35\\
# 0.0 & 0.09 & 0.6 & 0.65 & 0.1
# \end{bmatrix}
# \end{equation}

# ### Q16
# Answer to the question 8, chapter 13 of Zimmerman book.

# In this question, a set of $X$ is given and (a) finding a 3 crisp partition that groups $(1, 3)$ and $(10, 3)$ together w.r.t minimizing Euclidean norm is requested. For (b) Doing the same procedure but minimizing the variance is needed.
# \begin{equation}
# X = \{(1, 1), (1, 3), (10, 1), (10, 3), (5, 2)\}
# \end{equation}

# #### Part a

# To group two set of values $(1, 3)$ and $(10, 3)$, we cannot use euclidean distance measure for both features $x_1$ and $x_2$ because the euclidean distance of the values $(1, 3)$ and $(10, 3)$ is $\sqrt{(1-10)^2 + (3-3)^2} = 9$. So it's a big value and the points cannot be grouped together. 
# 
# If we notice the second feature subtraction we can find out that using just the second feature can help us group together $(1, 3)$ and $(10, 3)$ using euclidean distance. So the procedure will goes as below
# 
# Grouping of $(1, 3)$ and $(10, 3)$ 
# \begin{equation}
# euclidean = \sqrt{(3-3)^2} = 0 
# \end{equation}
# Grouping $(1,1)$ and $(10, 1)$
# \begin{equation}
# euclidean = \sqrt{(1-1)^2} = 0 
# \end{equation}
# And just $(5, 2)$ is left as another partition. So using the second feature can help us group the $X$ set and represent us a perfect euclidean score of zero.

# #### Part b

# To minimize the variance we must first see how variance equation works. Variance shows us the sum of differences from mean value, so the equation will be as below
# \begin{equation}
# Var(x) = \frac{\sum_{i=0}^{n} (x_i - \bar{x})^2}{n}
# \end{equation}

# So we first find the mean values as below
# \begin{equation}
# \bar{x_1} = \frac{1+1+10+10+5}{5} = 5.4 \\
# \bar{x_2} = \frac{1+3+1+3+2}{5} = 2
# \end{equation}
# So using the mean values of $x_1$ and $x_2$, we first try the method in part a to see does it works as before or not.

# In[69]:


## the variance of (1, 3) and (10, 3)
((1-5.4)**2 + (10-5.4)**2)/2, ((3-2)**2 + (3-2)**2)/2, 


# In[70]:


## the variance of (1, 1) and (10, 1)
((1-5.4)**2 + (10-5.4)**2)/2, ((1-2)**2 + (1-2)**2)/2, 


# In[71]:


## the variance of (5, 2)
((5-5.4)**2)/1, (2-2)**2/1


# We've calculated the variances above and it is obvious that choosing the second features as in part a can minimize the variances for us. So we would stick to the solution in part (a)

# ### Q17
# Answer the question 9, chapter 13 of Zimmerman book.

# This question wants us to find the cluster validity of figures 13-11 and 13-12 using partition coefficient and partition entropy.
# 
# First we have to take a look at the figures 13-11 and 13-12.
# ![image.png](attachment:image.png)
# Then the equation for partition coefficient is
# \begin{equation}
# F(\tilde{U}, c) = \sum_{k=1}^{n} \sum_{i=1}^{c} \frac{(\mu_{ik})^2}{n}
# \end{equation}
# And partition entropy is
# \begin{equation}
# H(\tilde{U}, c) = -\frac{1}{n} \sum_{k=1}^{n} \sum_{i=1}^{c} \mu_{ik} log_e(\mu_{ik})
# \end{equation}

# For Figure 13-11 the partition coefficient and entropy will be calculated as
# \begin{equation}
# F(\tilde{U}, c) = \frac{(.99)^2 + (.99)^2 + 1^2 + 1^2 + 1^2 + 1^2 + (.99)^2 + (.47)^2 + (.01)^2 + (0)^2 + (0)^2 + (0)^2 + (.01)^2 + (.01)^2 + 0^2}{15}
# \end{equation}
# And the entropy will be as (we simplified the equation)
# \begin{equation}
# H(\tilde{U}, c) = -\frac{1}{15} \big(3 \times 0.99 log_e 0.99 + 4 \times 1 log_e 1 + 0.47 log_e 0.47 + 3 \times 0.01 log_e 0.01 + 4 \times 0 log_e 0\big)
# \end{equation}

# In[72]:


## the results is
def power_2(value):
    """
    find the power 2 of the value
    """
    return value ** 2

def entropy(value):
    """
    return the entropy of the value
    """
    return value * np.log(value)

## partition coefficient
F = 3 * power_2(0.99) + 4 * power_2(1) + power_2(0.47) + 3 * power_2(0.01)
F = F / 15
## entropy
H = 3 * entropy(0.99) + 4 * entropy(1) + entropy(0.47) + 3 * entropy(0.01)
H = H / -15

print("Figure 13-11: Partition Coefficient", F)
print("Figure 13-11: Partition Entropy", H)


# And for figure 13-12 the calculations are as
# 
# Partition Coefficient:
# \begin{equation}
# F(\tilde{U}, c) = \\ \frac{(.86)^2 + (.97)^2 + (.86)^2 + (.94)^2 + (.99)^2 + (.94)^2 + (.88)^2 + (.50)^2 + (.12)^2 + (0.06)^2 + (0.01)^2 + (0.06)^2 + (.14)^2 + (.03)^2 + (0.14)^2}{15}
# \end{equation}
# Entropy:
# \begin{equation}
# H(\tilde{U}, c) = -\frac{1}{15} \big(2 \times 0.86 log_e 0.86 + 0.97 log_e 0.97 + 2 \times 0.94 log_e 0.94 + 0.99 log_e 0.99 + 0.88 log_e 0.88 + 0.50 log_e 0.50 + 0.12 log_e 0.12 + 2 \times 0.06 log_e 0.06 + 0.01 log_e 0.01 + 2 \times 0.14 log_e 0.14 + 0.03 log_e 0.03 \big)
# \end{equation}

# In[73]:


## partition coefficient
## because the equation was long we divided into three parts
F = 2 * power_2(0.86) + power_2(0.97) + 2 * power_2(0.94) + power_2(0.99)
F += power_2(0.88) + power_2(0.5) + power_2(0.12) + 2 * power_2(0.06) 
F += power_2(0.01) + 2 * power_2(0.14) + power_2(0.03)

F = F / 15

## entropy
## it is divided in two parts to avoid long length expressions
H = 2 * entropy(0.86) + entropy(0.97) + 2 * entropy(0.94) + entropy(0.99) + entropy(0.88)
H += entropy(0.5) + entropy(0.12) + 2 * entropy(0.06) + entropy(0.01) + 2 * entropy(0.14) + entropy(0.03) 
H = H / -15

print("Figure 13-12: Partition Coefficient", F)
print("Figure 13-12: Partition Entropy", H)


# ### Q18
# Answer the question 1, chapter 10 of Ross book.

# Using the table given below, And creating $\alpha$-cut$=0.5$, find how many clusters we could have? (Use max-min operator to first generate fuzzy similarity in $\tilde{R}$)
# ![image.png](attachment:image.png)

# In[74]:


def fuzzy_max_min(relation1, relation2):
    """
    apply fuzzy max-min on two relations
    
    Parameters:
    -----------
    relation1 : matrix_like
        the relation matrix containing fuzzy values
    relation2 : matrix_like
        the relation matrix containing fuzzy values
    
    Returns:
    ---------
    result_relation : matrix_like
        the result of max-min composition on relation1 and relation2
    """
    assert relation1.shape[1] == relation2.shape[0], 'Error: Relation shapes must match!'
    
    ## initialize an empty array with None values
    result_relation = np.full((relation1.shape[0], relation2.shape[1]), None)

    for i in range(relation1.shape[0]):
        for j in range(relation2.shape[1]):
            values = np.minimum(relation1[i], relation2[:, j].T)
            
            result_relation[i, j] = np.max(values)
    
    return result_relation


# In[75]:


table = np.matrix('.4 .1 .3 .8; .2 .7 .2 .1; .4 .2 .5 .1')
table


# In[76]:


table_relation = fuzzy_max_min(table, table.T)
table_relation


# So the $\tilde{R}$ is now created and for $\alpha$-cut$=0.5$ we can find the matrix below.

# In[77]:


## weak alpha cut is used
(table_relation >= 0.5).astype(np.int32)


# In the relation above we can see that three clusters are apeared and the patterns are divided into:
# \begin{equation}
# Cluster_1 = \{a\} \\ Cluster_2 = \{b\} \\ Cluster_3 = \{c\}
# \end{equation}

# ### Q19
# Answer question 4, chapter 10 of Ross book.

# In this question 7 values are given and using a $\tilde{U}^0$ as an initiali reference function, we need to apply a fuzzy clustering. ($m^{'}=2.0$ and $\epsilon_L \leq 0.01$)

# \begin{equation}
# Dataset = \begin{pmatrix}
# x_1 \\ x_2
# \end{pmatrix} =
# \begin{pmatrix}
# 2 & 9 & 9 & 5 & 8 & 5 & 6 \\
# 7& 3 & 4 & 6 & 8 & 11 & 1
# \end{pmatrix} \\
# \tilde{U}^0 = \begin{pmatrix}
# 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
# 1 & 1 & 1 & 1 & 1 & 1 & 0
# \end{pmatrix}
# \end{equation}
# Note: $\tilde{U}^0$ given in the question was wrong (it has 8 columns), So because of that we changed it to 7 columns to match the dataset size.

# First let's recall the equations needed for this exercise.
# \begin{equation}
# V_i = \frac{1}{\sum_{k=1}^n \mu_{ik}^m} \sum_{k=1}^n \mu_{ik}^m x_k \text{,     For i=1,..., c} \\
# \mu_{ik} = \frac{\big(\frac{1}{||x_k - v_i||_G^2} \big)^{1/(m-1)}}{ \sum_{j=1}^c \big(\frac{1}{||x_k - v_j||_G^2} \big)^{1/(m-1)}} \text{,     For i=1,..., c; k=1,..., n}
# \end{equation}

# In[78]:


## we define the given U reference function as our initial mu_ik
U = np.matrix('0 0 0 0 0 0 1; 1 1 1 1 1 1 0').T
## rename U to match the notation of Zimmerman book and our equations
mu = U.astype(np.float32)

dataset = np.matrix('2 9 9 5 8 5 6; 7 3 4 6 8 11 1').T

dataset.shape, U.shape


# In[79]:


## the parameter for C-means
m = 2


# In[80]:


def find_cluster_center(mu, x, m):
    """
    Find the cluster center for Fuzzy C-means or crisp K-means method
    depend on `mu` this function can either work for crisp K-means and Fuzzy C-means
    
    Parameters:
    ------------
    mu : 2D array
        reference matrix that represents the membership of each data to each cluster
    x : 2D array
        input data that have x and y coordinates
        
    Returns:
    ---------
    center : array
        center representing the center value for x and y data
    """
    with np.errstate(divide='ignore'):
        division = np.divide(1, np.sum(np.power(mu, m), axis=0))
        ## check the 1 divided by 0 values and change it to 0
        division = np.where(division == np.inf, 0, division)
        
        center = 0.0
        for i in range(len(mu)):
            center += np.power(mu[i], m).item() * x[i]
        
        center = np.multiply(division, center)
#             center = np.multiply(division, np.sum(np.multiply(np.power(mu, m), x) ,axis=0))

    return center

def find_reference_matrix(v_correspond, v, x, m):
    """
    Find the cluster reference matrix for for Fuzzy C-means or crisp K-means method
    depend on `v` this function can either work for crisp K-means and Fuzzy C-means
    
    
    Parameters:
    ------------
    v_correspond : array
        centers of belonging x to the cluster
    v : 2D array
        center of the all clusters
    x : 2D array
        input data that have x and y coordinates
    m : float bigger than 1
        parameter for the function
    
    Returns:
    ---------
    reference_matrix : array
        newly created reference array for one data
    """
    if m <= 1:
        raise ValueError("V cannot be less equal than 1!")
    with np.errstate(divide='ignore', invalid= 'ignore'):
        ## above and below of the division
        above = np.power(np.divide(1, np.linalg.norm(x - v_correspond)), 1/ (m-1))
        ## change infinity values to zero
        above = np.where(above == np.inf, 0, above)

        below = 0
        for center in v:
            below += np.power(np.divide(1, np.linalg.norm(x - center)), 1 / (m-1))
            ## change infinity values to zero
            below = np.where(below == np.inf, 0, below)
 
        reference_matrix = np.divide(above, below)
        ## change infinity values to zero
        reference_matrix = np.where(reference_matrix == np.inf, 0, reference_matrix)
        reference_matrix = np.where(np.isnan(reference_matrix), 0, reference_matrix)
        

    return reference_matrix


# In[81]:


def update_CMeans_values(mu, dataset, centers, m):
    """
    apply one iteration of fuzzy C-means updating center means and reference matrix
    
    Parameters:
    ------------
    mu : 2D array
        reference matrix that represents the membership of each data to each cluster
    dataset : 2D array
        2D array representing x and y coordinates
    centers : 2D array
        array of centers representing the centers of clusters
    m : float bigger than 1
        parameter for the function
        
    Returns:
    ---------
    mu_new : 2D array
        the updated reference values
    centers_new : 2D array
        the updated centers 
    """
    
    ## initialization of variables
    centers_new = np.copy(centers) 
    mu_new = np.copy(mu)
    
    ########### update the reference functions
    for i in range(len(dataset)):
        mu_new[i, 0] = find_reference_matrix(centers_new[0], centers_new, dataset[i], m)
        mu_new[i, 1] = find_reference_matrix(centers_new[1], centers_new, dataset[i], m)
        
        
    ########## update the centers

    
    ###### The commented code is for crisp K-means, we need the Fuzzy C-means
#     ## thoes belong to cluster 1
#     cluster1_mu = mu[np.array(mu[:, 0] < mu[:, 1]).flatten()]
#     ## and thoes belong to cluster number 0
#     cluster0_mu = mu[np.array(mu[:, 0] > mu[:, 1]).flatten()]

#     ## data that belongs to cluster 1
#     cluster1_data = dataset[np.array(mu[:, 0] < mu[:, 1]).flatten()]

#     ## data that belongs to cluster 0
#     cluster0_data = dataset[np.array(mu[:, 0] > mu[:, 1]).flatten()]     

#     ## update the centers using the new membership function
#     center1 = find_cluster_center(cluster1_mu, cluster1_data, m)
#     center2 = find_cluster_center(cluster0_mu, cluster0_data, m)
    
    ## using C-means
    center1 = find_cluster_center(mu[:, 0], dataset, m)
    center0 = find_cluster_center(mu[:, 1], dataset, m)
    
    centers_new = np.concatenate((center0, center1))
    
    return mu_new, centers_new


# In[82]:


def fit_Cmeans(dataset , mu, centers ,epsilon=0.01, m=2, max_iter=1000):
    """
    Fit on data iteratively using C-Means method
    
    Parametrs:
    -----------
    dataset : 2D array
        set of x and y data coordinates
    mu : 2D array
        reference matrix that represents the membership of each data to each cluster
    centers : 2D array
        array of centers representing the centers of clusters
    epsilon : float
        when to stop the iterations, where values are changing less than `epsilon`
        default is `0.01`
    m : float bigger than 1
        parameter for the function
        default value is `2`
    max_iter : positive integer
        if the method doesn't converge, how many iterations it should iterate
        default is `1000`
        
        
    Returns:
    ---------
    new_mu : 2D array
        new updated reference matrix
    new_centers : 2D array
        new updated center values
    """
    
    new_centers = np.copy(centers)
    new_mu = np.copy(mu)
    
    ## set to be the default condition
    condition = True
    ## set to have the count of iterations
    i = 0
    while condition:
        ## save the last iteration values to be able to find how values are changing
        old_mu, old_centers = new_mu, new_centers
        ## then update
        new_mu, new_centers = update_CMeans_values(new_mu, dataset, new_centers, m)
        
        measure1 = np.sum(new_mu - old_mu)
        measure2 = np.sum(new_centers - old_centers)
        
        if ((measure1 + measure2) < epsilon):
            print(f"Method converged to a local optimum in {i+1} iterations!")
            condition = False
        elif i > max_iter:
            print(f"Method coudn't converge to a local optimum and ended with {i+1} iterations!")
            condition = False
    
    return new_mu, new_centers


# In[83]:


center0 = find_cluster_center(mu[:, 0], dataset, m)
center1 = find_cluster_center(mu[:, 1], dataset, m)

centers = np.concatenate((center0, center1))


# In[84]:


fit_Cmeans(dataset, mu, centers)

