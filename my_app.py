#!/usr/bin/env python
# coding: utf-8

# In[1]:

import streamlit 
import pandas as pd
import numpy as np
pip install pulp
from pulp import *
import matplotlib.pyplot as plt
import random
random.seed(1447)


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


pd.set_option('display.max_colwidth', 0)
pd.set_option('display.max_columns', None)
pd.options.display.max_seq_items = 2000


# In[4]:


get_ipython().run_cell_magic('html', '', '<style>\n.dataframe td {\n    white-space: nowrap;\n}\n</style>')


# ## Import Data

# #### Manufacturing variable costs

# In[5]:


# Import Costs
manvar_costs = pd.read_excel('variable costs.xlsx', index_col = 0)
manvar_costs # int


# #### Freight costs

# In[6]:


# Import Costs
freight_costs = pd.read_excel('freight costs.xlsx', index_col = 0)
freight_costs # int 


# In[7]:


# make changes in freight costs 
for country in freight_costs:
    if country != 'INDIA':
        freight_costs[country]['INDIA'] += 200
freight_costs.astype(int)


# #### Variable Costs

# In[8]:


# Factory + Freight Variable Costs
var_cost = freight_costs/1000 + manvar_costs 
var_cost # float


# #### Fixed Costs

# In[9]:


# Factory Fixed Costs
fixed_costs = pd.read_excel('fixed cost.xlsx', index_col = 0)
fixed_costs # int


# #### Plants Capacity

# In[10]:


# Two types of plants: Low Capacity and High Capacity Plant
cap = pd.read_excel('capacity.xlsx', index_col = 0)
cap # int


# #### Demand 

# In[11]:


# Demand by Market
demand = pd.read_excel('demand.xlsx', index_col = 0)
demand # int 


# In[12]:


demand.info()


# In[13]:


# increased demand by 20% 
demand['Demand'] = demand['Demand']+(demand['Demand']*20/100)
demand.astype(int)


# In[ ]:

with st.sidebar.expander('Import Data'):
    # Import Costs
    manvar_costs = pd.read_excel('variable costs.xlsx', index_col = 0)
    freight_costs = pd.read_excel('freight costs.xlsx', index_col = 0)

    # make changes in freight costs 
    for country in freight_costs:
        if country != 'INDIA':
            freight_costs[country]['INDIA'] += 200
    freight_costs.astype(int)

    # Factory + Freight Variable Costs
    var_cost = freight_costs/1000 + manvar_costs 

    # Factory Fixed Costs
    fixed_costs = pd.read_excel('fixed cost.xlsx', index_col = 0)

    # Two types of plants: Low Capacity and High Capacity Plant
    cap = pd.read_excel('capacity.xlsx', index_col = 0)

    # Demand by Market
    demand = pd.read_excel('demand.xlsx', index_col = 0)

    # increased demand by 20% 
    demand['Demand'] = demand['Demand']+(demand['Demand']*20/100)
    demand.astype(int)

    st.write('## Manufacturing variable costs')
    st.write(manvar_costs)

    st.write('## Freight costs')
    st.write(freight_costs)

    st.write('## Factory + Freight Variable Costs')
    st.write(var_cost)

    st.write('## Factory Fixed Costs')
    st.write(fixed_costs)

    st.write('## Plants Capacity')
    st.write(cap)

    st.write('## Demand by Market')
    st.write(demand)

demand_increase = st.slider('Demand Increase Percentage', 0, 50, 20, 1)
demand['Demand'] = demand['Demand'] + (demand['Demand'] * demand_increase / 100)
demand.astype(int)


st.write('### Results Plant (Boolean)')
df_bool = pd.DataFrame(data=[y[plant_name[i]].varValue for i in range(len(plant_name))], index=[i + '-' + s for s in size for i in loc], columns=['Plant Opening'])
st.write(df_bool)

st.write('### Results Production')
df = pd.DataFrame(data=[[x[(i,j)].varValue for j in loc] for i in loc], index=loc, columns=loc)
st.write(df)






# ---

# ## Initial Calculation 
# (1 Scenario)

# In[14]:


# Define Decision Variables

# list of locations where plants can be opened
loc = ['USA', 'GERMANY', 'JAPAN', 'BRAZIL', 'INDIA']

# list of capacity size for each plant
size = ['LOW', 'HIGH']

# a list of tuples representing all possible combinations of location and size.
plant_name = [(i,s) for s in size for i in loc]

# a list of tuples representing all possible combinations of production locations.
prod_name = [(i,j) for i in loc for j in loc] 


# The objective function 
# minimizes the total costs, 
# which include fixed costs (associated with opening plants) and variable costs (associated with producing goods).
# Initialize Class
model = LpProblem("Capacitated Plant Location Model", LpMinimize)



# Create Decision Variables
# a dictionary of continuous decision variables representing the production level between every pair of production locations
x = LpVariable.dicts("production_", prod_name,
                     lowBound=0, upBound=None, cat='continuous')

# a dictionary of binary decision variables representing whether a plant is opened or not in each location and size combination.
y = LpVariable.dicts("plant_", 
                     plant_name, cat='Binary')




# Define Objective Function
# The objective function minimizes the total costs, 
# which include fixed costs (associated with opening plants) and variable costs (associated with producing goods)
model += (lpSum([fixed_costs.loc[i,s] * y[(i,s)] * 1000 for s in size for i in loc])
          + lpSum([var_cost.loc[i,j] * x[(i,j)]   for i in loc for j in loc]))



# Add Constraints
# The first set of constraints ensure 
# that the total production from each production location is equal to the demand for that location.
# The second set of constraints ensure 
# that the total production from each plant does not exceed its capacity (given by the combination of location and size).
for j in loc:
    model += lpSum([x[(i, j)] for i in loc]) == demand.loc[j,'Demand']    
for i in loc:
    model += lpSum([x[(i, j)] for j in loc]) <= lpSum([cap.loc[i,s]*y[(i,s)] * 1000
                                                       for s in size])                                                 

    

# Solve Model
model.solve()
# the status of the model 
print("Status: {}".format(LpStatus[model.status]))
# total cost 
print("Total Costs: {:,} ($/Month)".format(int(value(model.objective))))






# Results Plant (Boolean)
# A dataframe is created to display the decision variables for opening plants (y) and their corresponding values.
df_bool = pd.DataFrame(data = [y[plant_name[i]].varValue for i in range(len(plant_name))], index = [i + '-' + s for s in size for i in loc], 
                        columns = ['Plant Opening'])
df_bool


# In[15]:


plant_name


# In[16]:


# 1.transport capacity/ freight capacity constraint: x[(i,j)] <= transport_capacity[(i,j)]

# 2.prodcution balance constraint: lpSum([x[(i,j)] for i in loc]) == lpSum([x[(i,j)] for i in loc if (i,j) in prod_name])
#   production level at a location should be equal to the sum of production levels from all the plants supplying to that location


# #### Plant Opening

# In[17]:


# Plant Opening
cap_plot = cap.copy()

ax = df_bool.astype(int).plot.bar(figsize=(8, 5), edgecolor='black', color = 'tab:green', y='Plant Opening', legend= False)
plt.xlabel('Plant')
plt.ylabel('Open/Close (Boolean)')
plt.title('Initial Solution')
plt.show()


# ---

# ### Functions to simulate several scenarios
# #### Funtion to build the model

# In[18]:


def optimization_model(fixed_costs, var_cost, demand, demand_col, cap):
    '''Build the optimization based on input parameters'''
    # Define Decision Variables
    loc = ['USA', 'GERMANY', 'JAPAN', 'BRAZIL', 'INDIA']
    size = ['LOW', 'HIGH']
    plant_name = [(i,s) for s in size for i in loc]
    prod_name = [(i,j) for i in loc for j in loc]   

    # Initialize Class
    model = LpProblem("Capacitated Plant Location Model", LpMinimize)

    # Create Decision Variables
    x = LpVariable.dicts("production_", prod_name,
                         lowBound=0, upBound=None, cat='continuous')
    y = LpVariable.dicts("plant_", 
                         plant_name, cat='Binary')

    # Define Objective Function
    model += (lpSum([fixed_costs.loc[i,s] * y[(i,s)] * 1000 for s in size for i in loc])
              + lpSum([var_cost.loc[i,j] * x[(i,j)]   for i in loc for j in loc]))

    # Add Constraints
    for j in loc:
        model += lpSum([x[(i, j)] for i in loc]) == demand.loc[j,demand_col]
    for i in loc:
        model += lpSum([x[(i, j)] for j in loc]) <= lpSum([cap.loc[i,s]*y[(i,s)] * 1000
                                                           for s in size])                                                 
    # Solve Model
    model.solve()
    
    # Results
    status_out = LpStatus[model.status]
    objective_out  = pulp.value(model.objective)
    plant_bool = [y[plant_name[i]].varValue for i in range(len(plant_name))]
    fix = sum([fixed_costs.loc[i,s] * y[(i,s)].varValue * 1000 for s in size for i in loc])
    var = sum([var_cost.loc[i,j] * x[(i,j)].varValue for i in loc for j in loc])
    plant_prod = [x[prod_name[i]].varValue for i in range(len(prod_name))]
    return status_out, objective_out, y, x, fix, var


# #### Build the normal distribution of demand: N(demand, demand x COV)

# In[19]:


# # Normal Distribution
# N = 50
# df_demand = pd.DataFrame({'scenario': np.array(range(1, N + 1))})
# data = demand.reset_index()
# # Demand 
# CV = 0.5
# markets = data['(Units/month)'].values
# for col, value in zip(markets, data['Demand'].values):
#     sigma = CV * value
#     df_demand[col] = np.random.normal(value, sigma, N)
#     df_demand[col] = df_demand[col].apply(lambda t: t if t>=0 else 0)

# # Add Initial Scenario
# COLS = ['scenario'] + list(demand.index)
# VALS = [0] + list(demand['Demand'].values)
# df_init = pd.DataFrame(dict(zip(COLS, VALS)), index = [0])

# # Concat
# df_demand = pd.concat([df_init, df_demand])
# df_demand.to_excel('df_demand-{}PC.xlsx'.format(int(CV * 100)))
    
# df_demand.astype(int).head()


# In[20]:


# Different variation for different markets.
df_cv = pd.DataFrame({
    'Market': ['USA', 'GERMANY', 'JAPAN', 'BRAZIL', 'INDIA'],
    'CV': [0.1, 0.2, 0.4, 0.6, 0.8]
})


# In[21]:


# Normal Distribution
N = 200
df_demand = pd.DataFrame({'scenario': np.array(range(1, N + 1))})
data = demand.reset_index()
# Demand 

markets = data['(Units/month)'].values
for col, value in zip(markets, data['Demand'].values):
    cv = df_cv.loc[df_cv['Market'] == col, 'CV'].values[0] # Get CV value for market
    sigma = cv* value
    df_demand[col] = np.random.normal(value, sigma, N)
    df_demand[col] = df_demand[col].apply(lambda t: t if t>=0 else 0)

# Add Initial Scenario
COLS = ['scenario'] + list(demand.index)
VALS = [0] + list(demand['Demand'].values)
df_init = pd.DataFrame(dict(zip(COLS, VALS)), index = [0])

# Concat
df_demand = pd.concat([df_init, df_demand])
# df_demand.to_excel('df_demand-{}PC.xlsx'.format(int(cv * 100)))
# df_demand.to_excel('df_demand_new.xlsx')    
df_demand.astype(int).head()


# In[22]:


# Plot
figure, axes = plt.subplots(len(markets), 1)
colors = ['tab:green', 'tab:red', 'black', 'tab:blue', 'tab:orange']
for i in range(len(markets)):
    df_demand.plot(figsize=(20, 12), xlim=[0,N], x='scenario', y=markets[i], ax=axes[i], grid = True, color = colors[i])
    axes[i].axhline(df_demand[markets[i]].values[0], color=colors[i], linestyle="--")
plt.xlabel('Scenario')
plt.ylabel('(Units)')
plt.xticks(rotation=90)
plt.show()


# #### Calculation: Initial Scenario

# In[23]:


# Record results per scenario
list_scenario, list_status, list_results, list_totald, list_fixcost, list_varcost = [], [], [], [], [], []
# Initial Scenario
status_out, objective_out, y, x, fix, var = optimization_model(fixed_costs, var_cost, demand, 'Demand', cap)

# Add results
list_scenario.append('INITIAL')
total_demand = demand['Demand'].sum()
list_totald.append(total_demand)
list_status.append(status_out)
list_results.append(objective_out)
list_fixcost.append(fix)
list_varcost.append(var)
# Dataframe to record the solutions
df_bool = pd.DataFrame(data = [y[plant_name[i]].varValue for i in range(len(plant_name))], index = [i + '-' + s for s in size for i in loc], 
                        columns = ['INITIAL'])
df_bool


# In[24]:


list_status


# In[25]:


# Simulate all scenarios
demand_var = df_demand.drop(['scenario'], axis = 1).T

# Loop
for i in range(1, 200): # 0 is the initial scenario 
    # Calculations
    status_out, objective_out, y, x, fix, var = optimization_model(fixed_costs, var_cost, demand_var, i, cap)    
    
    # Append results
    list_status.append(status_out)
    list_results.append(objective_out)
    df_bool[i] = [y[plant_name[i]].varValue for i in range(len(plant_name))]
    list_fixcost.append(fix)
    list_varcost.append(var)
    total_demand = demand_var[i].sum()
    list_totald.append(total_demand)
    list_scenario.append(i)
# Final Results
# Boolean
df_bool = df_bool.astype(int)
# df_bool.to_excel('boolean-{}PC.xlsx'.format(int(cv * 100)))
# df_bool.to_excel('boolean_new.xlsx')
# Other Results
df_bool.head()


# In[ ]:





# ---

# ### Final Plot
# #### Boolean Alone

# In[26]:


# Plot the Grid
plt.figure(figsize = (20,4))
plt.pcolor( df_bool, cmap = 'Blues', edgecolors='k', linewidths=0.5)   #
plt.xticks([i + 0.5 for i in range(df_bool.shape[1])], df_bool.columns, rotation = 90, fontsize=12)
plt.yticks([i + 0.5 for i in range(df_bool.shape[0])], df_bool.index, fontsize=12)
plt.show()


# #### Add Demand 

# In[27]:


# Plot
figure, axes = plt.subplots(len(markets), 1)
colors = ['tab:green', 'tab:red', 'black', 'tab:blue', 'tab:orange']
for i in range(len(markets)):
    df_demand.plot(figsize=(15, 15), xlim=[1,N], x='scenario', y=markets[i], ax=axes[i], grid = True, color = colors[i])
    axes[i].axhline(df_demand[markets[i]].mean(), color=colors[i], linestyle="--")
plt.xlabel('Scenario')
plt.ylabel('(Units)')

# add the scenario plot
plt.figure(figsize=(15, 5))
plt.pcolor(df_bool, cmap = 'Blues', edgecolors='k', linewidths=0.5)   #
plt.xticks([i + 0.5 for i in range(df_bool.shape[1])], df_bool.columns, rotation = 90, fontsize=12)
# plt.yticks([i + 0.5 for i in range(df_bool.shape[0])], [d[0:9]+ '-H' * ('HIGH' in d) + '-L' * ('LOW' in d) for d in df_bool.index], fontsize=12)
plt.yticks([i + 0.5 for i in range(df_bool.shape[0])], [d[0:20]+ '-H' * ('HIGH' in d) + '-L' * ('LOW' in d) for d in df_bool.index], fontsize=12)
plt.xticks(rotation=90)
plt.show()


# ---

# ## Find the optimal Solution
# #### Unique Combinations

# In[28]:


# Unique combinations
df_unique = df_bool.T.drop_duplicates().T
df_unique.columns = ['INITIAL'] + ['C' + str(i) for i in range(1, len(df_unique.columns))]
# Plot the Grid
plt.figure(figsize = (12,4))
plt.pcolor( df_unique, cmap = 'Blues', edgecolors='k', linewidths=0.5)   #
plt.xticks([i + 0.5 for i in range(df_unique.shape[1])], df_unique.columns, rotation = 90, fontsize=12)
plt.yticks([i + 0.5 for i in range(df_unique.shape[0])], df_unique.index, fontsize=12)
plt.show()


# #### Number of Combinations 

# In[29]:


# Number of columns
COL_NAME, COL_NUMBER = [], []
for col1 in df_unique.columns:
    count = 0
    COL_NAME.append(col1)
    for col2 in df_bool.columns:
        if (df_bool[col2]!=df_unique[col1]).sum()==0:
            count += 1
    COL_NUMBER.append(count)
df_comb = pd.DataFrame({'column':COL_NAME, 'count':COL_NUMBER}).set_index('column')

my_circle = plt.Circle( (0,0), 0.8, color='white')
df_comb.plot.pie(figsize=(8, 8), x='column', y='count', legend= False, pctdistance=0.7,
                                          autopct='%1.0f%%', labeldistance=1.05, 
                                          wedgeprops = { 'linewidth' : 7, 'edgecolor' : 'white' })
plt.xlabel('Business Vertical')
# plt.title('{:.2f} M€ Budget Applications in 9 Vertical Markets'.format(df_p['TOTAL'].sum()/1e6))
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.axis('off')
plt.show()


# In[ ]:




