#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd
import numpy as np
from pulp import *
import matplotlib.pyplot as plt
import random
random.seed(1447)


# In[50]:


import warnings
warnings.filterwarnings("ignore")


# In[51]:


pd.set_option('display.max_colwidth', 0)
pd.set_option('display.max_columns', None)
pd.options.display.max_seq_items = 2000


# In[52]:


get_ipython().run_cell_magic('html', '', '<style>\n.dataframe td {\n    white-space: nowrap;\n}\n</style>')


# ## Import Data

# #### Manufacturing variable costs

# #### Transportation costs

# In[53]:


# Import Costs
freight_costs = pd.read_excel('freight_costs.xlsx', index_col = 0)


# In[54]:


freight_costs = freight_costs.T


# In[55]:


freight_costs = freight_costs[0:7]


# In[56]:


freight_costs.T.describe()


# In[57]:


boxplot_freight = freight_costs.T.boxplot(column=['Macon', 'Olathe', 'SLC', 'Fresno', 'Hamburg', 'Dallas', 'Indy'])
plt.show()


# #### Plants Capacity

# In[58]:


# Two types of plants: Low Capacity and High Capacity Plant
cap = pd.read_excel('capacity_new.xlsx', index_col = 0)
cap # int
cap = cap[0:7]


# In[59]:


cap["Capacity"] = cap["Capacity_new"]


# In[60]:


cap = cap.drop(columns = 'Capacity_new')


# In[61]:


# cap.loc["Indy","Capacity"]= 15000


# In[62]:


cap


# In[63]:


cap["Capacity"].sum()


# In[64]:


cap.info()


# #### Demand 

# In[65]:


# Demand by Market
demand = pd.read_excel('Demand_new.xlsx', index_col = 0)
demand = demand.sort_values(by=demand.columns[1], ascending=False)
 # int 


# In[66]:


demand.info()


# In[67]:


lead_time = pd.read_excel("lead_time.xlsx",index_col = 0)
lead_time = lead_time.T


# In[68]:


lead_time.T.describe()


# In[69]:


boxplot_LT = lead_time.T.boxplot(column=['Macon', 'Olathe', 'SLC', 'Fresno', 'Hamburg', 'Dallas', 'Indy'])
plt.show()


# ## --------------------------------------------------------------------------------------------------------------------------

# In[70]:


warehouses = cap.index.tolist()


# In[71]:


markets = demand.index.tolist()


# ### Iteration 1:without capacity constraint   

# In[72]:


import pulp


prob = pulp.LpProblem("Minimize Transportation Costs and Lead Time", pulp.LpMinimize)

route_name = [(i,j) for i in warehouses for j in markets] 


x = pulp.LpVariable.dicts("transport", route_name, lowBound=0, cat='Continuous')


prob += (pulp.lpSum(freight_costs.loc[i,j] * x[(i,j)] + lead_time.loc[i,j]  for i in warehouses for j in markets))



# for i in warehouses:
#     prob += pulp.lpSum(x[(i,j)] for j in markets) <= cap.loc[i]
for j in markets:
    prob += pulp.lpSum(x[(i,j)] for i in warehouses) == demand["Demand"].loc[j]


prob.solve()

# Print the optimal solution in descending order of demand
print("Optimal solution:")
for j in sorted(markets, key=lambda j: -demand["Demand"].loc[j]):
    for i in warehouses:
        if x[(i,j)].value() != 0:
            print("x_{}{} = {}".format(i, j, x[(i,j)].value()))
print("Total cost:", pulp.value(prob.objective))
data= []
for i in warehouses:
    for j in markets:
        if x[(i,j)].value() != 0:
            data.append([i, j, x[(i,j)].value()])
results = pd.DataFrame(data, columns=['Warehouse', 'Market', 'Transport'])
results['Transport'] = results['Transport'].round(3)


# In[73]:


total_cost = {}
for i in warehouses:
    total_cost[i] = 0

# Print the optimal solution and calculate the total cost for each warehouse
print("Optimal solution:")
for i in warehouses:
    for j in markets:
        if x[(i,j)].value() != 0:
 
            total_cost[i] += freight_costs.loc[i,j] * x[(i,j)].value() 
#     + lead_time.loc[i,j] * x[(i,j)].value()

# Print the total cost for each warehouse
for i in warehouses:
    print("Total cost for {}: {}".format(i, total_cost[i]))
    
    


# In[74]:


import pandas as pd

# Calculate and print results for each warehouse
results_1 = []
for i in warehouses:
    total_shipped = sum(x[(i,j)].value() for j in markets)
    capacity_left = float(cap.loc[i] - total_shipped)
    percentage_left = 100 * capacity_left / cap.loc[i]
    results_1.append({
        'Warehouse': i,
        'Total shipped': total_shipped,
        'Quantity left in warehouse': int(capacity_left),
        'Percentage left': float(round(percentage_left, 4)),
        'Original capacity': int(cap.loc[i])
    })


# Create DataFrame from results
df = pd.DataFrame(results_1)


# Print total cost
print("Total cost:", pulp.value(prob.objective))

df.set_index('Warehouse', inplace=True)



df['Total Cost'] = df.index.map(lambda w: total_cost[w])
df['per_unit_cost'] = df["Total Cost"]/df["Total shipped"]
df


# In[75]:


N = 50
df_demand = pd.DataFrame({'scenario': np.array(range(1, N + 1))})
data = demand.reset_index()
# Demand 
CV = 0.25
markets = data['Route'].values
for col, value in zip(markets, data['Demand'].values):
    sigma = CV * value
    df_demand[col] = np.random.normal(value, sigma, N)
    df_demand[col] = df_demand[col].apply(lambda t: t if t>=0 else 0)

# Add Initial Scenario
COLS = ['scenario'] + list(demand.index)
VALS = [0] + list(demand['Demand'].values)
df_init = pd.DataFrame(dict(zip(COLS, VALS)), index = [0])

# Concat
df_demand = pd.concat([df_init, df_demand])
    
df_demand.astype(int).head()
demand_var = df_demand.drop(['scenario'], axis = 1).T


# In[76]:


def optimization_model_itr_1(freight_costs, lead_time, demand, demand_col, cap):
    '''Build the optimization based on input parameters'''
    # Define Decision Variables
    warehouses
    markets
    route_name = [(i,j) for i in warehouses for j in markets] 

    # Initialize Class
    prob = pulp.LpProblem("Minimize Transportation Costs and Lead Time", pulp.LpMinimize)

    # Create Decision Variables
    x = pulp.LpVariable.dicts("transport", route_name, lowBound=0, cat='Continuous')

    # Define Objective Function
    prob += (pulp.lpSum(freight_costs.loc[i,j] * x[(i,j)] + lead_time.loc[i,j] for i in warehouses for j in markets))

    
    
    # Add Constraints
    for j in markets:
        prob += pulp.lpSum(x[(i,j)] for i in warehouses) == demand.loc[j,demand_col]                                                 
    # Solve Model
    prob.solve()
    # Print the optimal solution
    print("Optimal solution:")
    for i in warehouses:
        for j in markets:
            if x[(i,j)].value() != 0:
                print("x_{}{} = {}".format(i, j, x[(i,j)].value()))
    print("Total cost:", pulp.value(prob.objective))
    
    # Results
    status_out = LpStatus[prob.status]
    objective_out  = pulp.value(prob.objective)
    FC = sum([freight_costs.loc[i,j] * x[(i,j)].varValue for i in warehouses for j in markets])
    return status_out, objective_out,x,FC


# In[77]:


list_scenario, list_status, list_results, list_totald, list_FC,data_itr_1_mc = [], [], [], [], [],[]

for i in range(1, 50): # 0 is the initial scenario 
    # Calculations
    status_out, objective_out, x, FC = optimization_model_itr_1(freight_costs, lead_time, demand_var, i, cap)
    
    list_status.append(status_out)
    list_results.append(objective_out)
    list_FC.append(FC)
    total_demand = demand_var[i].sum()
    list_totald.append(total_demand)
    list_scenario.append(i)
    for i in warehouses:
        for j in markets:
            if x[(i,j)].value() != 0:
                data_itr_1_mc.append({
                    'Warehouse': i,
                    'Market': j
                })

    results_itr_1_mc = pd.DataFrame(data_itr_1_mc, columns=['Warehouse', 'Market'])


# In[78]:


results_itr_1_mc.info()


# In[79]:


results_itr_1_mc.drop_duplicates(subset=["Warehouse","Market"])


# ## --------------------------------------------------------------------------------------------------------------------------

# ### Iteration 2: with capacity constraint 
# 

# In[80]:


import pulp

prob_cap = pulp.LpProblem("Minimize Transportation Costs and Lead Time", pulp.LpMinimize)

route_name = [(i,j) for i in warehouses for j in markets] 

x = pulp.LpVariable.dicts("transport", route_name, lowBound=0, cat='Continuous')

prob_cap += (pulp.lpSum(freight_costs.loc[i,j] * x[(i,j)] + lead_time.loc[i,j]  for i in warehouses for j in markets))


for i in warehouses:
    prob_cap += pulp.lpSum(x[(i,j)] for j in markets) <= cap.loc[i]
for j in markets:
    prob_cap += pulp.lpSum(x[(i,j)] for i in warehouses) == demand["Demand"].loc[j]
# prob_cap += x[('Fresno', 90065)] == 1949.761775

### in streamlit should be able to fix a market's demand from a particular warehouse


prob_cap.solve()
#  Print the optimal solution
print("Optimal solution:")
# for i in warehouses:
#     for j in markets:
#         if x[(i,j)].value() != 0:
#             print("x_{}{} = {}".format(i, j, x[(i,j)].value()))
# print("Total cost:", pulp.value(prob_cap.objective))


print("Optimal solution:")
for j in sorted(markets, key=lambda j: -demand["Demand"].loc[j]):
    for i in warehouses:
        if x[(i,j)].value() != 0:
            print("x_{}{} = {}".format(i, j, x[(i,j)].value()))
print("Total cost:", pulp.value(prob_cap.objective))


data_cap = []
for i in warehouses:
    for j in markets:
        if x[(i,j)].value() != 0:
            data_cap.append([i, j, x[(i,j)].value()])
results_cap = pd.DataFrame(data_cap, columns=['Warehouse', 'Market', 'Transport'])
results_cap['Transport'] = results_cap['Transport'].round(3)


# In[81]:


total_cost_cap = {}
for i in warehouses:
    total_cost_cap[i] = 0

# Print the optimal solution and calculate the total cost for each warehouse
print("Optimal solution:")
for i in warehouses:
    for j in markets:
        if x[(i,j)].value() != 0:
 
            total_cost_cap[i] += freight_costs.loc[i,j] * x[(i,j)].value()# + lead_time.loc[i,j] * x[(i,j)].value()

# Print the total cost for each warehouse
for i in warehouses:
    print("Total cost for {}: {}".format(i, total_cost_cap[i]))


# In[82]:


import pandas as pd

# Calculate and print results for each warehouse
results_2 = []
for i in warehouses:
    total_shipped = sum(x[(i,j)].value() for j in markets)
    capacity_left = float(cap.loc[i] - total_shipped)
    percentage_left = 100 * capacity_left / cap.loc[i]
    results_2.append({
        'Warehouse': i,
        'Total shipped': total_shipped,
        'Quantity left in warehouse': int(capacity_left),
        'Percentage left': float(round(percentage_left, 4)),
        'Original capacity': int(cap.loc[i])
    })


# Create DataFrame from results
df_cap = pd.DataFrame(results_2)


# Print total cost
print("Total cost:", pulp.value(prob_cap.objective))
df_cap.set_index('Warehouse', inplace=True)
df_cap['Total Cost'] = df_cap.index.map(lambda w: total_cost_cap[w])
df_cap['per_unit_cost'] = df_cap["Total Cost"]/df_cap["Total shipped"]
df_cap


# In[83]:


def optimization_model_itr_2(freight_costs, lead_time, demand, demand_col, cap):
    '''Build the optimization based on input parameters'''
    # Define Decision Variables
    warehouses
    markets
    route_name = [(i,j) for i in warehouses for j in markets] 

    # Initialize Class
    prob_cap = pulp.LpProblem("Minimize Transportation Costs and Lead Time", pulp.LpMinimize)

    # Create Decision Variables
    x = pulp.LpVariable.dicts("transport", route_name, lowBound=0, cat='Continuous')

    # Define Objective Function
    prob_cap += (pulp.lpSum(freight_costs.loc[i,j] * x[(i,j)] + lead_time.loc[i,j] for i in warehouses for j in markets))

    
    
    # Add Constraints
    for i in warehouses:
        prob_cap += pulp.lpSum(x[(i,j)] for j in markets) <= cap.loc[i]
    for j in markets:
        prob_cap += pulp.lpSum(x[(i,j)] for i in warehouses) == demand.loc[j,demand_col]                                                 
    # Solve Model
    prob_cap.solve()
    # Print the optimal solution
    print("Optimal solution:")
    for i in warehouses:
        for j in markets:
            if x[(i,j)].value() != 0:
                print("x_{}{} = {}".format(i, j, x[(i,j)].value()))
    print("Total cost:", pulp.value(prob_cap.objective))
    
    # Results
    status_out = LpStatus[prob_cap.status]
    objective_out  = pulp.value(prob_cap.objective)
    FC = sum([freight_costs.loc[i,j] * x[(i,j)].varValue for i in warehouses for j in markets])
    return status_out, objective_out,x,FC


# In[84]:


list_scenario_2, list_status_2, list_results_2, list_totald_2, list_FC_2,data_itr_2_mc =[], [], [], [], [], []

for i in range(1, 50): # 0 is the initial scenario 
    # Calculations
    status_out, objective_out, x, FC = optimization_model_itr_2(freight_costs, lead_time, demand_var, i, cap)
    
    list_status_2.append(status_out)
    list_results_2.append(objective_out)
    list_FC_2.append(FC)
    total_demand_2 = demand_var[i].sum()
    list_totald_2.append(total_demand_2)
    list_scenario_2.append(i)
    for i in warehouses:
        for j in markets:
            if x[(i,j)].value() != 0:
                data_itr_2_mc.append({
                    'Warehouse': i,
                    'Market': j
                    "Quantitiy": x[(i,j)].value()
                })

    results_itr_2_mc = pd.DataFrame(data_itr_2_mc, columns=['Warehouse', 'Market'])


# In[85]:


results_itr_2_mc.drop_duplicates(subset=["Warehouse","Market"])


# ## ----------------------------------------------------------------------------------------------------------------------------

# ###  Iteration 3: where from one warehouse complete demand of one market will be fulfilled.

# In[ ]:


# for streamlit use iteration 3


# In[40]:


import pulp

prob_3 = pulp.LpProblem("Minimize Transportation Costs and Lead Time", pulp.LpMinimize)

route_name = [(i,j) for i in warehouses for j in markets] 

x = pulp.LpVariable.dicts("transport", route_name, lowBound=0, cat='Continuous')
# add a binary variable for each combination of warehouse and market
y = pulp.LpVariable.dicts("supply", route_name, lowBound=0, upBound=1, cat='Binary')


prob_3 += (pulp.lpSum(freight_costs.loc[i,j] * x[(i,j)] + lead_time.loc[i,j]  for i in warehouses for j in markets))


# add constraints to link y and x variables
for i in warehouses:
    for j in markets:
        prob_3 += x[(i,j)] <= y[(i,j)] * demand["Demand"].loc[j]
for j in markets:
    prob_3 += pulp.lpSum(y[(i,j)] for i in warehouses) == 1
for i in warehouses:
    prob_3 += pulp.lpSum(freight_costs.loc[i,j] * y[(i,j)] for j in markets) <= cap.loc[i]
for i in warehouses:
    prob_3 += pulp.lpSum(x[(i,j)] for j in markets) <= cap.loc[i]
for j in markets:
    prob_3 += pulp.lpSum(x[(i,j)] for i in warehouses) == demand["Demand"].loc[j]


        
prob_3.solve()
#  Print the optimal solution
print("Optimal solution:")
# for i in warehouses:
#     for j in markets:
#         if x[(i,j)].value() != 0:
#             print("x_{}{} = {}".format(i, j, x[(i,j)].value()))
# print("Total cost:", pulp.value(prob_cap.objective))


# print("Optimal solution:")
for j in sorted(markets, key=lambda j: -demand["Demand"].loc[j]):
    for i in warehouses:
        if x[(i,j)].value() != 0:
            print("x_{}{} = {}".format(i, j, x[(i,j)].value()))
print("Total cost:", pulp.value(prob_3.objective))


data_3 = []
for i in warehouses:
    for j in markets:
        if x[(i,j)].value() != 0:
            data_3.append([i, j, x[(i,j)].value()])
results_3 = pd.DataFrame(data_3, columns=['Warehouse', 'Market', 'Transport'])
results_3['Transport'] = results_3['Transport'].round(3)


# In[ ]:


prob3


# In[36]:


total_cost_3 = {}
for i in warehouses:
    total_cost_3[i] = 0

# Print the optimal solution and calculate the total cost for each warehouse
print("Optimal solution:")
for i in warehouses:
    for j in markets:
        if x[(i,j)].value() != 0:
 
            total_cost_3[i] += freight_costs.loc[i,j] * x[(i,j)].value()# + lead_time.loc[i,j] * x[(i,j)].value()

# Print the total cost for each warehouse
for i in warehouses:
    print("Total cost for {}: {}".format(i, total_cost_3[i]))


# In[37]:


import pandas as pd

# Calculate and print results for each warehouse
results_33 = []
for i in warehouses:
    total_shipped = sum(x[(i,j)].value() for j in markets)
    capacity_left = float(cap.loc[i] - total_shipped)
    percentage_left = 100 * capacity_left / cap.loc[i]
    results_33.append({
        'Warehouse': i,
        'Total shipped': total_shipped,
        'Quantity left in warehouse': int(capacity_left),
        'Percentage left': float(round(percentage_left, 4)),
        'Original capacity': int(cap.loc[i])
    })


# Create DataFrame from results
df_3 = pd.DataFrame(results_33)


# Print total cost
print("Total cost:", pulp.value(prob_3.objective))
df_3.set_index('Warehouse', inplace=True)
df_3['Total Cost'] = df_3.index.map(lambda w: total_cost_3[w])
df_3['per_unit_cost'] = df_3["Total Cost"]/df_3["Total shipped"]
df_3


# In[38]:


prob_3.status


# In[41]:


def optimization_model_itr_3(freight_costs, lead_time, demand, demand_col, cap):
    '''Build the optimization based on input parameters'''
    # Define Decision Variables
    warehouses
    markets
    route_name = [(i,j) for i in warehouses for j in markets] 

    # Initialize Class
    prob_3 = pulp.LpProblem("Minimize Transportation Costs and Lead Time", pulp.LpMinimize)

    # Create Decision Variables
    x = pulp.LpVariable.dicts("transport", route_name, lowBound=0, cat='Continuous')
    y = pulp.LpVariable.dicts("supply", route_name, lowBound=0, upBound=1, cat='Binary')

    # Define Objective Function
    prob_3 += (pulp.lpSum(freight_costs.loc[i,j] * x[(i,j)] + lead_time.loc[i,j] for i in warehouses for j in markets))

    
    
    # Add Constraints
    for i in warehouses:
        for j in markets:
            prob_3 += x[(i,j)] <= y[(i,j)] * demand.loc[j,demand_col] 
    # add constraints to ensure that each market is supplied by only one warehouse
    for j in markets:
        prob_3 += pulp.lpSum(y[(i,j)] for i in warehouses) == 1
#     for i in warehouses:
#         prob_3 += pulp.lpSum(freight_costs.loc[i,j] * y[(i,j)] for j in markets) <= cap.loc[i]
    for i in warehouses:
        prob_3 += pulp.lpSum(x[(i,j)] for j in markets) <= cap.loc[i]
    for j in markets:
        prob_3 += pulp.lpSum(x[(i,j)] for i in warehouses) == demand.loc[j,demand_col] 

    prob_3.solve()
    # Print the optimal solution
    print("Optimal solution:")
    for i in warehouses:
        for j in markets:
            if x[(i,j)].value() != 0:
                print("x_{}{} = {}".format(i, j, x[(i,j)].value()))
    print("Total cost:", pulp.value(prob_3.objective))

    # Results
    status_out = LpStatus[prob_3.status]
    objective_out  = pulp.value(prob_3.objective)
    FC = sum([freight_costs.loc[i,j] * x[(i,j)].varValue for i in warehouses for j in markets])
    return status_out, objective_out,x,FC


# In[44]:


list_scenario_3, list_status_3, list_results_3, list_totald_3, list_FC_3,data_itr_3_mc = [], [], [], [], [], []

for i in range(1, 20):
    # Calculations
    status_out, objective_out, x, FC = optimization_model_itr_3(freight_costs, lead_time, demand_var, i, cap)

    list_status_3.append(status_out)
    list_results_3.append(objective_out)
    list_FC_3.append(FC)
    total_demand_3 = demand_var[i].sum()
    list_totald_3.append(total_demand_3)
    list_scenario_3.append(i)
    
    for i in warehouses:
        for j in markets:
            if x[(i,j)].value() != 0:
                data_itr_3_mc.append({
                    'Warehouse': i,
                    'Market': j
                })

    results_itr_3_mc = pd.DataFrame(data_itr_3_mc, columns=['Warehouse', 'Market'])


# In[45]:


list_status_3


# In[ ]:


# results_3.drop_duplicates(subset=["Warehouse","Market"])


# In[ ]:


# Have to check status for each scenario itr 3 has infeasible status in a few of them must check 


# ## -----------------------------------------------------------------------------------------------------------

# In[ ]:


with pd.ExcelWriter('Route_Mapping.xlsx') as writer:
    results.to_excel(writer, sheet_name='Without Capacity constraint', index=False)
    df.to_excel(writer, sheet_name='cost_no_cap', index=False)
    results_cap.to_excel(writer, sheet_name='With Capacity constraint', index=False)
    df_cap.to_excel(writer, sheet_name='cost_cap', index=False)
    results_3.to_excel(writer, sheet_name='1 market 1 warehouse cap', index=False)
    df_3.to_excel(writer, sheet_name='cost_1market_1warehouse', index=False)


# ## ----------------------------------------------------------------------------------------------------------------------
