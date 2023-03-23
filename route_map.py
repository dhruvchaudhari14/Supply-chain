import warnings


pip! install pulp
import pandas as pd
import numpy as np
from pulp import *
import matplotlib.pyplot as plt
import random
random.seed(1447)
import pulp
import streamlit as st
warnings.filterwarnings("ignore")


pd.set_option('display.max_colwidth', 0)
pd.set_option('display.max_columns', None)
pd.options.display.max_seq_items = 2000

import streamlit as st

# set the background color to red
st.set_page_config(page_title="Route Mapping App", page_icon=":truck:", layout="wide", initial_sidebar_state="expanded")
from PIL import Image

image = Image.open('logo.jpg')
st.sidebar.image(image, use_column_width=True)
# st.image(image)

import streamlit as st
# background-image: url("https://www.course5i.com/wp-content/themes/course5iTheme/images/C5-Logo.svg");
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-color: #FFFFFF; /* Set background color to light blue */
             
             background-attachment: fixed;
             background-size: 100% 100%;
         }}

         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()


# ----------------------------------------------------------------------------------------------------------------------------------------
st.sidebar.title("Upload data")

# File uploader widget
file = st.sidebar.file_uploader("Upload the data file", type=["xlsx", "xls"])
sheet_names = []

tabs = ["Data","Optimal Solution without capacity constraints.", "Optimal/Sub-Optimal Solution with capacity constraints."]
tab = st.sidebar.radio("Select output type:", tabs)

if file is not None:
    # Read Excel file
        excel_data = pd.ExcelFile(file)
        sheet_names = excel_data.sheet_names
        st.sidebar.write("Sheet Names in the file:", sheet_names)

        if sheet_names is not None:
            transport_costs = pd.read_excel(file, sheet_name=sheet_names[0], index_col=0)
            cap = pd.read_excel(file, sheet_name=sheet_names[1], index_col=0)
            demand = pd.read_excel(file, sheet_name=sheet_names[2], index_col=0)
            lead_time = pd.read_excel(file, sheet_name=sheet_names[3], index_col=0)
            fixed_data = pd.read_excel(file, sheet_name=sheet_names[4])
            markets = demand.index.tolist()
            warehouses = cap.index.tolist()
        if tab == "Data":
            st.write("This is the data section where you can view all the data used by the optimization algorithms.")
            show_data = st.checkbox("Show Transportation Costs",key = "transport")
            if show_data:
                st.header("Transport Costs")
                st.write("This shows transportation costs from warehouse to markets in $")
                st.write(transport_costs)
            show_data_cap = st.checkbox("Show Capacity for Warehouses",key = "wh")            
            if show_data_cap:
                st.header("Capacity for Warehouses")
                st.write("This shows capacity of each warehouse in units/month.")
                st.write(cap)
            show_data_demand = st.checkbox("Show Demand Data",key = "demand")
            if show_data_demand:
                st.header("Demand Data")
                st.write("This shows Demand of each market in a week.")
                st.write(demand)
            show_data_lt = st.checkbox("Show Lead Time Data",key = "lt")
            if show_data_lt:
                st.header("Lead Time data")
                st.write("This shows time in days from warehouse to market to transport goods.")
                st.write(lead_time)
                
            show_data_f_m_c = st.checkbox("Show Fixed Warehouse to Market Link",key = "fixed_data")
            if show_data_f_m_c:
                st.header("Fixed Warehouse to Market Link")
                st.write("This shows if there are any warehouses assigned to specific markets.")
                st.write(fixed_data)
        

# file = st.sidebar.file_uploader("Upload a file for transportation costs", type=["xlsx", "xls"])
# show_data = st.sidebar.checkbox("Show Transportation Costs",key = "transport")
# if file is not None:
#     try:
#         # Read Excel file
#         transport_costs = pd.read_excel(file,index_col = 0)
        
        
#         if show_data:
#             st.header("Transport Costs")
#             st.write(transport_costs)

        
#     except Exception as e:
#         # Show error message if there is an issue with file reading
#         st.error("Failed to load file. Error message: {}".format(str(e)))
        
        
        
# file = st.sidebar.file_uploader("Upload a file for warehouse capacity", type=["xlsx", "xls"])
# show_data = st.sidebar.checkbox("Show Warehouse Capacity",key = "cap")

# if file is not None:
#     try:
#         # Read Excel file
#         cap = pd.read_excel(file,index_col = 0)
#         warehouses = cap.index.tolist()

#         # Show the data in a table
#         if show_data:
#             st.header("Capacity for warehouses")
#             st.write(cap)        
#     except Exception as e:
#         # Show error message if there is an issue with file reading
#         st.error("Failed to load file. Error message: {}".format(str(e)))

        
        
# file = st.sidebar.file_uploader("Upload a file for Market demand", type=["xlsx", "xls"])
# show_data = st.sidebar.checkbox("Show Demand data",key = "demand")


# if file is not None:
#     try:
#         # Read Excel file
#         demand = pd.read_excel(file,index_col = 0)
#         markets = demand.index.tolist()

#         # Show the data in a table
#         if show_data:
#             st.header("Demand Data")
#             st.write(demand)
        
#     except Exception as e:
#         # Show error message if there is an issue with file reading
#         st.error("Failed to load file. Error message: {}".format(str(e)))        
        
        
        
        
        
# file = st.sidebar.file_uploader("Upload a file for Lead Time", type=["xlsx", "xls"])
# show_data = st.sidebar.checkbox("Show Lead Time data",key = "LT")

# if file is not None:
#     try:
#         # Read Excel file
#         lead_time = pd.read_excel(file,index_col = 0)

#         # Show the data in a table
        
#         if show_data:
#             st.header("Lead Time Data")
#             st.write(lead_time)
        
#     except Exception as e:
#         # Show error message if there is an issue with file reading
#         st.error("Failed to load file. Error message: {}".format(str(e)))     

       
        
        
# file = st.sidebar.file_uploader("Upload a file for fixed warehouses to markets data", type=["xlsx", "xls"],key="fixed")
# show_data = st.sidebar.checkbox("Show data",key = "fix")
# show_fixed_data = st.sidebar.checkbox("Enable Fixed Warehouse-Market data",key = "fixed_data")
# if file is not None:
#     try:
#         # Read Excel file
#         fixed_data = pd.read_excel(file)

#         # Show the data in a table
        
#         if show_data:
#             st.header("Fixed Warehouse to Market Data")
#             st.write(fixed_data)
        
#     except Exception as e:
#         # Show error message if there is an issue with file reading
#         st.error("Failed to load file. Error message: {}".format(str(e)))     


# ----------------------------------------------------------------------------------------------------------------------------------------


### Iteration 1:without capacity constraint
if tab == "Optimal Solution without capacity constraints.":
    st.write("In this section the optimization algorithm uses no constraint on warehouse capacity.")
    
    show_fixed_data = st.checkbox("Consider Fixed Warehouse-Market Link",key = "fixed_data_f_m_c_1")
    st.write("By using this respective markets will only get demand fulfilled by a fixed warehouse.")             
    run_code = st.checkbox("Run Iteration 1: Without Capacity Constraints")

    if run_code:
        prob = pulp.LpProblem("Minimize Transportation Costs and Lead Time", pulp.LpMinimize)

        route_name = [(i,j) for i in warehouses for j in markets] 

        x = pulp.LpVariable.dicts("transport", route_name, lowBound=0, cat='Continuous')


        prob += (pulp.lpSum(transport_costs.loc[i,j] * x[(i,j)] + lead_time.loc[i,j]  for i in warehouses for j in markets))

        if show_fixed_data:

            if fixed_data is not None:
                if {'Warehouse', 'Fixed_market'}.issubset(fixed_data.columns):
                    for index, row in fixed_data.iterrows():
                        fixed_WH = row['Warehouse']
                        fixed_market = row['Fixed_market']
                        prob += x[(fixed_WH, fixed_market)] == demand["Demand"].loc[fixed_market]
                else:
                    st.write("Error: CSV file does not have the expected column names.")
            else:
                fixed_WH = None
                fixed_market = None        



        for j in markets:
            prob += pulp.lpSum(x[(i,j)] for i in warehouses) == demand["Demand"].loc[j]


        prob.solve()


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
        results = pd.DataFrame(data, columns=['Warehouse', 'Route', 'Demand Fulfilled'])
        results['Demand Fulfilled'] = results['Demand Fulfilled'].round(0)
        st.write("Optimal Cost required: ",round(pulp.value(prob.objective),2))
        st.write("Status: ",LpStatus[prob.status])
        st.header("Optimal Warehouse to Market mapping")


        total_cost = {}
        for i in warehouses:
            total_cost[i] = 0

        for i in warehouses:
            for j in markets:
                if x[(i,j)].value() != 0:

                    total_cost[i] += int(transport_costs.loc[i,j] * x[(i,j)].value())
        #     + lead_time.loc[i,j] * x[(i,j)].value()

        # Calculate and print results for each warehouse
        results_1 = []
        for i in warehouses:
            total_shipped = sum(x[(i,j)].value() for j in markets)
            capacity_left = float(cap.loc[i] - total_shipped)
            percentage_left = 100 * capacity_left / cap.loc[i]
            results_1.append({
                'Warehouse': i,
                'Capacity utilized': int(total_shipped),
                'Remaining Capacity': int(capacity_left),
                'Remaining Capacity(%)': float(round(percentage_left, 2))
            })

        # Create DataFrame from results
        df = pd.DataFrame(results_1)

        # Print total cost
        print("Total cost:", pulp.value(prob.objective))
        df.set_index('Warehouse', inplace=True)
        df['Total Cost'] = df.index.map(lambda w: total_cost[w])

        cols = st.columns(2)

        cols[0].write("Optimal Warehouse to Market routes and quantity")
        cols[0].write(results)
        cols[1].write("Additional Data")
        cols[1].write(df)
        
    if st.checkbox("Export data to Excel"):
        st.write("Please add .xlsx extension")
        filename = st.text_input("Enter a filename","")
            
            
        if filename:
            
                # Export the data to Excel
            results.to_excel(filename, index=False)

                # Display a success message
            st.success(f"Data exported to {filename}!")




# # ## ----------------------------------------------------------------------------------------------------------------------------

###  Iteration 3: where from one warehouse complete demand of one market will be fulfilled.
if tab == "Optimal/Sub-Optimal Solution with capacity constraints.":
    st.write("In this section the optimization algorithm has capacity constraints along with the condition that demand for a market will be               fulfilled by on only one warehouse.")
   
    show_fixed_data_1 = st.checkbox("Consider Fixed Warehouse-Market Link",key = "fixed_data_f_m_c_2")
    st.write("By using this respective markets will only get demand fulfilled by a fixed warehouse.") 

    run_code = st.checkbox("Run Iteration 3: Demand from a market is fulfilled by only one warehouse with capacity constraints")

    if run_code:

        prob_3 = pulp.LpProblem("Minimize Transportation Costs and Lead Time", pulp.LpMinimize)

        route_name = [(i,j) for i in warehouses for j in markets] 

        x = pulp.LpVariable.dicts("transport", route_name, lowBound=0, cat='Continuous')
        # add a binary variable for each combination of warehouse and market
        y = pulp.LpVariable.dicts("supply", route_name, lowBound=0, upBound=1, cat='Binary')


        prob_3 += (pulp.lpSum(transport_costs.loc[i,j] * x[(i,j)] + lead_time.loc[i,j]  for i in warehouses for j in markets))

        if show_fixed_data_1:

            if fixed_data is not None:
                if {'Warehouse', 'Fixed_market'}.issubset(fixed_data.columns):
                    for index, row in fixed_data.iterrows():
                        fixed_WH = row['Warehouse']
                        fixed_market = row['Fixed_market']
                        prob_3 += x[(fixed_WH, fixed_market)] == demand["Demand"].loc[fixed_market]
                else:
                    st.write("Error: CSV file does not have the expected column names.")
            else:
                fixed_WH = None
                fixed_market = None

        # add constraints to link y and x variables
        for i in warehouses:
            for j in markets:
                prob_3 += x[(i,j)] <= y[(i,j)] * demand["Demand"].loc[j]
        for j in markets:
            prob_3 += pulp.lpSum(y[(i,j)] for i in warehouses) == 1
        for i in warehouses:
            prob_3 += pulp.lpSum(transport_costs.loc[i,j] * y[(i,j)] for j in markets) <= cap.loc[i]
        for i in warehouses:
            prob_3 += pulp.lpSum(x[(i,j)] for j in markets) <= cap.loc[i]
        for j in markets:
            prob_3 += pulp.lpSum(x[(i,j)] for i in warehouses) == demand["Demand"].loc[j]



        prob_3.solve()
        #  Print the optimal solution
        st.write("Optimal Cost required: ",round(pulp.value(prob_3.objective),2))
        st.write("Status: ",LpStatus[prob_3.status])


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
        results_3 = pd.DataFrame(data_3, columns=['Warehouse', 'Route', 'Demand Fulfilled'])
        results_3['Demand Fulfilled'] = results_3['Demand Fulfilled'].round(0)
        st.header("Optimal/Sub-Optimal Warehouse to Market mapping")
        

        total_cost_3 = {}
        for i in warehouses:
            total_cost_3[i] = 0

        for i in warehouses:
            for j in markets:
                if x[(i,j)].value() != 0:

                    total_cost_3[i] += int(transport_costs.loc[i,j] * x[(i,j)].value())# + lead_time.loc[i,j] * x[(i,j)].value()


        # Calculate and print results for each warehouse
        results_33 = []
        for i in warehouses:
            total_shipped = sum(x[(i,j)].value() for j in markets)
            capacity_left = float(cap.loc[i] - total_shipped)
            percentage_left = 100 * capacity_left / cap.loc[i]
            results_33.append({
                'Warehouse': i,
                'Capacity utilized': int(round(total_shipped, 0)),
                'Remaining Capacity': int(capacity_left),
                'Remaining Capacity(%)': float(round(percentage_left, 2)),
            })


        # Create DataFrame from results
        df_3 = pd.DataFrame(results_33)


        # Print total cost
        print("Total cost:", pulp.value(prob_3.objective))
        df_3.set_index('Warehouse', inplace=True)
        df_3['Total Cost'] = df_3.index.map(lambda w: total_cost_3[w])
        cols_2 = st.columns(2)

        cols_2[0].write("Optimal Warehouse to Market routes and quantity")
        cols_2[0].write(results_3)
        cols_2[1].write("Additional Data")
        cols_2[1].write(df_3)
        
    if st.checkbox("Export data to Excel"):
        st.write("Please add .xlsx extension")
        filename = st.text_input("Enter a filename","")
            
            
        if filename:
            
                # Export the data to Excel
            results_3.to_excel(filename, index=False)

                # Display a success message
            st.success(f"Data exported to {filename}!")


# ------------------------------------------------------------------------------------------------------------------------------------------

# # In[ ]:


# with pd.ExcelWriter('Route_Mapping.xlsx') as writer:
#     results.to_excel(writer, sheet_name='Without Capacity constraint', index=False)
#     df.to_excel(writer, sheet_name='cost_no_cap', index=False)
#     results_cap.to_excel(writer, sheet_name='With Capacity constraint', index=False)
#     df_cap.to_excel(writer, sheet_name='cost_cap', index=False)
#     results_3.to_excel(writer, sheet_name='1 market 1 warehouse cap', index=False)
#     df_3.to_excel(writer, sheet_name='cost_1market_1warehouse', index=False)


# -------------------------------------------------------------------------------------------------------------------------------------------
