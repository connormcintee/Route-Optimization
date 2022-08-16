import sys
import subprocess

import VRP as v
try:
    import pandas as pd
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install','pandas'])
    import pandas as pd

try:
    import json
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install','json'])
    import json
try:
    import warnings
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install','warnings'])
    import warnings
warnings.filterwarnings("ignore")

def first_stage(t=5):
    '''
    Inputs:
        t:
            Number of seconds to run the optimization for each school
    Output:
        None
    Function:
        Outputs the first stage optimization data to data.json
    '''
    ##gets the time matrix (FNE)
    #dist = pd.read_excel("Student Count by Bus Stop (1.16.2020).xlsx",sheet_name=3)
    dist = pd.read_excel("Student Count by Bus Stop.xlsx",sheet_name=3)
    n=len(list(dist.columns))-2
    ##gets the max demand table
    #demand_data = pd.read_excel("Student Count by Bus Stop (1.16.2020).xlsx",sheet_name=1)
    demand_data = pd.read_excel("Student Count by Bus Stop.xlsx",sheet_name=1)
    for i, row in demand_data.iterrows():
        students=row['Student Count']
        if students>110:
            row['Student Count']=55
            demand_data.at[i,'Student Count']-=55
            demand_data.append(row)
        elif students>55:
            row['Student Count']=55
            demand_data.at[i, 'Student Count'] -= 55
            demand_data.append(row)
    #creates the data frame to put the solutions in
    final_data=pd.DataFrame(columns = ["school id","routes", "total_distance", "num_vehicle_used", "route_len_list", "load_list", "WC_load_list"])
    key=pd.read_excel("Compiled Shuttle Count.xlsx",sheet_name=1)
    #solve the probelm for each school
    #print(id)
    final_dic = {
                "school id": [],
                "routes": [],
                "total_distance": [],
                "num_vehicle_used":[],
                "route_len_list": [],
                "load_list": [],
                "WC_load_list": []
            }
    a=demand_data["School Number"].unique()
    for id in a:
        schools_stop_id=int(key['Stop ID'][key['School ID']==id].values[0])
        school_dist=dist.iloc[schools_stop_id+1].values.tolist()[2:(n+3)]
        print(id)
        #get demand for each stop for the school
        demand_table=demand_data[demand_data["School Number"]==id]
        #creates empty demand matrix
        demand=[]
        #fills the demand list
        for i in range(1,(n+1)):
            try: demand.append(int(demand_table[demand_table["Stop ID"] == i]["Student Count"]))
            except: demand.append(0)
        demand.append(0)
        #creates empty distance matrix
        distance=[]
        #fills the distance matrix
        for i in range(1,(n+1)):
            distance.append(dist.iloc[i].values.tolist()[2:(n+2)]+[school_dist[i-1]])
        distance.append(school_dist+[0])
        #calls the VRP function
        routes2, total_distance, num_vehicle_used, route_len_list2, load_list2, WC_load_list2=v.main(dist_matrix=distance, demand=demand, demandWC=[0] * 46, num_bus=100, capacities=[72] * 100, capacitiesWC=[60] * 100, penalty=1000, ortime=t)
        routes=[]
        route_len_list=[]
        load_list=[]
        WC_load_list=[]
        for i, route in enumerate(routes2):
            route[-1]=schools_stop_id
            if len(route)!=2:
                routes.append(route)
                route_len_list.append(route_len_list2[i])
                load_list.append(load_list2[i])
                WC_load_list.append(WC_load_list2[i])
        print(routes)
        print("Number of Buses: ",num_vehicle_used)
        if num_vehicle_used:
            dic = {
                "school id": id,
                "routes": routes,
                "total_distance":total_distance,
                "num_vehicle_used":num_vehicle_used,
                "route_len_list":route_len_list,
                "load_list":load_list,
                "WC_load_list":WC_load_list
            }
            routes=[list(map(int, route)) for route in routes]
            final_dic["school id"].append(int(id))
            final_dic["routes"].append(routes)
            final_dic["total_distance"].append(float(total_distance))
            final_dic["num_vehicle_used"].append(int(num_vehicle_used))
            final_dic["route_len_list"].append(list(map(int, route_len_list)))
            final_dic["load_list"].append(list(map(int, load_list)))
            final_dic["WC_load_list"].append(list(map(int, WC_load_list)))
            final_data=final_data.append(dic,ignore_index=True)

    with open("data", "w") as fp:
        json.dump(final_dic, fp)
