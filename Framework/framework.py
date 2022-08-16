'''
Authors: Connor McIntee, David Russman, Min Fei
Contributors: Nathan LaSalle, Justin Chen, Karthik Vempati

Date: June 13, 2022


'''



import sys
import subprocess

try:
    import numpy as np
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install','numpy'])
    import numpy as np

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
    import osmnx as ox
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install','osmnx'])
    import osmnx as ox

try:
    import warnings
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install','warnings'])
    import warnings
warnings.filterwarnings("ignore")
from first_stage import first_stage
from reader import reader


ox.config(log_console=True)
ox.__version__
ox.utils.config(timeout=600)

try:
    import plotly.graph_objects as go
    import plotly_express as px
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install','plotly'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'plotly.graph_objects'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'plotly_express'])
    import plotly.graph_objects as go
    import plotly_express as px
ox.config(use_cache=True, log_console=True)
tok='pk.eyJ1IjoiZmZtbTg1MjEiLCJhIjoiY2tvMm55anJtMHZ0dDJ3dGRjbnBia3B5cyJ9.e93W5B85thtMy1R1xXGRVg'
px.set_mapbox_access_token(tok)

#load the map object of Denver
G1 = ox.load_graphml('denver_entire.graphml')
G1 = ox.add_edge_speeds(G1)
G1 = ox.add_edge_travel_times(G1)


class Optimizer():
    def __init__(self,current_system=False):
        self.current_system=current_system
        dist = pd.read_excel("Student Count by Bus Stop.xlsx", sheet_name=3)
        dist = np.array(dist)
        self.dist = dist[1:(dist.shape[0]+1), 2:(dist.shape[1]+1)]
        global dist1
        dist1= self.dist
        self.key=pd.read_excel("Compiled Shuttle Count.xlsx",sheet_name=1)
        self.stop_locations= pd.read_excel("Student Count by Bus Stop.xlsx",sheet_name=0)
        str='current_routes' if self.current_system else 'data'
        with open(str, "r") as fp:
            first_stage = json.load(fp)
        self.data=first_stage

    class Route():
        def __init__(self, schools, route, Optimizer, old_routes=[], load = 0):
            o=Optimizer
            self.route = route
            self.schools = schools
            self.schools_stop_id=[]
            self.load = load
            if not o.current_system:
                for school in self.schools:
                    self.schools_stop_id.append(int(o.key['Stop ID'][o.key['School ID']==school].values[0]))
            self.time = 0
            for i, stop in enumerate(self.route):
                if i != 0:
                    self.time += Optimizer.dist[stop, stop2]
                stop2 = stop
            self.old_route=old_routes

        def recalculate_time(self):
            '''
            Inputs:
                Route Object
            Output:
                None
            Function:
                Recalculates the travel time of the Route
            '''
            self.time = 0
            for i, stop in enumerate(self.route):
                if i != 0:
                    self.time += dist1[stop, stop2]
                stop2 = stop

        def get_old_route_times(self):
            '''
            Inputs:
                Route Object
            Output:
                Time:
                    The time of the old routes that were combined
                    into the Route Object
            '''
            time=0
            for i,r in enumerate(self.old_route):
                time+=r.time
            return time

    def total_time(self):
        '''
        Inputs:
            Optimizer Object
        Output:
            Time:
                The total time of all the Routes in the system
        '''
        time=0
        for i,r in enumerate(self.routes):
            time+=r.time
        return time

    def load_summary(self):
        '''
        Inputs:
            Optimizer Object
        Output:
            loads:
                List of list where each inner list has 3 load metrics and
                there is one inner list per Route in the system
        '''
        loads=[]
        for i, r in enumerate(self.routes):
            num_stops = len(r.route)
            num_schools = len(r.schools)
            route_load = r.load
            load=[]
            load.append(route_load)
            load.append(route_load/num_stops)
            load.append(route_load/num_schools)
            loads.append(load)
        return loads

    def route_capacity(self):
        '''
        Inputs:
            Optimizer Object
        Output:
            capacity:
                A list of a capacity metric for each Route in the system
        '''
        num_routes = len(self.routes)
        capacity = []
        for i in range(num_routes):
            capacity.append([])
        for i, r in enumerate(self.routes):
            load_per_stop = r.load/len(r.route)
            drop_per_school = r.load/len(r.schools)
            school_ids = r.schools_stop_id
            tracker = 0
            for j in r.route:
                if j in school_ids:
                    tracker += load_per_stop - drop_per_school
                    capacity[i].append(tracker)
                else:
                    tracker += load_per_stop
                    capacity[i].append(tracker)
        return capacity

    def summary(self):
        '''
        Inputs:
            Optimizer Object
        Output:
            None
        Function:
            Prints out summary information about the system
        '''
        times=[]
        for i,r in enumerate(self.routes):
            times.append(r.time)
        count=len(times)
        total_time=sum(times)
        mean = total_time / count
        variance = sum([((x - mean) ** 2) for x in times]) / count
        sd = variance ** 0.5
        ud_kids, ud_stops = unsatisfied_demand(self.routes)
        udag_kids, udag_stops = unsatisfied_demand_ac(self.routes)
        print('Number of Routes: ',count)
        print('Total Time:        %.2f minutes' %total_time)
        print('Average Time:      %.2f minutes' %mean)
        #print('Average Deviation: ', sd)
        print('Longest Route:     %.2f minutes' %max(times))
        print('Unmet Demand: ', ud_kids)
        print('Unmet Demand Against Current: ', udag_kids)

    def create_route_objects(self):
        '''
        Inputs:
            Optimizer Object
        Output:
            None
        Function:
            Creates the Route Objects for the system
        '''
        self.routes=[]
        if self.current_system:
            shuttles=self.data.pop(0)
            for i, route in enumerate(self.data):
                self.routes.append(self.Route([shuttles[i]], route, self))
        else:
            for i,s in enumerate(self.data['school id']):
                for j,route in enumerate(self.data['routes'][i]):
                    self.routes.append(self.Route([s],route,self,load = self.data['load_list'][i][j]))


    def greedy_combine_routes(self, Route1, Route2):
        '''
        Inputs:
            Optimizer Object
            Route1:
                First Route Object to be combined
            Route2:
                Second Route Object to be combined
        Output:
            Route3:
                The combined Route Object of Route1 and Route2
        '''
        r1 = Route1.route.copy()
        r2 = Route2.route.copy()
        r3 = []
        for i in range(len(r1) + len(r2)):
            if not r1:
                r3 = r3 + r2
                break
            if not r2:
                r3 = r3 + r1
                break
            if i == 0:
                r3.append(r1.pop(0))
            else:
                if self.dist[r3[-1], r1[0]] >= self.dist[r3[-1], r2[0]]:
                    r3.append(r2.pop(0))
                else:
                    r3.append(r1.pop(0))
        route = []
        for i, s in enumerate(r3):
            if i == 0:
                pass
            else:
                if self.dist[s, old_s] != 0 or old_s == 0:
                    route.append(s)
            old_s = s
        Route3 = self.Route(Route1.schools + Route2.schools, route,self,[Route1,Route2], Route1.load + Route2.load)
        #for i in range(100):
            #if len(Route3.route) > 4:
                #Route3= self.clean_route(Route3)
        return Route3

    def choose_routes(self):
        '''
        Inputs:
            Optimizer Object
        Output:
            ret_value:
                A tuple of two Route Objects that have the highest
                similarity score
        '''
        m=0
        ret_val=(0,0)
        for i,r1 in enumerate(self.routes):
            for j,r2 in enumerate(self.routes):
                if (i != j):
                    m1=self.similarity(r1,r2)
                    if m < m1:
                        m=m1
                        ret_val=(r1,r2)
        return ret_val

    def clean_all(self,n,aggressive=False):
        '''
        Inputs:
            Optimizer Object
            n:
                Number of times to iterate through the Routes and clean them
            aggressive:
                Boolean argument; If True, then the function calls clean_route()
                on each Route, if False, the function calls aggressive_clean_route()
                on each Route
        Output:
            None
        Function:
            Rearranges the orderings of each Route to reduce travel time
        '''
        for j in range(n):
            for i,r in enumerate(self.routes):
                if aggressive: self.routes[i]=self.aggressive_clean_route(r)
                else: self.routes[i]=self.clean_route(r)

    def aggressive_clean_route(self, Route1):
        '''
        Inputs:
            Optimizer Object
            Route1:
                Route Object to be cleaned
        Output:
            Route1:
                Cleaned version of the Route Object
        Functions:
            Takes the first stop in the Route and iteratively adds
            the next closest stop in the Route
        '''
        route = [0]
        for i, s in enumerate(Route1.route):
            if i == 0:
                pass
            else:
                if self.dist[s, old_s] != 0 or old_s == 0:
                    route.append(s)
            old_s = s
        route2=[route.pop(0),route.pop(1)]
        for i in range(len(route)):
            m=np.inf
            s1=0
            for s in route:
                d=self.dist[route2[-1],s]
                if d<m:
                    m=d
                    s1=s
            route.pop(route.index(s1))
            route2.append(s1)
        Route1.route = route2
        Route1.recalculate_time()
        return Route1

    def clean_route(self, Route1):
        '''
        Inputs:
            Optimizer Object
            Route1:
                Route Object to be cleaned
        Output:
            Route1:
                Cleaned version of the Route Object
        Function:
            Looks at every stop and sees if swapping that stop with another
            stop reduces the travel time of the Route
        '''
        route = [0]
        for i, s in enumerate(Route1.route):
            if i == 0:
                pass
            else:
                if self.dist[s, old_s] != 0 or old_s == 0:
                    route.append(s)
            old_s = s
        Route1.route = route
        n = len(Route1.route)
        for i in range(1,n-1):
            for j in range(1,n-1):
                r=route.copy()
                route[i], route[j]=route[j], route[i]
                if self.calc_time(route)>=self.calc_time(r):
                    route=r
        Route1.route = route
        Route1.recalculate_time()
        return Route1

    def calc_time(self,route):
        '''
        Inputs:
            Optimizer Object
            route:
                List of stops for a Route
        Output:
            time:
                The travel time of the stops
        '''
        time=0
        for i, stop in enumerate(route):
            if i != 0:
                if stop == 44:
                    pass
                time += self.dist[stop, stop2]
            stop2 = stop
        return time

    def similarity(self, r1, r2):
        '''
        Inputs:
            Optimizer Object
            r1:
                Route Object
            r2:
                Route Object
        Output:
            similarity:
                Metric for how similar the two Routes are (how well they
                might perform as a combined Route)
        '''
        count=0
        for i, stop in enumerate(r1.route):
            if stop in r2.route: count+=1
        perc=count/(len(r1.route)+len(r2.route))
        return (perc+0.1)/(len(r1.route)+len(r2.route))

    def optimize(self,num_routes):
        '''
        Inputs:
            Optimizer Object
            num_routes:
                Number of Routes to Optimize to
        Output:
            None
        Function:
            Combine Routes until the number of Routes in the system
            equals num_routes
        '''
        n=len(self.routes)
        while n > num_routes:
            (r1,r2) = self.choose_routes()
            r3= self.greedy_combine_routes(r1,r2)
            self.routes.remove(r1)
            self.routes.remove(r2)
            self.routes.append(r3)
            n=len(self.routes)

    def output_system(self, workbook, worksheet='data'):
        '''
        Inputs:
            Optimizer Object
            workbook:
                Name of output file (.xlsx)
            worksheet:
                Name of output sheet (default= data)
        Output:
            None
        Function:
            Outputs summary information about the system to an Excel file
        '''
        times=[]
        for i,r in enumerate(self.routes):
            times.append(r.time)
        count=len(times)
        total_time=sum(times)
        mean = total_time / count
        variance = sum([((x - mean) ** 2) for x in times]) / count
        sd = variance ** 0.5
        ud_kids, ud_stops = unsatisfied_demand(self.routes)
        udag_kids, udag_stops = unsatisfied_demand_ac(self.routes)
        data = [count, total_time, mean, sd, ud_kids, udag_kids]
        dic={'Number of Shuttles':count,
             'Total Time': total_time,
             'Avg Time': mean,
             'SD Time': sd,
             'ud': ud_kids,
             'udAG': udag_kids}
        df = pd.DataFrame(dic,index=[0])
        df.to_excel(workbook, sheet_name=worksheet)


    def output_routes_vrp(self,workbook,worksheet='data'):
        '''
        Inputs:
            Optimizer Object
            workbook:
                Name of output file (.xlsx)
            worksheet:
                Name of output sheet (default= data)
        Output:
            None
        Function:
            Outputs Route information for the new system to an Excel file
        '''
        df=pd.DataFrame(columns=['Schools Served','Route','Time','Max Capacity','Avg Capacity','SD Capacity'])
        dic={
            'Schools Served': [],
            'Route': [],
            'Time': [],
            'Max Capacity': [],
            'Avg Capacity': [],
            'SD Capacity': [],

        }
        capacity = self.route_capacity()
        for i, r in enumerate(self.routes):
            dic['Schools Served']=r.schools
            route=[]
            for j, s in enumerate(r.route):
                if s>0: route.append(self.stop_locations[self.stop_locations['Stop ID'] == s]['Stop Name'].values[0])
            dic['Route']=route
            dic['Max Capacity'] =np.max(capacity[i])
            dic['Avg Capacity'] = np.mean(capacity[i])
            dic['SD Capacity'] = (np.var(capacity[i]))**.5
            dic['Time'] = r.time
            df=df.append(dic,ignore_index=True)
            df.to_excel(workbook, sheet_name = worksheet)

    def output_routes_curr(self, workbook, worksheet='data'):
        '''
        Inputs:
            Optimizer Object
            workbook:
                Name of output file (.xlsx)
            worksheet:
                Name of output sheet (default= data)
        Output:
            None
        Function:
            Outputs summary information about the current system to an Excel file
        '''
        df=pd.DataFrame(columns=['Shuttle','Route','Time','Old Shuttle','Old Shuttle Time'])
        dic={
            'Shuttle': [],
            'Route': [],
            'Time': [],
            'Old Shuttle': [],
            'Old Shuttle Time': [],
        }
        for i, r in enumerate(self.routes):
            dic['Shuttle']=i
            route=[]
            for j, s in enumerate(r.route):
                if s>0: route.append(self.stop_locations[self.stop_locations['Stop ID'] == s]['Stop Name'].values[0])
            dic['Route']=route
            dic['Time'] = r.time
            dic['Old Shuttle'] = r.schools
            dic['Old Shuttle Time'] = r.get_old_route_times()
            df=df.append(dic,ignore_index=True)
            df.to_excel(workbook, sheet_name = worksheet)

    def remove_stop(self,stop):
        '''
        Inputs:
            Optimizer Object
            stop:
                Stop to be removed from the system
        Output:
            None
        Function:
            Removes the stop from the system
        '''
        for i, r in enumerate(self.routes):
            self.remove_stop_in_route( r, stop)

    def remove_stop_in_route(self, r, stop):
        '''
        Inputs:
            Optimizer Object
            r:
                Route Object
            stop:
                Stop to be removed from the system
        Output:
            None
        Function:
            Removes the stop from the Route
        '''
        route = []
        for j, s in enumerate(r.route):
            if s != stop: route.append(s)
        r.route = route
        r.recalculate_time()

    def graph(self,routes):
        '''
        Inputs:
            Optimizer Object
            routes:
                List of Route Objects to be plotted
        Output:
            Ploty Plot Object
        Function:
            Plots the Routes on a Map of Denver
        '''
        if type(routes)!=list:
            routes=[routes]
        combined_df = pd.DataFrame(
            columns=['X_from', 'Y_from', 'X_to', 'Y_to', 'Vehicle_ID', 'Vehicle_Name'])
        dic = {
            'X_from': 0,
            'Y_from': 0,
            'X_to': 0,
            'Y_to': 0,
            'Vehicle_ID': 0,
            'Vehicle_Name': 0,
            'Schools': 'Schools'
        }
        for j,r in enumerate(routes):

            bus = r.schools[0]
            buss=r.route[1:len(r.route)]
            actual_route = []
            dic['Vehicle_Name'] = str(j+1)
            if self.current_system: dic['Vehicle_Name'] = r.schools[0]
            dic['Vehicle_ID'] = j+1
            for i, s in enumerate(buss):
                if i != 0:
                    dat = self.stop_locations[self.stop_locations['Stop ID'] == s]
                    dic['Y_from'] = dic['Y_to']
                    dic['X_from'] = dic['X_to']
                    dic['X_to'] = dat['Longitude'].values[0]
                    dic['Y_to'] = dat['Latitude'].values[0]
                    combined_df = combined_df.append(dic, ignore_index=True)
                else:
                    dat = self.stop_locations[self.stop_locations['Stop ID'] == s]
                    dic['X_to'] = dat['Longitude'].values[0]
                    dic['Y_to'] = dat['Latitude'].values[0]
                old_s = s
            dic['Y_from'] = dic['Y_to']
            dic['X_from'] = dic['X_to']
            combined_df = combined_df.append(dic, ignore_index=True)

        fig = px.scatter_mapbox(combined_df, lon='X_from', lat='Y_from', zoom=13, width=1000, height=800)
        c = {'lon': -104.7906519, 'lat': 39.7846467}
        fig = px.line_mapbox(combined_df, lon='X_from', lat='Y_from', color="Vehicle_Name",
                             text='Vehicle_Name', zoom=12, center=c, height=300, mapbox_style='streets')
        # plot all the stops

        stop_locations = pd.read_excel("Student Count by Bus Stop.xlsx", sheet_name=0)
        long_list = stop_locations["Longitude"].to_list()
        lat_list = stop_locations["Latitude"].to_list()
        txtlist = stop_locations["Stop Name"].to_list()

        fig.add_trace(go.Scattermapbox(
            mode="markers+text",  # include text mode
            lon=long_list,  # list of longitude.
            lat=lat_list,  # list of latitude, similar as above.
            marker={'size': 10, 'color': 'red'},
            text=txtlist))  # include text list

        fig.update_layout(
            width=1000,
            height=1000,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="LightSteelBlue",
        )
        fig.write_html("vis.html")
        return fig.show()

demand_data = pd.read_excel("Student Count by Bus Stop.xlsx",sheet_name=1)
key=pd.read_excel("Compiled Shuttle Count.xlsx",sheet_name=1)

def verify_system(routes):
    '''
    Inputs:
        routes:
            List of Route Objects
    Output:
        Boolean:
            True if system meets all the demand
            False otherwise
    Function:
        Checks to see if the system meets the maximum demand data
    '''
    for i, row in demand_data.iterrows():
        _from=row['Stop ID']
        _to=int(key['Stop ID'][key['School ID']==row['School Number']].values[0])
        stop_to_school=False
        for j, r in enumerate(routes):
            if (_from in r.route) and (_to in r.route):
                if r.route.index(_from)<=(len(r.route) - 1 - r.route[::-1].index(_to)):
                    stop_to_school=True
                    break
        if stop_to_school==False:
            return False
    return True


def verify_against_current(routes):
    '''
    Inputs:
        routes:
            List of Route Objects
    Output:
        Boolean:
            True if system meets all the demand
            False otherwise
    Function:
        Checks to see if the system meets the demand that the
        current system meets
    '''
    curr = Optimizer(current_system=True)
    curr.create_route_objects()
    for i, row in demand_data.iterrows():
        _from = row['Stop ID']
        _to = int(key['Stop ID'][key['School ID'] == row['School Number']].values[0])
        stop_to_school = False
        for j, r in enumerate(routes):
            if (_from in r.route) and (_to in r.route):
                if r.route.index(_from) <= (len(r.route) - 1 - r.route[::-1].index(_to)):
                    stop_to_school = True
                    break
        if not stop_to_school:
            stop_to_school2 = False
            for j, r in enumerate(curr.routes):
                if (_from in r.route) and (_to in r.route):
                    if r.route.index(_from) <= (len(r.route) - 1 - r.route[::-1].index(_to)):
                        stop_to_school2 = True
                        break
            if stop_to_school2: return False
    return True


def unsatisfied_demand(routes):
    '''
    Inputs:
        routes:
            List of Route Objects
    Output:
        (Students,Stops):
            Tuple of how many Students were left at how many stops
    Function:
        Checks each stop to school pair in the maximum demand data
        and if the demand for the pair is not met, the function
        adds the number of students to Students and adds 1 to Stops
    '''
    kids = 0
    stops = 0
    for i, row in demand_data.iterrows():
        _from = row['Stop ID']
        _to = int(key['Stop ID'][key['School ID'] == row['School Number']].values[0])
        stop_to_school = False
        for j, r in enumerate(routes):
            if (_from in r.route) and (_to in r.route):
                if r.route.index(_from) <= (len(r.route) - 1 - r.route[::-1].index(_to)):
                    stop_to_school = True
                    break
        if stop_to_school == False:
            kids += row['Student Count']
            stops+=1
    return kids, stops

def unsatisfied_demand_ac(routes):
    '''
    Inputs:
        routes:
            List of Route Objects
    Output:
        (Students,Stops):
            Tuple of how many Students were left at how many stops
    Function:
        Checks each stop to school pair in the maximum demand data
        that the current system supports, and if the demand for the
        pair is not met, the function adds the number of students to
        Students and adds 1 to Stops
    '''
    curr = Optimizer(current_system=True)
    curr.create_route_objects()
    kids = 0
    stops = 0
    for i, row in demand_data.iterrows():
        _from = row['Stop ID']
        _to = int(key['Stop ID'][key['School ID'] == row['School Number']].values[0])
        stop_to_school = False
        for j, r in enumerate(routes):
            if (_from in r.route) and (_to in r.route):
                if r.route.index(_from) <= (len(r.route) - 1 - r.route[::-1].index(_to)):
                    stop_to_school = True
                    break
        if not stop_to_school:
            stop_to_school2 = False
            for j, r in enumerate(curr.routes):
                if (_from in r.route) and (_to in r.route):
                    if r.route.index(_from) <= (len(r.route) - 1 - r.route[::-1].index(_to)):
                        stop_to_school2 = True
                        break
            if stop_to_school2:
                kids += row['Student Count']
                stops += 1
    return kids, stops

new=Optimizer()
new.create_route_objects()
new.graph(new.routes)