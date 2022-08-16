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
    from datetime import datetime, time, timedelta
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install','datetime'])
    from datetime import datetime, time, timedelta
try:
    import json
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'json'])
    import json
def reader():
    '''
    Inputs:
        None
    Output:
        None
    Function:
        Creates the Route of the Current System from Excel Files
    '''
    shuttle_schedule = pd.read_excel("Shuttle Schedule with Targets.xlsx",sheet_name=2)
    print(shuttle_schedule.head())
    shuttles=shuttle_schedule.columns.to_list()[3:]
    stops=shuttle_schedule['Stop'].to_list()
    stops_id=shuttle_schedule['Stop ID'].to_list()
    print(stops)
    print(shuttles)
    zero=datetime(year=1900,month=1,day=1,hour=0,minute=30)
    temp_data=[]
    new_shuttles=[]
    rray=[]
    bol_list=[1]*100
    for j,bus in enumerate(shuttles):
        route=shuttle_schedule[bus].to_list()
        tim=[]
        r=0
        for i, stop in enumerate(route):
            if type(stop) is str:
                stop=stop[:-4]
                t = datetime.strptime(str(stop), '%H:%M')
                tim.append(float((t-zero).seconds))
            elif type(stop) is float:
                r += 1
                tim.append(np.inf)
            else:
                t= datetime.strptime(str(stop)[:-3], '%H:%M')
                tim.append(float((t-zero).seconds))
        temp_data.append(tim)
        rray.append(r)
        try:
            if shuttles[j][0:-2]==shuttles[j-1]:
                bol_list[j]=0
                temp_data[j-1]=temp_data[j-1]+tim
                rray[j-1]=rray[j-1]+rray[j]
            if shuttles[j][0:-2]==shuttles[j-2]:
                bol_list[j]=0
                temp_data[j-2]=temp_data[j-2]+tim
                rray[j-2]=rray[j-2]+rray[j]
        except:
            pass


    actual_r=[]
    actual_temp_data=[]
    for i in range(len(temp_data)):
        if bol_list[i]:
            actual_temp_data.append(temp_data[i])
            new_shuttles.append(shuttles[i])
            actual_r.append(rray[i])

    final_td=[]
    for i, secs in enumerate(actual_temp_data):
        tim=secs
        sorted_route=np.argsort(tim)
        sorted_route = sorted_route[0:-actual_r[i]]
        route_s=[]
        route_names=[]
        for f,j in enumerate(sorted_route):
            print(j)
            k=int(j)
            if k>=(len(stops_id)):
                k=k-(len(stops_id))
                if k>=(len(stops_id)):
                    k=k-(len(stops_id))
            route_s.append(k)
            route_names.append(stops_id[k])
        final_td.append(route_names)
    with open("current_routes", "w") as fp:
        json.dump([new_shuttles]+final_td, fp)



