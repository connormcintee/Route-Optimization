import numpy as np
import random
import pandas as pd
from math import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy.spatial.distance import cdist
from datetime import date
import gurobipy as gp
from gurobipy import *
import time
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

class Assigner_stu_to_sch(object):
    '''
    Assign each student to a school
    '''
    def __init__(self, students,schools, min_living_distance):
        self.students = students
        self.schools = schools
        self.min_living_distance=min_living_distance
        #self.boundary_students=self.assign_boundary_students2()
    
    def assign_boundary_students(self): #assign boundary student to boundary schools
        stu=self.students
        sch=self.schools
        boundary_stu=stu.loc[stu['type']=='boundary']
        boundary_sch=sch.loc[sch['type']=='boundary']
        if min(len(boundary_stu),len(boundary_sch))<1:
            return None
        boundary_stu['point_X'] = [(x, y)[0] for x, y in zip(boundary_stu['X'], boundary_stu['Y'])]
        boundary_stu['point_Y'] = [(x, y)[1] for x, y in zip(boundary_stu['X'], boundary_stu['Y'])]
        boundary_sch['point_X'] = [(x, y)[0] for x, y in zip(boundary_sch['X'], boundary_sch['Y'])]
        boundary_sch['point_Y'] = [(x, y)[1] for x, y in zip(boundary_sch['X'], boundary_sch['Y'])]
        boundary_stu['shortest_distance'] = 'NA'
        boundary_stu['closest']='NA'
        boundary_stu['closest'] = boundary_stu['closest'].astype(object)
        for id in boundary_stu['ID'].values.tolist():
            #print('~~~~ student ID {}~~~~~'.format(id))
            search = 100
            stu_x=boundary_stu.loc[boundary_stu['ID']==id,'X'].values[0]
            stu_y = boundary_stu.loc[boundary_stu['ID'] == id, 'Y'].values[0]
            boundary_sch1 = boundary_sch.loc[boundary_sch['X'].between(stu_x - search, stu_x + search)
                                             | boundary_sch['Y'].between(stu_y - search,stu_y + search)]
            num_candidates = len(boundary_sch1)
            # print('num_candidates',num_candidates)
            while num_candidates < 1:
                search += 200
                boundary_sch1 = boundary_sch.loc[
                    boundary_sch['X'].between(stu_x - search, stu_x + search) | boundary_sch['Y'].between(
                        stu_y - search, stu_y + search)]
                num_candidates = len(boundary_sch1)
                #print('num_candidates', num_candidates)
            (the_closest_point_X,the_closest_point_Y, the_closest_distance)= self.closest_point( (stu_x,stu_y), list((x,y) for x, y in zip(boundary_sch['X'], boundary_sch['Y'])))
            boundary_stu.loc[boundary_stu['ID'] == id, 'shortest_distance'] = the_closest_distance
            boundary_stu.loc[boundary_stu['ID'] == id, 'closest_X'] = the_closest_point_X
            boundary_stu.loc[boundary_stu['ID'] == id, 'closest_Y'] = the_closest_point_Y
        temp = boundary_sch[['point_X', 'point_Y', 'SchoolID']]  # .rename(columns={'point_X': 'closest_X','point_Y': 'closest_Y'})
        assigned_boundary_stu = pd.merge(boundary_stu, temp, how='left', left_on=['closest_X', 'closest_Y'],
                                         right_on=['point_X', 'point_Y'])
        return assigned_boundary_stu[['ID','X','Y','type','enroll_year','grad_year','sub_ID','shortest_distance','SchoolID']]

    def assign_charter_students(self,randomess_for_charter=0):
        #print('----------assigning charter student----------')
        stu = self.students
        sch = self.schools
        charter_stu = stu.loc[stu['type'] == 'charter']
        charter_sch = sch.loc[sch['type'] == 'charter']
        charter_sch['closest'] = [(x, y) for x, y in zip(charter_sch['X'], charter_sch['Y'])]
        charter_stu['point'] = [(x, y) for x, y in zip(charter_stu['X'], charter_stu['Y'])]
        len1=len(charter_stu)
        len2=len(charter_sch)
        if min(len1,len2)<1:
            return None
        charter_sch_ids=charter_sch['SchoolID'].tolist()
        if randomess_for_charter==0:
            assignment = random.choices(charter_sch_ids, k=len1)
            charter_stu['SchoolID']=assignment
            temp=charter_sch[['SchoolID','closest','X','Y']]
            temp.rename(columns={'X': 'SchoolX','Y':'SchoolY'},inplace='True')
        combine=charter_stu.merge(temp, on='SchoolID')
        combine['shortest_distance']=abs(combine['X']-combine['SchoolX'])+abs(combine['Y']-combine['SchoolY'])
        combine.drop(['SchoolX','SchoolY'], axis=1,inplace=True)
        return combine[['ID','X','Y','type','enroll_year','grad_year','sub_ID','shortest_distance','SchoolID']]

    def assign_all_students(self):
        df1 = self.assign_boundary_students()
        df2 = self.assign_charter_students()
        df = pd.concat([df1, df2],ignore_index=True)
        df.sort_values(by=['ID', 'sub_ID'], inplace=True)
        df = df.reset_index()
        df.drop(['index'], axis=1,inplace=True)
        df.to_csv('total_students_assigned_date {}.csv'.format(date.today()), index=False)
        return df

    def closest_point(self, point, points):
        """ Find closest point from a list of points. """
        cdist_ = cdist([point], points, 'minkowski', p=1)
        arg_min = cdist_.argmin()
        min_ = cdist_.min()
        return (points[arg_min][0], points[arg_min][1],min_)
 
    def students_close_removed(self):
        '''
        input all students dataframe
        :return: students without too-close, dataframe write to csv file
        '''
        students=self.assign_all_students()
        students_far=students.loc[students['shortest_distance']>self.min_living_distance]
        students_far.to_csv('total_far_students_assigned.csv'.format(date.today()), index=False)
        return students_far

    def get_far_enrollment_over_years(self):
        stu_df=self.students_close_removed()
        stu_enroll_variation = []
        num_year = max(stu_df['enroll_year'].values)
        for i in range(7, num_year + 1):
            ss = get_active_stu(stu_df, i)
            stu_enroll_variation.append(len(ss))
        # plt.figure(figsize=(20, 20))
        plt.plot(range(1, 1 + len(stu_enroll_variation)), stu_enroll_variation, marker='o')
        plt.xticks(range(1, 1 + len(stu_enroll_variation)))
        plt.title('Number of total students in district across years')
        plt.xlabel('year')
        plt.ylabel('enrollment')
        plt.savefig('YOY enrollment variation.png')
        plt.show()

def get_active_stu(stu_df,year):
    active_stu_Df=stu_df[(stu_df['enroll_year']<=year)&(stu_df['grad_year']>=year)]
    return active_stu_Df

def get_enrollment_over_years(stu_df):
    stu_enroll_variation = []
    num_year=max(stu_df['enroll_year'].values)
    for i in range(7, num_year + 1):
        ss = get_active_stu(stu_df, i)
        stu_enroll_variation.append(len(ss))
    return stu_enroll_variation

class assigner_students_to_stops(object):

    '''
    Assign students to stops
    '''
    def __init__(self,active_students, potential_stops,max_walking_distance):
        self.active_students=active_students
        self.potential_stops=potential_stops
        self.max_walking_distance=max_walking_distance
    def calculate_covering_matrix(self):
        students=self.active_students
        stops=self.potential_stops
        max_walking_distance = self.max_walking_distance

        XA = students[['X', 'Y']].to_numpy(dtype='float')
        XB = stops[['X', 'Y']].to_numpy(dtype='float')
        Y = cdist(XA, XB, metric='minkowski', p=1.)  # 8seconds
        YY=pd.DataFrame(Y)
        YY.to_hdf(r'distance_matrix{}.h5'.format(date.today()), key='stage', mode='w')  # 6G

        Z = np.zeros(np.shape(Y))
        Z[Y < max_walking_distance] = 1  # 1 minute
        ZZ=pd.DataFrame(Z)
        ZZ.to_hdf(r'binary_covering_matrix{}.h5'.format(date.today()), key='stage',
                                mode='w')  # 6G,8seconds
        return ZZ, YY

# formulation jointly
def open_stop_per_sch(binary_covering_array,active_stu):
    '''
    :param covering_matrix: array, not dataframe. active students & potential stops
    :return: number of stops opened, and index of open stops within given matrix
    '''
    #in every year, pass different stu_df and different stops
    students_by_school=active_stu.groupby(['SchoolID']).groups
    start_time1 = time.clock()
    m=Model('stop_selection_per_sch')
    m.Params.LogToConsole = 0
    # m.Params.OutputFlag=0
    # logfile = open('mip1 %s.log' % (122), 'w')
    m.params.LogFile = 'm2.log'
    #print('initiating gurobi model')
    X={}
    for j in range(np.shape(binary_covering_array)[1]):
        for s in students_by_school.keys():
            X[j,s]=m.addVar(lb=0,ub=1,vtype = GRB.BINARY,name="X_%s_%s"%(j,s))
    #X=m.addVars(range(len(np.shape(binary_covering_array)[1])),range(len(students_by_school)),lb=0,ub=1,vtype = GRB.BINARY,name="x_js")
    for s in students_by_school.keys():
        covering_matrix=binary_covering_array[students_by_school[s],:]
        cover_rows = [np.nonzero(t)[0] for t in covering_matrix]
        for i in range(len(cover_rows)):
            m.addConstr(quicksum(X[j,s] for j in cover_rows[i]) >= 1)
    obj=gp.quicksum(X[j,s] for j in range(np.shape(binary_covering_array)[1]) for s in students_by_school.keys()  )
    m.setObjective(obj, GRB.MINIMIZE)
    m.Params.MIPGap = 0.001  # 0.1%
    m.Params.TimeLimit = 60*3  # 3 minutes
    run_time1 = time.clock() - start_time1
    start_time2 = time.clock()
    m.optimize()
    run_time2 = time.clock() - start_time2
    #m.write("out.lp")
    #print (m.display())
    objective = m.getObjective()
    objective_value=objective.getValue()
    active_stops_index=[(j,s) for j in range(np.shape(binary_covering_array)[1]) for s in students_by_school.keys() if X[j,s].x > 0.999]
    num_stops_opened_per_sch=len(active_stops_index)
    num_combined_stop_opened=0
    for j in range(np.shape(binary_covering_array)[1]):
        if any( [X[j,s].x>0.9999 for s  in students_by_school.keys()]):
            num_combined_stop_opened+=1
    return objective_value,active_stops_index,num_stops_opened_per_sch,num_combined_stop_opened,run_time1,run_time2


#formulation seprately
def open_stop(binary_covering_array,active_stu):
    students_by_school=active_stu.groupby(['SchoolID']).groups
    #schoolid=active_stu['SchoolID'].unique()[0]
    start_time = time.clock()
    m=Model('stop_selection')
    m.Params.LogToConsole = 0
    #m.Params.OutputFlag=0
    #logfile = open('mip1 %s.log' % (122), 'w')
    #m.params.LogFile='m1.log'
    #print('initiating gurobi model')
    X=m.addVars(np.shape(binary_covering_array)[1],lb=0,ub=1,vtype = GRB.BINARY,name="x_")
    covering_matrix=binary_covering_array[active_stu.index]
    cover_rows = [np.nonzero(t)[0] for t in covering_matrix]
    for i in range(len(cover_rows)):
        m.addConstr(quicksum(X[j] for j in cover_rows[i]) >= 1)
    obj=gp.quicksum(X[j] for j in range(np.shape(binary_covering_array)[1]))
    m.setObjective(obj, GRB.MINIMIZE)
    m.Params.MIPGap = 0.001    # 0.1%
    m.Params.TimeLimit = 60*3  # 3 minutes
    m.optimize()
    #m.write("out.lp")
    #print (m.display())
    run_time = time.clock() - start_time
    objective = m.getObjective()
    objective_value=objective.getValue()
    active_stops_index=[a for a in range( np.shape(binary_covering_array)[1]) if X[a].x > 0.999]
    #num_stops_opened=len(active_stops_index)
    return objective_value,active_stops_index,run_time#,num_stops_opened,

#completely resolve bus stops every year
def assign_stu_to_stops(active_stu_df,stops):#
    #(opened_stop_list,num_open_stops)=open_stops(active_stu_df,stops)
    for sid in active_stu_df['ID'].values:

        opened_stop_index=active_stu_df.loc[active_stu_df['ID']==sid,'index_stops']
        open_stop_df=stops.iloc[opened_stop_index.tolist()[0]]
        sx=active_stu_df.loc[active_stu_df['ID']==sid,'X'].values[0]
        sy=active_stu_df.loc[active_stu_df['ID']==sid,'Y'].values[0]
        cdist_ = cdist([(sx,sy)],[(x,y) for x, y in zip(open_stop_df.X, open_stop_df.Y)], 'minkowski', p=1)
        arg_min = cdist_.argmin()
        min_ = cdist_.min()
        stop_selected=open_stop_df.iloc[arg_min]['StopID']
        active_stu_df.loc[active_stu_df['ID']==sid,'StopID']=stop_selected
        active_stu_df.loc[active_stu_df['ID']==sid,'walking_distance']=min_
    return active_stu_df

#%% #incremental
def assign_stu_to_stops2(active_stu_df,opened_stop_index,students,allstops2):#
    #(opened_stop_list,num_open_stops)=open_stops(active_stu_df,stops)
    open_stop_df=allstops2.iloc[opened_stop_index]
    for sid in active_stu_df['ID'].values:
        #print('sid_inside',sid)
        sx=active_stu_df.loc[active_stu_df['ID']==sid,'X'].values[0]
        sy=active_stu_df.loc[active_stu_df['ID']==sid,'Y'].values[0]
        cdist_ = cdist([(sx,sy)],[(x,y) for x, y in zip(open_stop_df.X, open_stop_df.Y)], 'minkowski', p=1)
        arg_min = cdist_.argmin()
        #min_ = cdist_.min()
        stop_selected=open_stop_df.iloc[arg_min]['StopID']
        #print('stop_selected',stop_selected)
        active_stu_df.loc[active_stu_df['ID']==sid,'stop_assigned']=stop_selected
        students.loc[students['ID']==sid,'stop_assigned']=stop_selected
    return active_stu_df, students