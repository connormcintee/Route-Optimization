import numpy as np
import random
import pandas as pd
from math import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from datetime import date

class District(object):
    '''
    Create a district, generating schools and students for multiple years
    '''
    def __init__(self,
                 district_length,
                 min_distance_between_school,
                 num_of_boundary_school,
                 num_of_charter_school,
                 average_students_per_school,
                 percentage_students_boundary,
                 total_num_year,
                 num_of_grades,
                 CV,
                 seed1,
                 seed2,
                 defined_schools = None): #dataframe of pre-specified school locations

        self.average_students_per_school = average_students_per_school
        self.total_new_students_per_year=int(round(self.average_students_per_school*(num_of_boundary_school+num_of_charter_school)/num_of_grades))
        self.district_length=district_length
        self.min_distance_between_school=min_distance_between_school
        self.total_num_year = total_num_year
        self.num_of_charter_school=num_of_charter_school
        self.num_of_boundary_school=num_of_boundary_school
        self.percentage_students_boundary=percentage_students_boundary
        self.CV=CV
        self.schools = None
        self.students = None
        self.seed1=seed1
        self.seed2 = seed2
        #self.students=self.all_year_students()
        #self.sch_fig = self.plot_schools()
        #self.stu_fig = self.plot_students()

    #input predefined school locations
    def pre_define_schools(self, defined_schools):
        if defined_schools!= None:
            self.schools = defined_schools

    #generate random school locations
    def set_random_schools(self):
        self.schools= self.all_schools(self.seed1, self.seed2)

    def set_defined_schools(self,school_lotation_list):
        if not school_lotation_list == []:
            self.schools = school_lotation_list

    def generate_all_stops(self):
        all_potential_stop=[] #totally
        for i in range(int(self.district_length/100)+1):
            for j in range(int(self.district_length/100)+1):
                all_potential_stop.append((i*100,j*100))
        all_potential_stop = pd.DataFrame(all_potential_stop, columns =['X', 'Y'])
        all_potential_stop['StopID']=range(1,len(all_potential_stop)+1)
        return all_potential_stop

    def set_potential_stops(self):
        self.all_stops = self.generate_all_stops()

    def all_schools(self,seed1,seed2):
        bdx, bdy = self.school(self.num_of_boundary_school,seed1)
        boundary_schools = {'SchoolID': range(1, self.num_of_boundary_school + 1), 'X': bdx, 'Y': bdy, 'type': 'boundary'}
        boundary_schools = pd.DataFrame(boundary_schools, index=range(self.num_of_boundary_school))
        random.seed(1)
        ctx, cty = self.school(self.num_of_charter_school,seed2)
        charter_schools = {'SchoolID': range(self.num_of_boundary_school + 1, self.num_of_charter_school + self.num_of_boundary_school + 1), 'X': ctx, 'Y': cty, 'type': 'charter'}
        charter_schools = pd.DataFrame(charter_schools, index=range(self.num_of_boundary_school + 1, self.num_of_charter_school + self.num_of_boundary_school + 1))
        schools = pd.concat([boundary_schools, charter_schools])
        schools.to_csv('all_schools {}.csv'.format(date.today()), index=False)
        return schools

    def school(self, num_of_schools,seed):
        sch_x = []
        sch_y = []
        i = 0
        random.seed(seed)
        while i < num_of_schools:
            x = random.randint(0, self.district_length)
            y = random.randint(0, self.district_length)
            if i <= 0:
                sch_x.append(x)
                sch_y.append(y)
                i += 1
            if i > 0:
                llst = [sqrt((x - sch_x[j]) ** 2 + (y - sch_y[j]) ** 2) for j in range(len(sch_x))]
                if min(llst) > self.min_distance_between_school:
                    sch_x.append(x)
                    sch_y.append(y)
                    i += 1
        return sch_x, sch_y

    def plot_schools(self):
        schools=self.schools
        bdx = schools.loc[schools['type'] == 'boundary', 'X']
        bdy = schools.loc[schools['type'] == 'boundary', 'Y']
        ctx = schools.loc[schools['type'] == 'charter', 'X']
        cty = schools.loc[schools['type'] == 'charter', 'Y']
        plt.figure(figsize=(10, 10))
        plt.plot(ctx, cty, "v",color='red',markersize=12)
        plt.plot(bdx, bdy, "s",color='blue',markersize=12)
        plt.title('school locations')
        plt.legend(['charter school', 'boundary school'], prop={'size': 15})
        plt.show
        plt.savefig('School locations:{} boundary schools, {} charter schools .png'.format(len(bdx),len(ctx)))

    def all_year_students(self):#number of new students generted each year follow a normal distribution
        t = 1
        column_names = ["ID", "X", "Y", 'type', 'enroll_year', 'grad_year']
        students = pd.DataFrame(columns=column_names)
        while t <=  self.total_num_year +7:
            num_new_student = self.normal_stu(mu=self.total_new_students_per_year,sigma=round(self.CV*self.total_new_students_per_year))
            stu = self.generate_stu(new_student=num_new_student, alpha=self.percentage_students_boundary, timer=t)
            students = students.append(stu)
            t += 1
        students['ID'] = range(1, len(students) + 1)
        students['sub_ID'] = students['sub_ID'].round(0).astype(int)
        return students

    def set_students(self):
        self.students = self.all_year_students()

    def plot_students(self):
        '''
        :param stu: dataframe with columns:ID,X,Y,type
        :return: plot figure
        '''
        stu=self.students
        plt.figure(figsize=(20, 20))
        plt.plot(stu['X'], stu['Y'], ".", color='black', markersize=0.1)
        plt.show
        plt.savefig('student locations')

    def generate_stu(self, new_student, alpha, timer):  # M:size of the district,s:number of students to generate
        new_student= int(new_student)
        students = np.random.random((new_student, 2)) * self.district_length
        stuX = students[:, 0]
        stuY = students[:, 1]
        students = {'sub_ID': range(1, new_student + 1), 'X': stuX, 'Y': stuY, 'type': 'charter', 'enroll_year': timer,
                    'grad_year': 'NA'}
        students = pd.DataFrame(students, index=range(new_student))
        students['X'] = students['X'].round(0).astype(int)
        students['Y'] = students['Y'].round(0).astype(int)
        students['grad_year'] = students['enroll_year'].apply(self.geometric_stay)
        students.loc[random.sample(range(new_student),
                                   int(new_student * alpha)), 'type'] = 'boundary'  # id of students going to boundary schools, alpha percent.
        return students  # first column ID, students['X'],student['Y'], student['type'] (boundary or charter)
    def generate_random_stu(self,new_student,alpha,timer, current_grade):
        new_student = int(new_student)
        students = np.random.random((new_student, 2)) * self.district_length #location
        stuX = students[:, 0]
        stuY = students[:, 1]
        students = {'sub_ID': range(1, new_student + 1), 'X': stuX, 'Y': stuY, 'type': 'charter', 'enroll_year': timer,
                    'grad_year': 'NA'}
        students = pd.DataFrame(students, index=range(new_student))
        students['X'] = students['X'].round(0).astype(int)
        students['Y'] = students['Y'].round(0).astype(int)
        students['grad_year'] = max(0,students['enroll_year'].apply(self.geometric_stay)- current_grade) #grad_year: remaining time left in the school
        students.loc[random.sample(range(new_student),
                                   int(new_student * alpha)), 'type'] = 'boundary'  # id of students going to boundary schools, alpha percent.
        return students  # first column ID, students['X'],student['Y'], student['type'] (boundary or charter)

    def geometric_stay(self,enroll):  # probaiblity of stay in a school (geometric)
        rd = random.uniform(0, 1)
        if 0 < rd <= 0.01:
            out = enroll
        elif 0.01 < rd <= 0.02:
            out = enroll + 1
        elif 0.02 < rd <= 0.03:
            out = enroll + 2
        elif 0.03 < rd <= 0.04:
            out = enroll + 3
        elif 0.04 < rd <= 0.05:
            out = enroll + 4
        elif 0.05 < rd <= 1:
            out = enroll + 5
        return out  # last year of sta y

    def normal_stu(self,mu, sigma=5):  # new students generated follow normal distrbution
        return int(np.random.normal(mu, sigma))

    def output_dist_matrix(self):
        students = self.students
        stops = self.all_stops
        schools=self.schools
        students.to_csv('students {} .csv'.format(date.today()))
        stops.to_csv('stops {} .csv'.format(date.today()))
        schools.to_csv('schools {} .csv'.format(date.today()))

def generate_district(district_length, min_distance_between_school, num_of_boundary_school, num_of_charter_school,
                      average_students_per_school, percentage_students_boundary, total_num_year,num_of_grades,CV, seed1, seed2):
    return District(district_length, min_distance_between_school, num_of_boundary_school, num_of_charter_school, average_students_per_school, percentage_students_boundary, total_num_year, num_of_grades, CV, seed1, seed2)

