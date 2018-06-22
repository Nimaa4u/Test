# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

2+2


#%%

import math 
numsides = 8 
innerangleb = 360 / numsides 
halfanglea = innerangleb/2
onehalfsides=math.sin(math.radians(halfanglea))
sides = onehalfsides * 2
poly=numsides*sides
pi=poly/2
#%%

nima = [3,"cat",5, "nima"]

sina = [1,2,"babi",3,4,"father"]

gholam = [1,2,3,4]
#%%
mylist = [7,9,'a','ca',False]
#%%
def getrange(nimalist):
    return max(nimalist) - min(nimalist)

#%%
def getmax(glist):
    maxsofar=glist[0]
    for item in glist:
        print("ITEM = ",item)
        if item > maxsofar:
           maxsofar = item
           print ("MAXSOFAR", maxsofar)   
    return maxsofar
#%%
    
def getmin(minlist):
    minsofar=minlist[0]
    for item in minlist[1:]:
        if item < minsofar:
            minsofar = item
            print("MINSOFAR", minsofar)
    return minsofar
       
#%%
def mean(llist):
    mean = sum (llist)/len(llist)
    return mean 
#%%
    #list
def median(alist):
    copylist = alist[:]
    print ("copylist", copylist)
    copylist.sort()
    print ("sort", copylist)
    print (len(copylist))
    if len(copylist)%2 == 0:
        rightmid =len(copylist)//2
        leftmid =rightmid-1
        median = (copylist[leftmid]+copylist[rightmid])/2
        print (copylist[leftmid],copylist[rightmid])
    else: 
        print("else")
        mid = len (copylist)//2
        median = copylist[mid]
    return median 
#%%
#dictionary 

ages = {'david':45, 'brenda':46, 'nima':34, 'ghazal':20, 'Behnaz':29}    
for k in ages.keys():
    print(k)      
#%%
#

def mode(alist):
    countdict = {}
    print (countdict)
    for item in alist:
        if item in countdict:
            countdict[item] = countdict[item] + 1 
        else: 
            countdict[item] = 1
    countlist = countdict.values()
    print("countlist values",countlist)
    maxcount = max(countlist)
    
    modelist = []
    for item in countdict:
        if countdict[item] == maxcount:
            modelist.append(item)
    return modelist 
#%%
import random
import math

def carlo(time):
    
    incircle = 0 
    
    for i in range(time):
        x=random.random()
        y=random.random()
        #print (i)
        d = math.sqrt(x**2 + y**2)
        
        if d <= 1: 
            incircle = incircle + 1 
            
    pi = incircle / time * 4
        
    return pi
#%%
# page 55
acc =0 
for x in range(1,6):
    print (x)
    acc = acc + x 
print (acc)    
#%%

def leibniz(terms):
    acc = 0
    num = 4 
    den = 1 
        
    for aterm in range(terms):
        
        den = den +2 
        
    return acc
#%%
a = 5 
b = 3 
if a>b: 
    c=10 
else:
    c=20
#%%
# page 78

import random 
import math 
import turtle

def monte (numsim):
    wn = turtle.Screen()
    drawingt = turtle.Turtle()
    wn.setworldcoordinates(-2,-2,2,2)
    drawingt.up()
    drawingt.goto(-1,0)
    drawingt.down()
    drawingt.goto(1,0)
    
    drawingt.up()
    drawingt.goto(0,1)
    drawingt.down()
    drawingt.goto(0,-1)
    
    circle = 0 
    drawingt.up()
    
    for i in range(numsim):
        x = random.random()
        y = random.random()
        
        d = math.sqrt(x**2 + y**2)
        drawingt.goto(x,y)
        
        if d<= 1:
            circle = circle + 1 
            drawingt.color("blue")
        else:
            drawingt.color("red")
        drawingt.dot()
        
    pi = circle / numsim * 4 
    wn.exitonclick()

    return pi
#%% 
#page 87 
name = "Where Is My Love"
for i in range(len(name)):
    print(name[i])
#%%
name = "Ghazal Gharaee"
for i in range(len(name)):
    print(name[0:i+1])    
#%%    
for i in range(1000):
    print(i % 7, end= '')
#%%
def mode(alist):
    countdic = {}
    
    for item in alist:
        print("START OF LOOP ==>      ",item)
        if item in countdic: 
            countdic[item] = countdic[item] +1 
            print ("countdic[item]         ",countdic[item])
            print('countkey =========>>>>>>>>>>>', countdic.keys())
            print('countdic',countdic.values())
            print('countdic', countdic.items())
        else:
            countdic[item] = 1
    
    print("countdic ======>>>    ", countdic.items())    
    countlist = countdic.values()
    print("countdic value              ", countdic.values())
    print("countlist                 ",countlist)
    maxcount = max(countlist)
    
    modelist = [ ]
    if item in countdic:
        if countdic[item] == maxcount:
            modelist.append(item)
            
    return modelist
#%%    
import numpy as np
x = np.array([0,0,0])
c = np.array([175,90,160])
a = np.array([[2,1,4],[3,2,1],[1,3,3]])
b = np.array([630,550,600])

x3_only = int(np.min(b / a[:,2]))
lhs = np.array([np.sum(a[0]),np.sum(a[1]),np.sum(a[2])])
x.fill(int(np.min(b / lhs)))
obj_fun_val = x.dot(c)
print('All quantities the same:')
print('x:',x, 'Profit:',obj_fun_val)

#%%
import scipy.stats as sst
import numpy as np
import matplotlib.pyplot as plt

µ = 100
σ = 10
α = 0.05
x = sst.norm.ppf(1-α)

xv = np.linspace(µ-3*σ, µ+3*σ, 100, endpoint=True)     # 100 X-Axis points within ± 3σ of µ
print (xv)

yv = sst.norm.pdf(xv,µ,σ)   
print('yv=', yv)                           # 100 Y-Axis values

plt.plot(xv,yv,color='blue')			          # Show an X-Y chart

xvx = np.linspace(µ-3*σ, x, 100, endpoint=True)        # 100 X-Axis points from µ-3σ to x
yvx = sst.norm.pdf(xvx,µ,σ)                            # 100 Y-Axis values

plt.fill_between(xvx,0, yvx,color='red', alpha=0.25)   # Show a Filled X-Y Chart

txtx = (µ-3*σ + x)/2           
print(txtx)       
                 # X location for annotation text
txty = sst.norm.pdf(txtx,µ,σ)/2                        # Y location for annotation text
txt = 'P(X<=' + str(x) + ')'                           # annotation text                          
plt.annotate(txt,xy=(txtx,txty), fontsize=10)          # show annotation text 

plt.plot([µ-3.25*σ, µ+3.25*σ],[0,0],color='black', linewidth=0.75)  # Draw X-Axis line
plt.show()         
#%%
f =  open('D:\grades.txt')
n = 0
sum = 0.0
for line in f:
    line_items = line.split(',')
    student_name = line_items[0] + ' ' + line_items[1]
    major = line_items[2]
    grade = float(line_items[3])
    print(student_name, major, grade)     
    
#%%
f =  open('D:\grades.txt')
grade_by_major = {}
print('START FROM HERE')
print('grade_by_major=     ', grade_by_major)

n = 0
sum = 0.0
for line in f:
    print('============================================================')
    print ('LINE NUMBER = ', n )
    print('============================================================')
    
    line_items = line.split(',')
    print (line_items)
    major = line_items[2]
    grade = float(line_items[3])
    grade_by_major.setdefault(major,[0,0])
    print('---------------------------')
    print(grade_by_major)
   
    print('print= ', grade)
    grade_by_major[major][0] = grade_by_major[major][0] + grade
    grade_by_major[major][1] = grade_by_major[major][1] + 1
    
    print ('grade_by_major[major][0] = ', grade_by_major[major][0])
    print ('grade_by_major[major][1] = ', grade_by_major[major][1])
    print(grade_by_major)
    print('************************************************************')
    n=n+1
print (grade_by_major.items())    
for k, v in grade_by_major.items():
    print('%s\t%0.2f' % (k,v[0]/v[1]))
#%%
ages = {}

for i in range(10):
    i_value = i
    ages.setdefault(i_value,[0,0])
    for j in range(20):
        nima = 'nima'
        ages[i_value][0]=i
        ages[i_value][1]=j
print (ages)
#%%


f =  open('D:\On_Time_On_Time_Performance_1998_1.csv')
grade_by_major = {}
print('START FROM HERE')
print('grade_by_major=     ', grade_by_major)

n = 0
sum = 0.0
for line in f:
    if n == 0:
       n=n+1 
       exit
    else:
        
        print('============================================================')
        print ('LINE NUMBER = ', n )
        print('============================================================')
        
        line_items = line.split(',')
        print ('print line', line_items)
        print('major')
        major = line_items[0]
        print(major)
        print(line_items[33])
        grade = float(line_items[33])
        grade_by_major.setdefault(major,[0,0,0])
        print('---------------------------')
        print(grade_by_major)
       
        print('print= ', grade)
        grade_by_major[major][0] = grade_by_major[major][0] + grade
        grade_by_major[major][1] = grade_by_major[major][1] + 1
        
        print ('grade_by_major[major][0] = ', grade_by_major[major][0])
        print ('grade_by_major[major][1] = ', grade_by_major[major][1])
        print(grade_by_major)
        print('************************************************************')
        n=n+1
    
print (grade_by_major.items())    
for k, v in grade_by_major.items():
    print('%s\t%0.2f' % (k,v[0]/v[1]))
#%%
import time
    
f =  open('D:\On_Time_On_Time_Performance_1998_1.csv')
grade_by_major = {}
print('START FROM HERE')
print('grade_by_major=     ', grade_by_major)

n = 0
j = 0
sum = 0.0

for line in f:
    
    if n == 0:
        line_items = line.split(',')
        print ('N= ', n)
        n=n+1
        for j in range(50):                      
            print('J= ',j , 'line_itemsn= ', line_items[j])
    elif n > 0:
        line_items = line.split(',')
        major = line_items[0]
        print('YEAR = ', major)
        print('line_items = ', line_items[33])
        if line_items[33] == '': 
            grade = 0
        else:
            grade = float(line_items[33])
        
        grade_by_major.setdefault(major,[0,0,0])
        print('---------------------------')
        print(grade_by_major)
        
        print('print= ', grade)
        grade_by_major[major][0] = grade_by_major[major][0] + grade
        grade_by_major[major][1] = grade_by_major[major][1] + 1
        print ('grade_by_major[major][0] = ', grade_by_major[major][0])
        print ('grade_by_major[major][1] = ', grade_by_major[major][1])
        print(grade_by_major)
        print('************************************************************')
        n=n+1
    
print (grade_by_major.items())    
for k, v in grade_by_major.items():
    print('%s\t%0.2f' % (k,v[0]/v[1]))

#%%    
import pandas as pd
grades = pd.read_csv('D:\student_grades_header.txt')
print(grades)
grades.groupby('major').mean()

#%%
import pandas as pd
import scipy  
import scikits.bootstrap as bootstrap 

grades = pd.read_csv('D:\On_Time_On_Time_Performance_1998_1.csv')
#print(grades)
grades.groupby(['Carrier','Year','Month'])['DepDelay'].mean()
#grades.groupby(['Carrier','Year','Month'])['DepDelay'].size().unstack()
#grades.groupby(['Year'])['DepDelay'].mean()
#grades.groupby(['Carrier','Year','Month'])['DepDelay'].std()
#grades.groupby('DayofMonth').mean()

#%%
    
f =  open('D:\On_Time_On_Time_Performance_1998_1.csv')
grade_by_major = {}

n = 0
j = 0
sum = 0.0

for line in f:
    
    if n == 0:
        line_items = line.split(',')
        n=n+1    
    elif n > 0:
        line_items = line.split(',')
        major = line_items[8]

        if line_items[33] == '': 
            grade = 0
        else:
            grade = float(line_items[33])
        
        grade_by_major.setdefault(major,[0,0,0])
        
        grade_by_major[major][0] = grade_by_major[major][0] + grade
        grade_by_major[major][1] = grade_by_major[major][1] + 1

        n=n+1


print (grade_by_major.items())    
for k, v in grade_by_major.items():
    print('%s\t%0.2f' % (k,v[0]/v[1]))

#%%
from scipy.stats import t
from numpy import average, std
from math import sqrt

if __name__ == '__main__':
    # data we want to evaluate: average height of 30 one year old male and
    # female toddlers. Interestingly, at this age height is not bimodal yet
    data = [63.5, 81.3, 88.9, 63.5, 76.2, 67.3, 66.0, 64.8, 74.9, 81.3, 76.2,
            72.4, 76.2, 81.3, 71.1, 80.0, 73.7, 74.9, 76.2, 86.4, 73.7, 81.3,
            68.6, 71.1, 83.8, 71.1, 68.6, 81.3, 73.7, 74.9]
    mean = average(data)
    # evaluate sample variance by setting delta degrees of freedom (ddof) to
    # 1. The degree used in calculations is N - ddof
    stddev = std(data, ddof=1)
    # Get the endpoints of the range that contains 95% of the distribution
    t_bounds = t.interval(0.95, len(data) - 1)
    # sum mean to the confidence interval
    ci = [mean + critval * stddev / sqrt(len(data)) for critval in t_bounds]
    print ("Mean: %f" % mean)
    print ("Confidence Interval 95%%: %f, %f" % (ci[0], ci[1]))   
    
    
#%%    