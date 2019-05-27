# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 14:47:48 2019

@author: abhis
"""

import numpy as np
import pandas as pd
import csv
import re

import sys
sys.path.insert(0, 'C:/Users/abhis/AppData/Local/Programs/Python/Python37-32/Lib/site-packages')



def data_clean(df):
    new=[]
    for i, row in df.iterrows():
        newrow=[]
        for j, column in row.iteritems():
            column=str(column)        
            if(len(str(column))!=0):
                x=re.sub("%.*$", "", column)
                if(len(x)<5):
                    newrow.append(float(x)/100)
        new.append(newrow)
    df=pd.DataFrame(new)
    return df

#table read and cleaning
mar=pd.read_csv('Table2.csv')
mardf=data_clean(mar)

df3=pd.read_csv('Table3.csv')
df3=data_clean(df3)

df4=pd.read_csv('Table4.csv')
df4=data_clean(df4)

df5=pd.read_csv('Table5.csv')
df5=data_clean(df5)

df6=pd.read_csv('Table6.csv')
df6=data_clean(df6)

df7=pd.read_csv('Table7.csv')
df7=data_clean(df7)

df8=pd.read_csv('Table8.csv')
df8.iloc[6][5]=30.9
df8=data_clean(df8)


def marginal(c1,c2):
    y=np.outer(c1,c2)
    mask = (np.nan_to_num(y) != 0).any(axis=1)
    x=y[mask]
    x=np.transpose(x) 
    mask = (np.nan_to_num(x) != 0).any(axis=1)
    x=x[mask]
    return x


def conditional(cdf,a,b,x1,y1,g,h):
    new=[]   
    for i in range(0,len(cdf.columns)):
        x=cdf[i][g:h]
        mul=cdf[i][0]
        x=x.multiply(mul)
        new.append(x)
    y=np.transpose(new)
    y=np.array(y).reshape(a,b)
    return y


def createTable(new,df):
    finalDF=pd.DataFrame()
    
    for i in range(0,len(new)):
        x1,y1,g,h=new[i][0],new[i][1],new[i][2],new[i][3]
        x1-=1
        y1-=1
        x=marginal(mardf[x1],mardf[y1])
        a,b=x.shape
        y=(conditional(df,a,b,x1,y1,g,h))
        f,g=y.shape
        sub=np.subtract(x,y)
        a,b=sub.shape
        nu=a*b
        nums=np.transpose(sub)
        x=np.abs(nums).reshape(nu,1)
        x = pd.DataFrame.from_records(x)
        finalDF=pd.concat([finalDF,x],axis=1)
    return finalDF

finalDF = pd.DataFrame()
new2=[3,4,5,6,7,8]

l=[]

for i in new2:
    if(i==3):
        new6=[[1,2,1,6],[1,4,6,10],[1,6,10,15]]
        x3=createTable(new6,df3)
        finalDF=pd.concat([finalDF,x3],axis=1)
        #print(x3)
        
    if(i==4):
        new6=[[2,3,1,4],[2,5,4,8]]
        x4=createTable(new6,df4)
        finalDF=pd.concat([finalDF,x4],axis=1)
        #print(x4)
        
    if(i==5):
        new6=[[3,2,1,6],[3,5,6,10],[3,6,10,15]]
        x5=createTable(new6,df5)
        finalDF=pd.concat([finalDF,x5],axis=1)
        #print(x5)
        
    if(i==6):
        new6=[[4,1,1,5],[4,2,5,10],[4,6,10,15]]
        x6=createTable(new6,df6)
        finalDF=pd.concat([finalDF,x6],axis=1)
        #print(x6)
        
    if(i==7):
        new6=[[5,2,1,6],[5,3,6,9]]
        x7=createTable(new6,df7)
        finalDF=pd.concat([finalDF,x7],axis=1)
        #print(x7)
        
    if(i==8):
        new6=[[6,1,1,5],[6,2,5,10],[6,3,10,13],[6,4,13,17]]
        x8=createTable(new6,df8)
        finalDF=pd.concat([finalDF,x8],axis=1)
        #print(x8)

x=list(finalDF.sum(axis=0))

d = {"x2/x1":x[0], "x4/x1":x[1], "x6/x1":x[2], "x3/x2":x[3],'x5/x2':x[4],'x2/x3':x[5],'x5/x3':x[6],'x6/x3':x[7]
          ,'x1/x4':x[8],'x2/x4':x[9],'x6/x4':x[10],'x2/x5':x[11],'x3/x5':x[12],'x1/x6':x[13],'x2/x6':x[14],'x3/x6':x[15],'x4/x6':x[16]}
new={}
x.sort(reverse=True)
result=[]

for i in d.keys():
    if(d[i]>0.140):
        result.append(d[i])
        new[i]=d[i]
    
#print(result)  
print(d)  
    




    






    


