# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 19:36:03 2020

@author: SSingh169
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



O= pd.read_csv('washout.csv')
#O=O[['GR','DTCO']]
O_t = O[O.DTCO<0]
O_T= O[O.DTCO>0]


def deg(B,i,k):
    A=B.copy()
    if i==0:
        A[k]=0
    else:
        
        a_1 = A[[k]].values
        poly_r = PolynomialFeatures(degree=i)
        a_poly_1 = poly_r.fit_transform(a_1)
        for j in range(i+1):
            A[k +str(j)] = a_poly_1[:,j]
        A=A.drop([k + str(0), k + str(1)],axis=1)
    return A   


def pop(C,B,n):
    A=C.copy()
    A['L'] = pd.qcut(B['GR'],n, labels=False)
    A['V']=  pd.qcut(B['RHO'],n, labels=False)
    J= A.groupby(['L','V']).size().reset_index()
    T= [0]
    k=0
    for i in range(len(J[0])):
        k+= J[0][i]
        T.append(k)
    C=A.copy()
    C= C.sort_values(['L','V'])    
    Z={}
    Z={i : C.iloc[T[i]:T[i+1],:] for i in range(len(J)) }
    return Z


import itertools

threshold=9.75
O_t = O[O.DTCO<0]
W_0 = O_t[O_t.HCAL< threshold]
W_1= O_t[O_t.HCAL> threshold]

#Divinding the respective data into groups
X_0 = pop(W_0,W_0,4)
X_1 = pop(W_1,W_1,4)

T_0={}
for i in range(len(X_0)):
     D=pd.DataFrame()
     D= O_T[O_T.RHO< ((np.max(X_0[i].RHO))+(np.std(X_0[i].RHO))*0.1)]
     D= D[D.RHO> ((np.min(X_0[i].RHO))-(np.std(X_0[i].RHO)*0.1))]
     #Density is the guiding factor
     D= D[D.GR > ((np.min(X_0[i].GR))-(np.std(X_0[i].GR)*0.6))]
     D= D[D.GR < ((np.max(X_0[i].GR))+(np.std(X_0[i].GR)*0.6))]
     T_0[i]= D

T_1={}
for i in range(len(X_1)):
     D=pd.DataFrame()
     #GR is the guiding factor
     D= O_T[O_T.GR< ((np.max(X_1[i].GR))+(np.std(X_1[i].GR))*0.3)]
     D= D[D.GR > ((np.min(X_1[i].GR))-(np.std(X_1[i].GR)*0.3))]
     
     T_1[i]= D   

def Washout_1(d):
    
    #t_E=[]
    #P=[]
    T,t = train_test_split(d,test_size=1/4,random_state=0)
        #plt.scatter(t.DTCO,t.GR)
    base= 500
    p=0
    for j in [1,2,3,4,5]:
        
        Z=T.copy()
        A=t.copy()
        Z=deg(Z,j,'GR')
        A=deg(A,j,'GR')
        l= LinearRegression()
        l.fit(Z.drop(['DTCO','HCAL','RHO'],axis=1), Z['DTCO'])#RHO dropped from evaluation of prediction data
        Z['Poly']= l.predict(Z.drop(['DTCO','HCAL','RHO'],axis=1))
        A['Poly']= l.predict(A.drop(['DTCO','HCAL','RHO'],axis=1))
        T_e = mean_squared_error(Z.DTCO,Z.Poly)
        t_e = mean_squared_error(A.DTCO,A.Poly)
        if t_e < base:
            base= t_e
            p= j
    return p
        
    
         
#Degree detection in washout zones 1= TRUE
G_1=[]
for i in range(len(T_1)):
    print(len(T_1[i]))
    k= Washout_1(T_1[i])
    #for i in range(len(k)):
    #    if k[i]== 0:
    #        k[i]= int(np.round(np.mean(k),0))#to avoid GR_1 error the error for that region is more than base error and thus the apt degree is not chosen
    G_1.append(k)

#prediction for washout zones
for i in range(len(X_1)):
    T= T_1[i].copy()
    T=deg(T,G_1[i],'GR')
    X_1[i]=deg(X_1[i],G_1[i],'GR')
    l=LinearRegression()
    l.fit(T.drop(['DTCO','HCAL','RHO'],axis=1), T['DTCO'])
    X_1[i]['Poly']= l.predict(X_1[i].drop(['DTCO','HCAL','RHO','L','V'],axis=1))
    from scipy.ndimage import gaussian_filter
    X_1[i]['DTCO']= gaussian_filter(X_1[i].Poly, sigma=0.05)

    









def Washout_0(d):
    
    #t_E=[]
    #P=[]
    T,t = train_test_split(d,test_size=1/4,random_state=0)
        #plt.scatter(t.DTCO,t.GR)
    base= 500
    d=0
    g=0
    a=[1,2,3,4,5]
    b=[1,2,3,4,5]
    C=list(itertools.product((x,y) for x in a for y in b))
    for j in range(len(C)):
        
        Z=T.copy()
        A=t.copy()
        Z=deg(Z,C[j][0][0],'GR')
        A=deg(A,C[j][0][0],'GR')
        Z=deg(Z,C[j][0][1],'RHO')
        A=deg(A,C[j][0][1],'RHO')
        
        l= LinearRegression()
        l.fit(Z.drop(['DTCO','HCAL'],axis=1), Z['DTCO'])
        Z['Poly']= l.predict(Z.drop(['DTCO','HCAL'],axis=1))
        A['Poly']= l.predict(A.drop(['DTCO','HCAL'],axis=1))
        T_e = mean_squared_error(Z.DTCO,Z.Poly)
        t_e = mean_squared_error(A.DTCO,A.Poly)
        if t_e < base:
            base= t_e
            d= C[j][0][1]
            g= C[j][0][0]
    return d,g

D_0=[]
G_0=[]
for i in range(len(T_0)):
    print(len(T_0[i]))
    d,g= Washout_0(T_0[i])
    #for i in range(len(k)):
    #    if k[i]== 0:
    #        k[i]= int(np.round(np.mean(k),0))#to avoid GR_1 error the error for that region is more than base error and thus the apt degree is not chosen
    D_0.append(d)
    G_0.append(g)

#Prediction 
for i in range(len(X_0)):
    T= T_0[i].copy()
    T=deg(T,G_0[i],'GR')
    T=deg(T,D_0[i],'RHO')
    X_0[i]=deg(X_0[i],G_0[i],'GR')
    X_0[i]=deg(X_0[i],D_0[i],'RHO')
    l=LinearRegression()
    l.fit(T.drop(['DTCO','HCAL'],axis=1), T['DTCO'])
    X_0[i]['Poly']= l.predict(X_0[i].drop(['DTCO','HCAL','L','V'],axis=1))
    from scipy.ndimage import gaussian_filter
    X_0[i]['DTCO']= gaussian_filter(X_0[i].Poly, sigma=0.05)
    X_0[i]=X_0[i][['DTCO','GR','RHO']]

lis= ['DTCO','GR','RHO']

W_0_F = pd.concat(X_0.values())
W_1_F= pd.concat(X_1.values())[lis]


O_t_F=pd.concat([W_0_F,W_1_F])

D_F= pd.concat([O_t_F,O_T])[lis]
D_F = D_F.sort_index()







#####END









def P_I(d,threshhold):
    t_E=[]
    P=[]
    W=[]
    Com = {}     
    for k in [1,2,3]:
        x = pop(d,d,k)# we have only divided the dataset into parts
        Error = 0
        for i in range(k):
            T,t = train_test_split(x[i],test_size=1/5,random_state=0)
            base= 500
            p=0
            w=0
            j = [1,2,3,4,5,8,10]
            l = [1,2,3,4,5,8,10]

           
            C=list(itertools.product((x,y) for x in j for y in l))
            
            for h in range(len(C)):
                Z=T.copy()
                A=t.copy()
                Z=deg(Z,C[h][0][0],'GR')
                A=deg(A,C[h][0][0],'GR')
                if (sum(Z.HCAL > threshhold)) > 0.2*(len(Z.HCAL)):
                    Z=deg(Z,0,'RHO')
                    A=deg(A,0,'RHO')
                else:
                    Z=deg(Z,C[h][0][1],'RHO')
                    A=deg(A,C[h][0][1],'RHO')
                l= LinearRegression()
                l.fit(Z.drop(['DTCO','L','V','HCAL'],axis=1), Z['DTCO'])
                Z['Poly']= l.predict(Z.drop(['DTCO','L','V','HCAL'],axis=1))
                A['Poly']= l.predict(A.drop(['DTCO','L','V','HCAL'],axis=1))
                T_e = mean_squared_error(Z.DTCO,Z.Poly)
                t_e = mean_squared_error(A.DTCO,A.Poly)
                if t_e < base:
                    base= t_e
                    p= C[h][0][0]
                    w= C[h][0][1]
             
            t_E.append(base)
            P.append(p)
            W.append(w)
            Error += base*base*len(x[i])*0.2
        Com[k] = (np.sqrt((Error/len(d))))
    C_S=sorted(Com.items(), key = lambda kv:(kv[1], kv[0]))
    return C_S[0][0], Com


