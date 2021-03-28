# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 15:38:41 2020

@author: SSingh169
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 17:20:35 2020

@author: SSingh169
"""

C_S=sorted(Com.items(), key = lambda kv:(kv[1], kv[0]))

d_2 = pd.read_csv('KGAE.csv')
d_2=d_2[['GR','DTCO']]#'DT','VPVS']]
d= d_2[d_2.DTCO<0]
d_T= d_2[d_2.DTCO>0]
d_2= d_2[d_2.DTCO>0]
d_2=d_2[d_2.GR<4.5]



def hel(A,i):
    a_1 = A.loc[:,'GR'].values.reshape(len(A),1)
    b_1 = A.loc[:,'DT'].values.reshape(len(A),1)
    poly_r = PolynomialFeatures(degree=i)
    a_poly_1 = poly_r.fit_transform(a_1)
    b_poly_1 = poly_r.fit_transform(b_1)
    for j in range(i+1):
        A['GR_'+ str(j)] = a_poly_1[:,j]
        A['DT_' + str(j)] = b_poly_1[:,j]
    A=A.drop(['GR_0','GR_1','DT_0','DT_1'],axis=1)
    return A 
def pop(A,B,n):
    A['L'] = pd.qcut(B['GR'],n, labels=False)
    A['V']=  pd.qcut(B['DT'],n, labels=False)
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
from sklearn.model_selection import train_test_split

def P_I(d):
    t_E=[]
    P=[]
    Com = {}     
    for k in [1]:#2,3,4,10,12,15,20]
        x = pop(d,d,k)
        Error = 0
        for i in range(k):
            T,t = train_test_split(x[i],test_size=1/5,random_state=0)
            base= 2500
            p=0
            for j in [1,2,3]:
                Z=T.copy()
                A=t.copy()
                Z=hel(Z,j)
                A=hel(A,j)
                l= LinearRegression()
                l.fit(Z.drop(['DTCO','L','V'],axis=1), Z['DTCO'])
                Z['Poly']= l.predict(Z.drop(['DTCO','L','V'],axis=1))
                A['Poly']= l.predict(A.drop(['DTCO','L','V'],axis=1))
                T_e = mean_squared_error(Z.DTCO,Z.Poly)
                t_e = mean_squared_error(A.DTCO,A.Poly)
                if t_e < base:
                    base= t_e
                    p= j
             
            t_E.append(base)
            P.append(p) 
            Error += base*base*len(x[i])
        Com[k] = (np.sqrt((Error/len(d_2))))
    C_S=sorted(Com.items(), key = lambda kv:(kv[1], kv[0]))
    return C_S[0][0], Com




def L_o_P(d,o):
    
    t_E=[]
    P=[]
    x = pop(d,d,o)    
    Error=0
    for i in range(o):
        T,t = train_test_split(x[i],test_size=1/4,random_state=0)
        #plt.scatter(t.DTCO,t.GR)
        base= 2500
        p=0
        for j in [1,2,3,5]:
            Z=T.copy()
            A=t.copy()
            Z=hel(Z,j)
            A=hel(A,j)
            l= LinearRegression()
            l.fit(Z.drop(['DTCO','L','V'],axis=1), Z['DTCO'])
            Z['Poly']= l.predict(Z.drop(['DTCO','L','V'],axis=1))
            A['Poly']= l.predict(A.drop(['DTCO','L','V'],axis=1))
            T_e = mean_squared_error(Z.DTCO,Z.Poly)
            t_e = mean_squared_error(A.DTCO,A.Poly)
            if t_e < base:
                base= t_e
                p= j
        t_E.append(base)
        P.append(p)
    return P     
         
def Itp(d,o,P):
    z={}
    y = pop(d,d,o)
    

    for i in range(o):
        T= y[i][y[i].DTCO>0]
        t= y[i][y[i].DTCO<0]
    
        if len(t)<= 0:
            z[i]=y[i]
        
    
        else:
            print(len(t))
            T= hel(T,P[i])
            t=hel(t,P[i])
            l= LinearRegression()
            l.fit(T.drop(['DTCO','L','V'],axis=1), T['DTCO'])
            t['DTCO']= l.predict(t.drop(['DTCO','L','V'],axis=1))
            z[i]= pd.concat([T[['GR','DTCO','L','V']],t[['GR','DTCO','L','V']]])
            z[i] = z[i].sort_index()
    O_r = pd.concat(z.values())  
    O_r = O_r.sort_index()  
    d=d.sort_index() 
    d['Poly']= d['DTCO']   
    d['Poly']= O_r['DTCO']
    return d
    

O_T= pop(d_T,d_T,5)
O= pop(d_2,d_2,5) 
A_a=[]     
K={}
for i in range(25):
    print(len(O_T[i]))
    a,A = P_I(O_T[i])
    k= L_o_P(O_T[i],a)
    #for i in range(len(k)):
    #    if k[i]== 0:
    #        k[i]= int(np.round(np.mean(k),0))#to avoid GR_1 error the error for that region is more than base error and thus the apt degree is not chosen
    A_a.append(a)
    K[i]=k
    
            
#    F[i]= Itp(O[i],a,k)
for i in range(25):
    for j in range(len(K[i])):
        if K[i][j] == 0:
            K[i][j]=1
            
                  
    
    
 
a,A= P_I(O_T[2])     
k= L_o_P(O_T[2],a) 
f=Itp(O[2],a,k)
f.describe()
O[2].describe()



F={}
for i in range(25):
    F[i]= Itp(O[i],A_a[i],K[i])
for i in range(12):
    print(F[i].describe())
    
Fi=pd.concat(F.values())  
Fi=Fi.sort_index()  
Fi.describe()

from scipy.ndimage import gaussian_filter
Fi['Po_Smth']= gaussian_filter(Fi.Poly, sigma=0.45)
Fi['Po_Smth']= np.round(Fi['Poly'],2)




F_d=Fi.iloc[d.index,:]



from sklearn.model_selection import train_test_split
D_train,D_test = train_test_split(d_T,test_size=1/6,random_state=0)

Linreg_1.fit(D_train.drop(['DTCO'],axis=1), D_train['DTCO'])
e= mean_squared_error(D_train.DTCO, Linreg_1.predict(D_train.drop(['DTCO'],axis=1)))
e= mean_squared_error(D_test.DTCO, Linreg_1.predict(D_test.drop(['DTCO'],axis=1)))
