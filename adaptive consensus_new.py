# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 20:14:02 2023

@author: jason
"""

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import time


#%% Dynamics
#A=np.array([[-2.9,0.3,0.4,1.2],[-0.1,-0.2,0.6,1.5],[1.2,2.1,-2.8,3.4],[1,-2,-2.5,-2.5]])
#B=np.array([[0,0],[0,0],[-1,0.5],[-0.1,0.2]])


#A = np.array([[ 0,  1,  0,  0],[ 0,  0,  1,  0],[ 0,  0,  0,  1],[ 2, -1,  3,  1]])  # Unstable eigenvalues
#B = np.array([[0, 0],[1, 0],[0, 1],[1, 1]])


#A = np.array([[0.5, 1, 0, 0], [0, -0.2, 1, 0], [0, 0, 0.3, 2], [0, 0, 0, 0.1]])
#B = np.array([[1, 0], [0, 1], [1, 1], [0, 1]])


#A = np.array([[0.8, 0, 0, 1], [0, -0.5, 1, 0], [0, 0, 0.2, 0], [0, 1, 0, -0.1]])
#B = np.array([[1, 0], [0, 1], [1, 1], [0, 1]])

#A = np.array([[5, 0, 0, 2], [0, -3, 4, 0], [2, 0, 3, 5], [0, 4, 0, -1]])
#B = np.array([[2, 0], [0, 3], [4, 2], [1, 1]])

A = np.array([[0.8, 0.3, 0.2, 1.1], [0.4, -0.5, 1.2, 0.6], [0.7, 0.9, 0.2, 0.5], [1.3, 1.1, 0.4, -0.1]])
B = np.array([[1.2, 0.7], [0.6, 1.3], [1.1, 1.4], [0.9, 1.2]])

#A = np.array([[5.2, 2.3, 4.1, 6.0], [2.7, -3.5, 5.1, 4.3], [4.5, 3.8, 2.1, 5.0], [6.0, 5.1, 3.5, -2.0]])
#B = np.array([[1, 0], [0, 1], [1, 1], [1, 0]])

I=np.eye(4)
r=np.eye(2)

L=np.array([[2.168,-1.037,0,-0.865,-0.266],
            [-1.037,1.037,0,0,0],
            [0,0,1.651,-1.651,0],
            [-0.865,0,-1.651,2.863,-0.347],
            [-0.266,0,0,-0.347,0.613]])

#L=np.array([[ 2, -1, -1,  0,  0],
       #[-1,  2, -1,  0,  0],
       #[-1, -1,  3, -1,  0],
       #[ 0,  0, -1,  2, -1],
       #[ 0,  0,  0, -1,  1]])



# Calculate eigenvalues
eigenvalues = np.linalg.eigvals(L)

eig_sorted=np.sort(eigenvalues)
smallest_eigenvalue=eig_sorted[1]


h=0.001


alp=1/(2*smallest_eigenvalue)

z=1


#%% Solve Riccati equation
P = linalg.solve_continuous_are(A, B, I, r)

#P = np.array([[1.9008, 1.3065, 0,0],[1.3065, 2.4835,0,0],\
#              [0,0,1.9008, 1.3065],[0,0,1.3065, 2.4835]])

K=B.T@P

#K=B.T@P

A_cl = A - B @ K

# Check eigenvalues of the closed-loop system
eigvals_cl = np.linalg.eigvals(A_cl)
print("Closed-loop eigenvalues (A - BK):", eigvals_cl)

#%% Consensus
start_time=time.time()
N=3500

t=np.arange(0, h*N, h )

x1 = np.array([-6.125, 4.375, 9.875, -8.125]).T
The1 = np.array([1]).T
The1_r = np.array([3]).T

x2 = np.array([1.0, -8.5, -2.5, 10.0]).T
The2 = np.array([1]).T
The2_r = np.array([6]).T

x3 = np.array([-5.0, 7.5, -3.5, 1.0]).T
The3 = np.array([3]).T
The3_r = np.array([1.5]).T

x4 = np.array([10.125, -3.875, -2.875, -3.375]).T
The4 = np.array([2]).T
The4_r = np.array([5.5]).T

x5 = np.array([1.125, 0.125, -0.875, -0.375]).T
The5 = np.array([5]).T
The5_r = np.array([0.5]).T

PHI1=np.zeros([2*N])
PHI2=np.zeros([2*N])
PHI3=np.zeros([2*N])
PHI4=np.zeros([2*N])
PHI5=np.zeros([2*N])

error_list = np.zeros([N,1])
error_list_q = np.zeros([N,1])

consensus1=np.zeros([N,5])
consensus2=np.zeros([N,5])
consensus3=np.zeros([N,5])
consensus4=np.zeros([N,5])

THE1=np.zeros([N])

The1_s=np.zeros([N])
The2_s=np.zeros([N])
The3_s=np.zeros([N])
The4_s=np.zeros([N])
The5_s=np.zeros([N])

for i in range(N):
    
    consensus1[i,0]=x1[0]
    consensus1[i,1]=x2[0]
    consensus1[i,2]=x3[0]
    consensus1[i,3]=x4[0]
    consensus1[i,4]=x5[0]
    
    consensus2[i,0]=x1[1]
    consensus2[i,1]=x2[1]
    consensus2[i,2]=x3[1]
    consensus2[i,3]=x4[1]
    consensus2[i,4]=x5[1]
    
    consensus3[i,0]=x1[2]
    consensus3[i,1]=x2[2]
    consensus3[i,2]=x3[2]
    consensus3[i,3]=x4[2]
    consensus3[i,4]=x5[2]
    
    consensus4[i,0]=x1[3]
    consensus4[i,1]=x2[3]
    consensus4[i,2]=x3[3]
    consensus4[i,3]=x4[3]
    consensus4[i,4]=x5[3]
    

            
    The1_s[i]=The1
    The1_s[i]=The1
    The2_s[i]=The2
    The3_s[i]=The3
    The4_s[i]=The4
    The5_s[i]=The5
    
    Phi1=np.array([0.4157+0.3473*np.exp(-z*i)*x1[0],0.4157+0.3473*np.exp(-z*i)*x1[1]])
    PHI1[2*i:2*(i+1)]=Phi1
    Phi2=np.array([0.4017+0.5474*np.exp(-z*i)*x2[0],0.4017+0.5474*np.exp(-z*i)*x2[1]])
    PHI2[2*i:2*(i+1)]=Phi2
    Phi3=np.array([0.0302+0.5233*np.exp(-z*i)*x3[0],0.0302+0.5233*np.exp(-z*i)*x3[1]])
    PHI3[2*i:2*(i+1)]=Phi3
    Phi4=np.array([0.1996+0.2433*np.exp(-z*i)*x4[0],0.1996+0.2433*np.exp(-z*i)*x4[1]])
    PHI4[2*i:2*(i+1)]=Phi4
    Phi5=np.array([0.2634+0.3597*np.exp(-z*i)*x5[0],0.2634+0.3597*np.exp(-z*i)*x5[1]])
    PHI5[2*i:2*(i+1)]=Phi5
    
    The1=h*(Phi1.T@B.T@P@(L[0,1]*(x1-x2)+L[0,2]*(x1-x3)+L[0,3]*(x1-x4)+L[0,4]*(x1-x5))
            -PHI1.T@PHI1*(The1-The1_r))+The1
    The2=h*(Phi2.T@B.T@P@(L[1,0]*(x2-x1)+L[1,2]*(x2-x3)+L[1,3]*(x2-x4)+L[1,4]*(x2-x5))
            -PHI2.T@PHI2*(The2-The2_r))+The2
    The3=h*(Phi3.T@B.T@P@(L[2,0]*(x3-x1)+L[2,1]*(x3-x2)+L[2,3]*(x3-x4)+L[2,4]*(x3-x5))
            -PHI3.T@PHI3*(The3-The3_r))+The3
    The4=h*(Phi4.T@B.T@P@(L[3,0]*(x4-x1)+L[3,1]*(x4-x2)+L[3,2]*(x4-x3)+L[3,4]*(x4-x5))
            -PHI4.T@PHI4*(The4-The4_r))+The4
    The5=h*(Phi5.T@B.T@P@(L[4,0]*(x5-x1)+L[4,1]*(x5-x2)+L[4,2]*(x5-x3)+L[4,3]*(x5-x4))
            -PHI5.T@PHI5*(The5-The5_r))+The5
    
    x1=h*(A@x1+B@(alp*K@(L[0,1]*(x1-x2)+L[0,2]*(x1-x3)+L[0,3]*(x1-x4)+L[0,4]*(x1-x5))
                  -Phi1*(The1-The1_r)))+x1
    x2=h*(A@x2+B@(alp*K@(L[1,0]*(x2-x1)+L[1,2]*(x2-x3)+L[1,3]*(x2-x4)+L[1,4]*(x2-x5))
                  -Phi2*(The2-The2_r)))+x2
    x3=h*(A@x3+B@(alp*K@(L[2,0]*(x3-x1)+L[2,1]*(x3-x2)+L[2,3]*(x3-x4)+L[2,4]*(x3-x5))
                  -Phi3*(The3-The3_r)))+x3
    x4=h*(A@x4+B@(alp*K@(L[3,0]*(x4-x1)+L[3,1]*(x4-x2)+L[3,2]*(x4-x3)+L[3,4]*(x4-x5))
                  -Phi4*(The4-The4_r)))+x4
    x5=h*(A@x5+B@(alp*K@(L[4,0]*(x5-x1)+L[4,1]*(x5-x2)+L[4,2]*(x5-x3)+L[4,3]*(x5-x4))
                  -Phi5*(The5-The5_r)))+x5
        
    error_list[i,0]= np.linalg.norm(x1-x2)**2+np.linalg.norm(x2-x3)**2\
        +np.linalg.norm(x3-x4)**2+np.linalg.norm(x4-x5)**2
    
end_time=time.time()
totaltime=end_time - start_time

plt.figure()
plt.plot(t,consensus1)
plt.grid(True)
plt.xlabel("Time (seconds)")
plt.ylabel(r"$x_{i,1}$") 
plt.legend([r'$x_{1,1}$', r'$x_{2,1}$', r'$x_{3,1}$', r'$x_{4,1}$', r'$x_{5,1}$'])
#plt.title(r"Consensus of $x_{i,1}$")

plt.figure()
plt.plot(t,consensus2)
plt.grid(True)
plt.xlabel("Time (seconds)")
plt.ylabel(r"$x_{i,2}$")
plt.legend([r'$x_{1,2}$', r'$x_{2,2}$', r'$x_{3,2}$', r'$x_{4,2}$', r'$x_{5,2}$'])
#plt.title(r"Consensus of $x_{i,2}$")

plt.figure()
plt.plot(t,consensus3)
plt.grid(True)
plt.xlabel("Time (seconds)")
plt.ylabel(r"$x_{i,3}$")
plt.legend([r'$x_{1,3}$', r'$x_{2,3}$', r'$x_{3,3}$', r'$x_{4,3}$', r'$x_{5,3}$'])
#plt.title(r"Consensus of $x_{i,3}$")

plt.figure()
plt.plot(t,consensus4)
plt.grid(True)
plt.xlabel("Time (seconds)")
plt.ylabel(r"$x_{i,4}$")
plt.legend([r'$x_{1,4}$', r'$x_{2,4}$', r'$x_{3,4}$', r'$x_{4,4}$', r'$x_{5,4}$'])
#plt.title(r"Consensus of $x_{i,4}$")

plt.figure()
plt.plot(t,The1_s, color='r')
plt.axhline(The1_r, color='r', linestyle='--')
plt.plot(t,The2_s, color='b')
plt.axhline(The2_r, color='b', linestyle='--')
plt.plot(t,The3_s, color='g')
plt.axhline(The3_r, color='g', linestyle='--')
plt.plot(t,The4_s, color='purple')
plt.axhline(The4_r, color='purple', linestyle='--')
plt.plot(t,The5_s, color='orange')
plt.axhline(The5_r, color='orange', linestyle='--')
plt.grid(True)
plt.xlabel("Time (seconds)")
plt.ylabel(r"$\theta$")
plt.legend([r'$\theta_1$', r'$\theta_1^{ref}$',r'$\theta_2$', r'$\theta_2^{ref}$'\
            ,r'$\theta_3$', r'$\theta_3^{ref}$',r'$\theta_4$', r'$\theta_4^{ref}$'\
                ,r'$\theta_5$', r'$\theta_5^{ref}$'], loc='upper center', bbox_to_anchor=(1.1, 1))
plt.tight_layout()
#plt.title("Parameter estimation")



#plt.figure()
#plt.plot(error_list, color='b')
#plt.grid(True)
#plt.title("X_Error")
    




#%% Consensus with quantizer

start_time=time.time()

x1 = np.array([-6.125, 4.375, 9.875, -8.125]).T
The1 = np.array([1]).T
The1_r = np.array([3]).T

x2 = np.array([1.0, -8.5, -2.5, 10.0]).T
The2 = np.array([1]).T
The2_r = np.array([6]).T

x3 = np.array([-5.0, 7.5, -3.5, 1.0]).T
The3 = np.array([3]).T
The3_r = np.array([1.5]).T

x4 = np.array([10.125, -3.875, -2.875, -3.375]).T
The4 = np.array([2]).T
The4_r = np.array([5.5]).T

x5 = np.array([1.125, 0.125, -0.875, -0.375]).T
The5 = np.array([5]).T
The5_r = np.array([0.5]).T

sig=40

PHI1=np.zeros([2*N])
PHI2=np.zeros([2*N])
PHI3=np.zeros([2*N])
PHI4=np.zeros([2*N])
PHI5=np.zeros([2*N]) 

consensus1=np.zeros([N,5])
consensus2=np.zeros([N,5])
consensus3=np.zeros([N,5])
consensus4=np.zeros([N,5])


error_list_q = np.zeros([N,1])

x1_list=np.zeros([N,4])
for i in range(N):
    
    qx1=np.floor(x1/sig +1/2)*sig

    qx2=np.floor(x2/sig +1/2)*sig

    qx3=np.floor(x3/sig +1/2)*sig

    qx4=np.floor(x4/sig +1/2)*sig

    qx5=np.floor(x5/sig +1/2)*sig
    
    consensus1[i,0]=qx1[0]
    consensus1[i,1]=qx2[0]
    consensus1[i,2]=qx3[0]
    consensus1[i,3]=qx4[0]
    consensus1[i,4]=qx5[0]
    
    consensus2[i,0]=qx1[1]
    consensus2[i,1]=qx2[1]
    consensus2[i,2]=qx3[1]
    consensus2[i,3]=qx4[1]
    consensus2[i,4]=qx5[1]
    
    consensus3[i,0]=qx1[2]
    consensus3[i,1]=qx2[2]
    consensus3[i,2]=qx3[2]
    consensus3[i,3]=qx4[2]
    consensus3[i,4]=qx5[2]
    
    consensus4[i,0]=qx1[3]
    consensus4[i,1]=qx2[3]
    consensus4[i,2]=qx3[3]
    consensus4[i,3]=qx4[3]
    consensus4[i,4]=qx5[3]
    
    The1_s[i]=The1
    The2_s[i]=The2
    The3_s[i]=The3
    The4_s[i]=The4
    The5_s[i]=The5

    
    
    Phi1=np.array([0.4157+0.3473*np.exp(-z*i)*x1[0],0.4157+0.3473*np.exp(-z*i)*x1[1]])
    PHI1[2*i:2*(i+1)]=Phi1
    Phi2=np.array([0.4017+0.5474*np.exp(-z*i)*x2[0],0.4017+0.5474*np.exp(-z*i)*x2[1]])
    PHI2[2*i:2*(i+1)]=Phi2
    Phi3=np.array([0.0302+0.5233*np.exp(-z*i)*x3[0],0.0302+0.5233*np.exp(-z*i)*x3[1]])
    PHI3[2*i:2*(i+1)]=Phi3
    Phi4=np.array([0.1996+0.2433*np.exp(-z*i)*x4[0],0.1996+0.2433*np.exp(-z*i)*x4[1]])
    PHI4[2*i:2*(i+1)]=Phi4
    Phi5=np.array([0.2634+0.3597*np.exp(-z*i)*x5[0],0.2634+0.3597*np.exp(-z*i)*x5[1]])
    PHI5[2*i:2*(i+1)]=Phi5
    

    The1=h*(Phi1.T@B.T@P@(L[0,1]*(qx1-qx2)+L[0,2]*(qx1-qx3)+L[0,3]*(qx1-qx4)+L[0,4]*(qx1-qx5))
            -PHI1.T@PHI1*(The1-The1_r))+The1
    The2=h*(Phi2.T@B.T@P@(L[1,0]*(qx2-qx1)+L[1,2]*(qx2-qx3)+L[1,3]*(qx2-qx4)+L[1,4]*(qx2-qx5))
            -PHI2.T@PHI2*(The2-The2_r))+The2
    The3=h*(Phi3.T@B.T@P@(L[2,0]*(qx3-qx1)+L[2,1]*(qx3-qx2)+L[2,3]*(qx3-qx4)+L[2,4]*(qx3-qx5))
            -PHI3.T@PHI3*(The3-The3_r))+The3
    The4=h*(Phi4.T@B.T@P@(L[3,0]*(qx4-qx1)+L[3,1]*(qx4-qx2)+L[3,2]*(qx4-qx3)+L[3,4]*(qx4-qx5))
            -PHI4.T@PHI4*(The4-The4_r))+The4
    The5=h*(Phi5.T@B.T@P@(L[4,0]*(qx5-qx1)+L[4,1]*(qx5-qx2)+L[4,2]*(qx5-qx3)+L[4,3]*(qx5-qx4))
            -PHI5.T@PHI5*(The5-The5_r))+The5
    
    
    
    x1=h*(A@x1+B@(alp*K@(L[0,1]*(qx1-qx2)+L[0,2]*(qx1-qx3)+L[0,3]*(qx1-qx4)+L[0,4]*(qx1-qx5))
                  -Phi1*(The1-The1_r)))+x1
    x2=h*(A@x2+B@(alp*K@(L[1,0]*(qx2-qx1)+L[1,2]*(qx2-qx3)+L[1,3]*(qx2-qx4)+L[1,4]*(qx2-qx5))
                  -Phi2*(The2-The2_r)))+x2
    x3=h*(A@x3+B@(alp*K@(L[2,0]*(qx3-qx1)+L[2,1]*(qx3-qx2)+L[2,3]*(qx3-qx4)+L[2,4]*(qx3-qx5))
                  -Phi3*(The3-The3_r)))+x3
    x4=h*(A@x4+B@(alp*K@(L[3,0]*(qx4-qx1)+L[3,1]*(qx4-qx2)+L[3,2]*(qx4-qx3)+L[3,4]*(qx4-qx5))
                  -Phi4*(The4-The4_r)))+x4
    x5=h*(A@x5+B@(alp*K@(L[4,0]*(qx5-qx1)+L[4,1]*(qx5-qx2)+L[4,2]*(qx5-qx3)+L[4,3]*(qx5-qx4))
                  -Phi5*(The5-The5_r)))+x5
    
    error_list_q[i,0]= np.linalg.norm(x1-x2)**2+np.linalg.norm(x2-x3)**2\
        +np.linalg.norm(x3-x4)**2+np.linalg.norm(x4-x5)**2
    
    x1_list[i]=x1
    
end_time=time.time()
totaltimeq=end_time - start_time


plt.figure()
plt.plot(t,consensus1)
plt.grid(True)
plt.xlabel("Time (seconds)")
plt.ylabel(r"$x_{i,1}$")
plt.legend([r'$x_{1,1}$', r'$x_{2,1}$', r'$x_{3,1}$', r'$x_{4,1}$', r'$x_{5,1}$'])
#plt.title(r"Consensus of $x_{i,1}$ with quantization")

plt.figure()
plt.plot(t,consensus2)
plt.grid(True)
plt.xlabel("Time (seconds)")
plt.ylabel(r"$x_{i,2}$")
plt.legend([r'$x_{1,2}$', r'$x_{2,2}$', r'$x_{3,2}$', r'$x_{4,2}$', r'$x_{5,2}$'])
#plt.title(r"Consensus of $x_{i,2}$ with quantization")

plt.figure()
plt.plot(t,consensus3)
plt.grid(True)
plt.xlabel("Time (seconds)")
plt.ylabel(r"$x_{i,3}$")
plt.legend([r'$x_{1,3}$', r'$x_{2,3}$', r'$x_{3,3}$', r'$x_{4,3}$', r'$x_{5,3}$'])
#plt.title(r"Consensus of $x_{i,3}$ with quantization") 

plt.figure()
plt.plot(t,consensus4)
plt.grid(True)
plt.xlabel("Time (seconds)")
plt.ylabel(r"$x_{i,4}$")
plt.legend([r'$x_{1,4}$', r'$x_{2,4}$', r'$x_{3,4}$', r'$x_{4,4}$', r'$x_{5,4}$'])
#plt.title(r"Consensus of $x_{i,4}$ with quantization") 



plt.figure()
plt.plot(t,The1_s, color='r')
plt.axhline(The1_r, color='r', linestyle='--')
plt.plot(t,The2_s, color='b')
plt.axhline(The2_r, color='b', linestyle='--')
plt.plot(t,The3_s, color='g')
plt.axhline(The3_r, color='g', linestyle='--')
plt.plot(t,The4_s, color='purple')
plt.axhline(The4_r, color='purple', linestyle='--')
plt.plot(t,The5_s, color='orange')
plt.axhline(The5_r, color='orange', linestyle='--')
plt.grid(True)
plt.xlabel("Time (seconds)")
plt.ylabel(r"$\theta$")
plt.legend([r'$\theta_1$', r'$\theta_1^{ref}$',r'$\theta_2$', r'$\theta_2^{ref}$'\
            ,r'$\theta_3$', r'$\theta_3^{ref}$',r'$\theta_4$', r'$\theta_4^{ref}$'\
                ,r'$\theta_5$', r'$\theta_5^{ref}$'], loc='upper center', bbox_to_anchor=(1.1, 1))
plt.tight_layout()
#plt.title("Parameter estimation with quantization")


plt.figure()
plt.plot(t,error_list, color='b')
plt.grid(True)
plt.xlabel("Time (seconds)")
plt.ylabel(r'$X_{error}$')
plt.legend([r'$X_{error}$'])
#plt.title("Consensus Error")

plt.figure()
plt.plot(t,error_list_q, color='b')
plt.grid(True)
plt.xlabel("Time (seconds)")
plt.ylabel(r'$q(X)_{error}$')
plt.legend([r'$q(X)_{error}$'])

#plt.figure()
#plt.plot(The2_s)
#plt.axhline(The2_r, color='r', linestyle='--')
#plt.grid(True)
#plt.figure()
#plt.plot(The3_s)
#plt.axhline(The3_r, color='r', linestyle='--')
#plt.grid(True)
#plt.figure()
#plt.plot(The4_s)
#plt.axhline(The4_r, color='r', linestyle='--')
#plt.grid(True)
#plt.figure()
#plt.plot(The5_s)
#plt.axhline(The5_r, color='r', linestyle='--')
#plt.grid(True)




