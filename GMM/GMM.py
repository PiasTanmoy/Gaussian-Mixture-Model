
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt



# In[100]:



def datasetGeneration(k, numberOfDataPoint):
    '''
    Pick 3 set of mean u, daviation SD 
    2 attributes/features. u = 2 columns. u = 3 rows
    E = 3*3
    Diagonal matrix
    '''
    
    u = []
    E = np.array(np.zeros((k, 2, 2)))
    x = []
    y = []
    wx = 0;
    for i in range(k):
        x1 = np.random.randint(10+ i*30, i*30+100)
        x2 = np.random.randint(10+ i*30, i*30+100)
        ux = [x1, x2]
        u.append(ux)
        
        e1 = np.random.randint(50, 120)
        e2 = np.random.randint(50, 120)

        E[i][0][0] = e1
        E[i][1][1] = e2
        
        x3, y3 = np.random.multivariate_normal(u[i], E[i], int(numberOfDataPoint/k)).T
        plt.plot(x3, y3, '.')
        plt.plot(x1, x2, 'X')
        #print(x3.size, y3.size)
        x = np.concatenate((x ,x3))
        y = np.concatenate((y,y3))

        
    dataset = np.array(list(zip(x, y)))
    #print(len(dataset))
        
    plt.axis('equal')
    plt.savefig("fileName")
    plt.show()
    #print(dataset)
        

    return (u, E, dataset)


numberOfClusters = 3
numberOfDatapoint = 600
u_source, E_source, dataset = datasetGeneration(numberOfClusters, numberOfDatapoint)
w = [0.3, 0.3, 0.4]
#print(dataset)
print(E_source)

x = dataset
u_source = np.array(u_source)
print(u_source)


# In[4]:


def N(xj, ui, Ei):
    D = 2
    part1 = 1/(np.sqrt( np.power( (2*np.pi), D ) * determinant(Ei) ))
    Einverse = np.linalg.inv(Ei)
    part2 = np.reshape(xj-ui, (2,1))
    part3 = (xj-ui)
    part4 = np.matmul(Einverse, part2)
    part5 = np.matmul(part3, part4)
    #print(-.5*part5)
    part6 = np.exp(-.5*part5)
    ans = part1*part6
    
    return ans[0]
    
    
    

def determinant(MAT):
    return np.linalg.det(MAT)


##check

xj_test = np.array([52, 100])
ui_test = np.array([20, 52])
Ei_test = np.array([[ 50. , 0.],[ 0., 80.]])

N(xj_test, ui_test, Ei_test )


# In[122]:


# Step 1
# Randomly initialize everything


def randomInitialization(k):
    '''
    Pick 3 set of mean u, daviation SD 
    2 attributes/features. u = 2 columns. u = 3 rows
    E = 3*3
    Diagonal matrix
    '''
    
    u = []
    E = np.array(np.zeros((k, 2, 2)))
    x = []
    y = []
    #w = [0.3, 0.3, 0.4]
    w = np.zeros((1, k))[0]
    w = np.array(w)
    sum_w=0;
    for i in range(k):
        w[i] = 1
    for i in range(k):
        sum_w += w[i]
    for i in range(k):
        w[i] = (w[i] / sum_w)
        
        
    wx = 0;
    for i in range(k):
        x1 = np.random.randint(40+10*i, k*20+50)
        x2 = np.random.randint(40+10*i, k*20+50)
        ux = [x1, x2]
        u.append(ux)
        
        e1 = np.random.randint(50, 100)
        e2 = np.random.randint(50, 100)

        E[i][0][0] = e1
        E[i][1][1] = e2
        
    return (u, E, w)

k = 3
u, E, w = randomInitialization(k)
print(u, E, w)


# In[ ]:


# E step
allP=np.zeros((k, len(dataset)))
numberOfDatapoint = 600
numberOfClusters = 3
u_source, E_source, dataset = datasetGeneration(numberOfClusters, numberOfDatapoint)
dataset = np.array(dataset)



def p_ij(i, xj, u, E, w, k):
    part1 = w[i]*N(xj, u[i], E[i])
    part2 = 0
    for i in range(k):
        part2 += w[i]*N(xj, u[i], E[i])
    #print(part1/part2)
    return (part1/part2)

def getAllP(dataset, u, E, w, k):
    allP=np.zeros((k, len(dataset)))
    for i in range(k):
        for j in range(len(dataset)):
            allP[i][j] = p_ij(i, dataset[j], u, E, w, k)
    
    return allP

k = 3
u, E, w = randomInitialization(k)

allP = getAllP(dataset, u, E, w, k)

        


# In[7]:


# M step
def calculateU(i, allP, dataset):
    sum_p = 0
    u_m = np.zeros((1, 2))
    N = len(dataset)
    
    for j in range(N):
        sum_p += allP[i][j]
    
    for j in range(N):
        xj = dataset[j]
        pij = allP[i][j]
        u_m += (pij*xj)
            
    return (u_m/sum_p)

def calculateCov(i, allP, dataset, u):
    E_m = np.zeros((2, 2))
    sum_p = 0
    N = len(dataset)
    
    for j in range(N):
        sum_p += allP[i][j]
            
            
    for j in range(N):
        xj = dataset[j]
        ui = u[i]
        pij = allP[i][j]
        
        part1 = np.reshape(xj-ui, (2,1))
        part2 = part1.T
        part3 = np.matmul(part1, part2)
        part3 = pij*part3
        E_m += part3
    
    ans = E_m / sum_p
    return (ans)

def calculateWeight(i, allP, dataset):
    sum_p=0
    N = len(dataset)
    for j in range(N):
        sum_p += allP[i][j]
    
    return (sum_p/N)


# In[8]:


# E step
import math
def logLikelihood(dataset, k, u, E, w):
    sum_n = 0
    sum_log = 0
    n = len(dataset)
    for j in range(n):
        sum_n = 0
        for i in range(k):
            #def N(xj, ui, Ei):
            #print("N: " , dataset[j], u[i], E[i])
            part1 = N(dataset[j], u[i], E[i])
            part1 = w[i]*part1
            #print("part1 " , part1) 
            sum_n += part1
        #print("sum_n: " , sum_n)  
        #print("sum_n: " , sum_n)
        part2 = math.log(sum_n)
        sum_log += part2
    #print("sum_log: " , sum_log)  
    return sum_log


# In[63]:


print(dataset)
print(u, E, w)


# In[156]:


u_m = np.zeros((2, 2))
print(u_m)


# In[131]:


f = open('workfile', 'w')


numberOfDatapoint = 1000
numberOfClusters = 3
k = numberOfClusters

u_source, E_source, dataset = datasetGeneration(numberOfClusters, numberOfDatapoint)
dataset = np.array(dataset)
plt.savefig("fileName")
print("u_source: " , u_source,"\nE_source: ",  E_source)

u, E, w = randomInitialization(k)
u = np.array(u)
print("Initialization\n")
print("u: ", u,"\nE: ", E,"\nw: ", w)

cur = logLikelihood(dataset, k, u, E, w)
prev = 0
#while(np.abs(cur - prev) > 0.1):


#for runTest in range(6):
count=0;
while(np.abs(cur - prev) > 0.001):
    allP = getAllP(dataset, u, E, w, k)
    allP = np.array(allP)
    #def calculateU(i, allP, dataset):
    #def calculateCov(i, allP, dataset, u):
    #def calculateWeight(i, allP, dataset):

    for i in range(k):
        u[i] = calculateU(i, allP, dataset)
        u = np.array(u)
        
        E[i] = calculateCov(i, allP, dataset, u)
        E = np.array(E)
        
        w[i] = calculateWeight(i, allP, dataset)
        w = np.array(w)
    
    print("u\n", u, "E\n", E, "w\n", w)
    #f.write("u\n", u, "E\n", E, "w\n", w)
    #plt.plot()
    
    #def logLikelihood(dataset, k, u, E, w):
    print('logLikelihood: ', cur)
    f.write( str(cur))
    
    prev = cur
    cur = logLikelihood(dataset, k, u, E, w)
    
    x, y = zip(*dataset)
    plt.plot(x, y, '.', 'b')
    plt.title('Iteration ' + str(count))
    
    for i in range(k):
        plt.plot(u[i][0], u[i][1], 'o', 'm')
        
    #plt.plot(u[0][0], u[0][1], 'o', 'm')
    #plt.plot(u[1][0], u[1][1], 'o', 'm')
    #plt.plot(u[2][0], u[2][1], 'o', 'm')

    for i in range(k):
        plot_cov_ellipse(E[i], u[i], 2, None, alpha=0.5)
    
    fileName = 'data' + str(count) + '.png'
    plt.savefig(fileName)
    count+=1
    plt.show()

print("u: ", u,"\nE: ", E,"\nw: ", w)




# In[381]:


x, y = zip(*dataset)
plt.plot(x, y, '.')

for i in range(k):
    plot_cov_ellipse(E[i], u[i], 2, None, alpha=0.5)
plt.show()


# In[374]:


print(dataset)
x, y = zip(*dataset)
#print(x)
plt.plot(x, y, '.')


# In[330]:


u, E, w = randomInitialization(k)
print(u,E, w)
prev = logLikeliVal = logLikelihood(dataset, k, u, E, w)
print(prev)


# In[60]:


import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip



# In[110]:




cov = [[[99. , 0.],
  [ 0., 54.]],

 [[80. , 0.],
  [ 0., 71.]],

 [[71. , 0.],
  [ 0. ,97.]]] 
pos = [[103 , 82],[ 68 , 96],[ 99 ,130]] 
x3, y3 = np.random.multivariate_normal(pos[0], cov[0], int(300)).T
plt.plot(x3, y3, '.')
#print(cov[0])
plot_cov_ellipse(cov[0], pos[0], 2, None, alpha=0.5)
plt.plot(pos[0][0], pos[0][1], 'X', 'r')
plt.show()

print(pos[0][0], pos[0][1])


# In[116]:


w = np.zeros((1, 4))
w = np.array(w)
print(w)

