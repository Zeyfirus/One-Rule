import pandas as pd
import numpy as np
N,M=list(map(int,input().split()))
colum1=np.array(input().split())
colum=np.append(colum1,'result')
data=[]
for _ in range (N):
    data.append(input().split())
df=pd.DataFrame(data=data,columns=colum)
K=int(input())
data_2=[]
for _ in range (K):
    data_2.append(input().split())
df_2=pd.DataFrame(data=data_2,columns=colum1)
class OneRuleClassificator:
    def __init__(self):
        pass
    def fit(self, X, y):
        unique =[]
        for i in X.T:
            unique.append(list(set(i)))
        means=[[list() for _ in range (len(i))] for i in unique]
        for i,j,m in zip(unique,X.T,means):
            for v,a in zip (j,y):
                m[i.index(v)].append(a)
            for i in range (len(m)):
                if sum(map(int,m[i]))>=len(m[i])/2:
                    m[i]=1
                else:
                    m[i]=0
        errors=[[] for _ in unique]
        for x,a in zip(X,y):
            for i,j,e,m in zip (unique,x,errors,means):
                e.append((m[i.index(j)]-int(a))**2)
        for i in range (len(errors)):
            errors[i]=sum(errors[i])/len(errors[i])
        self.index = np.argmin(errors)
        self.classification = means[self.index]
        self.unique = unique[self.index]
        return self
    def predict(self, X):
        result=[]
        for i in X:
            result.append(self.classification[self.unique.index(i[self.index])])
        return result
a=(colum for colum in colum1)
X=df[a].to_numpy()
Y=df['result'].to_numpy()
X_2=df_2.to_numpy()
model=OneRuleClassificator().fit(X, Y)
for i in model.predict(X_2):
   print(i)