import numpy as np
from collections import Counter
from sklearn.datasets import load_breast_cancer as d 
from sklearn.model_selection import train_test_split as tts 
class Node:
    def __init__(self,feature=None,threshold=None,left=None,right=None,*,value=None):
        self.feature=feature
        self.threshold=threshold
        self.left=left 
        self.right=right
        self.value=value
class DecisionTree:
    def __init__(self,min_sample_split=2,max_depth=10,n_features=None):
        self.min_sample_split=min_sample_split
        self.max_depth=max_depth
        self.n_features=n_features
        self.root=None


    
    def fit(self,X,y):
        self.n_features=X.shape[1] if self.n_features is None else min(self.n_features,X.shape[1])
        
        self.root=self.grow_tree(X,y)


        return self 


    def grow_tree(self,X,y,depth=0):

        n_sample,n_features=X.shape
        n_labels=len(np.unique(y))

        if n_sample<=self.min_sample_split or n_labels==1 or depth>=self.max_depth:
            value=Counter(y).most_common()[0][0]
            return Node(value=value)


        num_features=np.random.choice(n_features,self.n_features,replace=False)
        best_feature,best_threshold=self.best_split(X,y,num_features) 


        left_idx,right_idx=self.split(X[:,best_feature],best_threshold)

        left=self.grow_tree(X[left_idx,:],y[left_idx],depth+1)
        right=self.grow_tree(X[right_idx,:],y[right_idx],depth+1)

        return Node(feature=best_feature,threshold=best_threshold,left=left,right=right)


    def best_split(self,X,y,num_features):

        best_gain=-1 

        best_feature,best_threshold=None,None

        for feature in num_features:
            X_c=X[:,feature]

            possible_threshold=np.unique(X_c)

            for threshold in possible_threshold:
                gain=self.information_gain(X_c,y,threshold)

                if gain>best_gain:
                    best_gain=gain
                    best_feature=feature
                    best_threshold=threshold


        return best_feature,best_threshold



    def information_gain(self,X_c,y,threshold):
        parent_entropy=self.entropy(y)

        left_idx,right_idx=self.split(X_c,threshold)

        if len(left_idx)==0 or len(right_idx)==0:
            return 0 


        w_l,w_r=len(left_idx)/len(y),len(right_idx)/len(y)
        e_l,e_r=self.entropy(y[left_idx]),self.entropy(y[right_idx])

        child_entropy=(w_l*e_l)+(w_r*e_r)

        information_gain=parent_entropy-child_entropy 

        return information_gain 

    def entropy(self,y):
        clss=np.bincount(y)

        clss=clss/len(y)

        return -np.sum([p*np.log(p) for p in clss if p>0])



    def split(self,X_c,threshold):
        left=np.argwhere(X_c<=threshold).flatten()
        right=np.argwhere(X_c>threshold).flatten()


        return left,right
    


    def predict(self,X):
        y_pred=np.array([self.traverse(self.root,x) for x in X])
        return y_pred
    

    def traverse(self,node,x):
        
        if node.value is not None:
            return node.value
        


        if x[node.feature]<=node.threshold:
            return self.traverse(node.left,x)
        
        return self.traverse(node.right,x)