import numpy as np 


class NeuralNet:
    def __init__(self,n_hidden,epochs=100,eta=0.1):
        self.n_hidden=n_hidden 
        self.epochs=epochs
        self.eta=eta

    def fit(self,X,y):
        n_sample,n_features=X.shape
        n_classes=len(np.unique(y))
        self.classes_=np.unique(y)


        y_en=np.zeros(shape=(n_classes,n_sample))

        for idx,c in enumerate(y):
            y_en[c,idx]=1

        y=y_en
        
        # print(y)

        rgen=np.random.RandomState(seed=1)
        self.w1=rgen.normal(loc=0.0,scale=0.1,
                             size=(self.n_hidden,n_features))
        self.b1=rgen.normal(loc=0.0,scale=0.1,
                             size=(self.n_hidden,1))
        
        self.w2=rgen.normal(loc=0.0,scale=0.1,
                             size=(n_classes,self.n_hidden))
        self.b2=rgen.normal(loc=0.0,scale=0.1,
                             size=(n_classes,1))
        


        for _ in range(self.epochs):
            losses=0
            for idx,x in enumerate(X):
                x_temp=x.T.reshape(-1,1)
                y_temp=y[:,idx].reshape(-1,1)

                a_2,z_2,a_1,z_1=self.forward_propagation(x_temp)
                
            #   print(y[:,idx],a_2)
                losses+=self.calculate_loss(a_2,y_temp)

                self.backward_propagation(a_2,z_2,a_1,z_1,x_temp,y_temp)
            
            print(f'Loss :{losses}')
            pass

        return self
    


    def forward_propagation(self,x):
        
        z1=np.dot(self.w1,x)+self.b1
        
        a1=self.sigmoid(z1)


        z2=np.dot(self.w2,a1)+self.b2 
        a2=self.sigmoid(z2)

        


        return a2,z2,a1,z1 
    
    
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))


    def calculate_loss(self,a,y):
        loss=np.sum((y*(np.log(a)))-((1-y)*(np.log(1-a))),axis=0)
        # print(loss[0])
        return loss[0]

        
    def backward_propagation(self,a2,z2,a1,z1,X,y):
        
        dz2=a2-y  
        


        dw2=np.dot(dz2,a1.T)

        db2=np.sum(dz2,keepdims=True)

        dz1=np.dot(self.w2.T,dz2) * (a1*(1-a1))

        dw1=np.dot(dz1,X.T)

        db1=np.sum(dz1,keepdims=True)

        


        self.update(dw2,db2,dw1,db1)


    def update(self,dw2,db2,dw1,db1):
        self.w2=self.w2+(-self.eta*dw2)
        self.b2=self.b2+(-self.eta*db2)

        self.w1=self.w1+(-self.eta*dw1)
        self.b1=self.b1+(-self.eta*db1)

    
    def predict(self,X):
        
        
        y_pred=[]
        for idx,x in enumerate(X):
          x_temp=x.T.reshape(-1,1)
          a2,z2,a1,z1=self.forward_propagation(x_temp)
          print()

          pred=self.classes_[np.argmax(a2,axis=0)[0]]
         

          y_pred.append(pred)

        return y_pred
    

    def threshold_function(self,a):
        return np.where(a>=0.5,1,0)

        




        






      


