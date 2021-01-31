import pandas as pd
import numpy as np

class naive_Bayes_Operator:

    def __init__(self,_lambda = 1):

        self.data = pd.read_excel("Data.xlsx",engine = "openpyxl")
        self.N = self.data.shape[0]
        self._lambda_ = _lambda


    # Feature:dim1,dim2,Feature's value:val1,val2 
    def conditional_probability(self,dim1,dim2,val1,val2):

        #Feature 1 and Feature 2
        p12 = self.data.loc[(self.data[self.data.columns[dim1]]==val1) & (self.data[self.data.columns[dim2]]==val2)].shape[0] + self._lambda_            

        #Get Sj
        Sj = self.data.iloc[:,dim1].unique().size
        
        
        #Feature 2
        p2 = self.data.loc[self.data[self.data.columns[dim2]]==val2].shape[0] + Sj*self._lambda_
        
        return (p12/p2)

    #Feature: dim , Feature's value: val
    def probability(self,dim,val):

        N = self.data.shape[0]

        K = self.data.iloc[:,0].unique().size
            
        return ((self.data.loc[self.data[self.data.columns[dim]] == val].shape[0] + self._lambda_) / (N + K*self._lambda_))



    #naive_Bayes_Operator: sample: list
    def _Operator_(self,sample):

        max_p = 0

        for i in range(self.data.iloc[:,0].unique().size):

            p = 1
            data_class = self.data.iloc[:,0].unique()[i]

            for j in range(len(sample)):

                p *= self.conditional_probability(j+1,0,sample[j],data_class)

            
            p *= self.probability(0,data_class)

            if p > max_p:
                
                max_p = p
                class_p = data_class


        return class_p
