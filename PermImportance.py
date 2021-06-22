import numpy as np
import matplotlib.pyplot as plt
from xgboost import DMatrix
#####
##
## Modified code of https://github.com/aghoshpub/permutationImportancePhysics
## I have updated the code so that it works with data frames and uses DMatrices to be compatible with XGBoost
##
####


class PermulationImportance:
    def __init__(self, model, X, y, weights=None, n_iterations=3, scoreFunction="AUC", usePredict_poba=False):
        self.X = X
        self.y = y
        self.model = model
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.ones(len(X))
        self.n_iterations = n_iterations

        if scoreFunction == "AUC":
            self.scoreFunction = self.auc_score
        elif scoreFunction == "amsasimov":
            self.scoreFunction = self.significance_score
        elif scoreFunction == "F1":
            self.scoreFunction = self.F1_score            
        else:
            self.scoreFunction = scoreFunction

        self.usePredict_poba = usePredict_poba
        self.mean = None
        self.error = None
        self.std = None
        self.colNames = X.columns
        print(self.colNames)

    def auc_score(self, X_eval, y_true, weights):
        #@FIXME: Use AUC that can handle negative weights instead
        from sklearn.metrics import roc_auc_score
       
        X_eval = DMatrix(X_eval)
        if self.usePredict_poba:
            y_pred = self.model.predict_proba(X_eval)[:,1].ravel() # do here to allow flexibility for custom score function
        else:
            y_pred = self.model.predict(X_eval).ravel()
        return roc_auc_score(y_score=y_pred, y_true=y_true, sample_weight=weights)

    def significance_score(self, X_eval, y_true, weights):
        def amsasimov(s,b):
            from math import sqrt,log
            if b<=0 or s<=0:
                return 0
            try:
                return sqrt(2*((s+b)*log(1+float(s)/b)-s))
            except ValueError:
                print (1+float(s)/b)
                print (2*((s+b)*log(1+float(s)/b)-s))
            #return s/sqrt(s+b)
        
        X_eval = DMatrix(X_eval)
        if self.usePredict_poba:
            y_pred = self.model.predict_proba(X_eval)[:,1].ravel() # do here to allow flexibility for custom score function
        else:
            y_pred = self.model.predict(X_eval).ravel()
        first, last, n_cuts = 0.2, 1., 30
        #@TODO: Histogram loop with numba or parallelise
        int_sig = [weights[(y_true ==1) & (y_pred > th_cut)].sum() for th_cut in np.linspace(first,last,num=n_cuts)]
        int_bkg = [weights[(y_true ==0) & (y_pred > th_cut)].sum() for th_cut in np.linspace(first,last,num=n_cuts)]
        vZ = [amsasimov(s=sumsig,b=sumbkg) for (sumsig,sumbkg) in zip(int_sig,int_bkg)]
        bestiZ = max(vZ)
        return bestiZ
    
    def F1_score(self, X_eval, y_true, weights):
        #@FIXME: Use AUC that can handle negative weights instead
        from sklearn.metrics import f1_score
       
        X_eval = DMatrix(X_eval)
        if self.usePredict_poba:
            y_pred = self.model.predict_proba(X_eval)[:,1].ravel() # do here to allow flexibility for custom score function
        else:
            y_pred = self.model.predict(X_eval).ravel()
        
        y_bin = (y_pred > 0.1).astype(int)
        return f1_score(y_true,y_bin,sample_weight=weights)
    
    def shuffle_column(self, X):
       
       for col in X.columns:
            hold = X[col].copy()
            X[col] = np.random.permutation(X[col])
            yield X
            X[col] = hold

    def evaluate(self):
        from math import sqrt
        scores = []
        trueScore = self.scoreFunction(X_eval=self.X, y_true=self.y, weights=self.weights)
        # TODO: parallelise
        for i in range (self.n_iterations):
            # TODO: parallelise
            scores.append( [self.scoreFunction(X_eval=x, y_true=self.y, weights=self.weights)
                           for x in self.shuffle_column(self.X)])
        scores_drop = trueScore - np.array(scores)
        self.mean, self.std = np.mean(scores_drop, axis=0), np.std(scores_drop, axis=0)
        self.error = self.std/sqrt(self.n_iterations)
        return self.mean, self.error

    def dislayResults(self,colNames=None, asc=False):
        #@TODO: add background colours
        if self.mean is None:
            self.evaluate()
        if colNames is None:
            colNames = self.colNames
        colNames = np.array(colNames).reshape(-1,1)
        # this converts numbers to text,  can use pandas intead, with title per column
        table = np.concatenate([self.mean.reshape(-1,1), self.error.reshape(-1,1), self.std.reshape(-1,1), colNames], axis=1)
        if (asc):
            ind = np.argsort( self.mean ) # ascending order
        else:
            ind = np.argsort( self.mean )[::-1] # decending order
        table = table[ind] # sort table my mean
        print (table)
        return table
    def plotBars(self,colNames=None,PlotTitle=None):
        if self.mean is None:
            self.evaluate()
        if colNames is None:
            colNames = self.colNames
        colColours = ['g' if m > 0 else 'r' for m in self.mean]
        x_pos = np.arange(len(colNames))
        fig, ax = plt.subplots()
        ax.bar(x_pos, self.mean, yerr=self.error, align='center', alpha=0.5,
               ecolor='black', capsize=10, color=colColours)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(colNames, rotation = 90)
        if PlotTitle is not None:
            ax.set_title(PlotTitle)
            
        return plt
