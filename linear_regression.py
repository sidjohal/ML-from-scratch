class LinReg(object):
    
    def __init__(self):
        self.t0 = 200
        self.t1 = 100000
        
    # model
    def predict (self, X):
        return X@self.w
    
    # loss fn
    def loss (self, X, y):
        e = y - self.predict(X)
        return 0.5 * (e.T @ e)
    
    # evaluation
    def rmse(self,X, y):
        return np.sqrt(2/X.shape[0] * self.loss(X, y))
    
    
    # optimization_method = normal_equation
    def fit(self, X, y):
        self.w = np.linalg.pinv(X) @ y
        return self.w
    
    
    # optimization_method = gradient descent
    # gradient of Loss fn
    def calculate_gradient(self, X, y):
        return X.T @ (self.predict(X) - y)
    
    # weight update rule
    def update_weights(self, grad, lr):
        return (self.w - lr * grad)
    
    # reducing step size
    def learning_schedule(self, t):
        return self.t0 / (self.t0 + self.t1)
    
    
    # simple/batch gradient descent
    def gd(self, X, y, epochs, lr):
        self.w = np.zeros(X.shape[1])
        self.w_all = list() 
        self.err_all = list()
        
        for i in range(epochs):
            dJdw = self.calculate_gradient(X, y)
            self.w_all.append(self.w)
            self.err_all.append(self.loss(X, y))
            self.w = self.update_weights(dJdw, lr)
        
        return self.w
    
    
    #mini batch gradient descent
    def mbgd(self, X, y, epochs, batch_size):
        t = 0
        self.w = np.zeros(X.shape[-1])  
        self.w_all = []
        self.err_all = []

        for epoch in range(epochs):
            shuffled_indices = np.random.permutation(X.shape[0])
            X_shuffled = X[shuffled_indices]
            y_shuffled = y[shuffled_indices]

            for i in range(0, X.shape[0], batch_size):
                t += 1
                x1 = X_shuffled[i:i+batch_size]
                y1 = y_shuffled[i:i+batch_size]

                self.w_all.append(self.w)
                self.err_all.append(self.loss(x1, y1))

                dJdw = 2/batch_size * self.calculate_gradient(x1, y1)
                self.w = self.update_weights(dJdw, self.learning_schedule(t))

        return self.w, self.err_all
    
    
    # stochastic gradient descent
    def sgd(self, X, y, num_epochs):
        batch_size = 1
        t = 0
        self.w = np.zeros(X.shape[1])  #initializing arbitrary values.
        self.w_all = list()
        self.err_all = list()
        
        for epoch in range(num_epochs):
            shuffled_indices = np.random.permutation(X.shape[0])
            X_shuffled = X[shuffled_indices]
            y_shuffled = y[shuffled_indices]

            for i in range(0, X.shape[0], batch_size):
                t += 1
                x1 = X_shuffled[i:i+batch_size]
                y1 = y_shuffled[i:i+batch_size]

                self.w_all.append(self.w)
                self.err_all.append(self.loss(X, y))

                dJdw = 2/batch_size * self.calculate_gradient(x1, y1)
                self.w = self.update_weights(dJdw, self.learning_schedule(t))

        return self.w, self.err_all


