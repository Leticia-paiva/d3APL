import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Tuple

class LogisticRegression(ClassifierMixin, BaseEstimator):
    """Our Logistic Regression implemented from scratch."""
    
    def __init__(self, learning_rate: float = 0.01,
                 n_epochs: int = 100, alpha: float = 0.01, random_state : int = 42, optimizer: str = "batch"):
        """
        Parameters
        ----------
        learning_rate : float, default=0.01
            Learning rate.
        n_epochs : int, default=100
            Number of epochs for training (convergence stop).
        alpha : float, default=0.01
            Constant that multiplies the regularization term.
            Use 0 to ignore regularization (standard Logistic Regression).
        random_state : int, default=42
            Seed used for generating random numbers.
        """
        assert (learning_rate is not None) and (learning_rate > 0.0), \
        f'Learning rate must be > 0. Passed: {learning_rate}'
        
        assert (n_epochs is not None) and (n_epochs > 0), \
        f'Number of epochs must be > 0. Passed: {n_epochs}'
        
        assert (alpha is not None) and (alpha >= 0), \
        f'Alpha should be >= 0. Passed: {alpha}'
        
        optList = ['batch', 'mini', 'stochastic']
        assert (optimizer is not None) and (optimizer in optList), \
        f'Optimizer one of {optList}. Passed.: {optimizer}'
        
        # public ==> todo mundo tem acesso para leitura e escrita direta
        # private ==> apenas a classe tem acesso para leitura e escrita direta
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.alpha = alpha
        self.random_state = random_state
        self.optimizer = optimizer
        
        # parameters to be trained/learned
        self.classes_ = []
        self.__indexes_dict = {}
        self.__w = None  # weight array
        self.__b = None  # bias
        self.__w_dict = {}
        self.__b_dict = {}
        
    
    # a special method used to represent a class object as a string, called with print() or str()
    def __str__(self):
        msg = f'Learning rate: {self.learning_rate}\n' \
              f'Number of epochs: {self.n_epochs}\n' \
              f'Regularization constant (alpha): {self.alpha}\n' \
              f'Optimizer: {self.optimizer}\n\n' \
              f'Random state: {self.random_state}\n\n' \
              f'Trained?: {self.is_fitted()}\n'
        return msg
    
    
    # getter: access the function as an attribute - it is not possible to set values through it
    @property
    def coef_(self) -> ndarray:
        """Return the weight matrix (learned parameters) if the estimator was fitted/trained.
           Otherwise, raise an exception.
        """
        assert self.is_fitted(), 'The instance is not fitted yet.'
        return self.__w_dict
    
    
    # getter: access the function as an attribute - it is not possible to set values through it
    @property
    def intercept_(self) -> float:
        """Return the bias (learned intercepet) if the estimator was fitted/trained.
           Otherwise, raise an exception.
        """
        assert self.is_fitted(), 'The instance is not fitted yet.'
        return self.__b
    
    
    def is_fitted(self) -> bool:
        return self.__w is not None
    
    
    def __sigmoid(self, z: ndarray) -> ndarray:
        return 1 / (1 + np.e ** (-z))
    
    
    def __log_loss(self, y: ndarray, p_hat: ndarray, eps: float = 1e-15):
        '''Return the log loss for a given estimation and ground-truth (true labels).
        
        log is undefined for 0. Consequently, the log loss is undefined for `p_hat=0` (because of log(p_hat)) and `p_hat=1` (because of ln(1 - p_hat)).
        To overcome that, we clipped the probabilities to max(eps, min(1 - eps, p_hat)), where `eps` is a tiny constant.
        Parameters
        ----------
        y : ndarray, shape (n_samples,)
            True labels of input samples.
        p_hat : ndarray
            Estimated probabilities of input samples.
        eps : float, default=1e-15
            Epsilon term used to avoid undefined log loss at 0 and 1.
        
        Returns
        -------
        log_loss : float
            Computed log loss.
        '''
        
        p_hat_eps = np.maximum(eps, np.minimum(1 - eps, p_hat))
        
        # shape: (n_samples,)
        losses = -(y * np.log(p_hat_eps) + (1 - y) * np.log(1 - p_hat_eps))
        log_loss = losses.mean()
        
        return log_loss
    
    # function to create a list containing mini-batches
    def __create_mini_batches(self, X, y, batch_size):
        mini_batches = []
        data = np.column_stack((X, y))
        np.random.shuffle(data)
        n_minibatches = data.shape[0] // batch_size
        i = 0

        for i in range(n_minibatches):
            mini_batch = data[i * batch_size:(i + 1)*batch_size, :]
            X_mini = mini_batch[:, :-1]
            Y_mini = mini_batch[:, -1].reshape((-1, 1))
            mini_batches.append((X_mini, Y_mini))
        if data.shape[0] % batch_size != 0:
            mini_batch = data[i * batch_size:data.shape[0]]
            X_mini = mini_batch[:, :-1]
            Y_mini = mini_batch[:, -1].reshape((-1, 1))
            mini_batches.append((X_mini, Y_mini))
        return mini_batches    
    
    def __gradient(self, X: ndarray, y: ndarray, p_hat: ndarray,
                   w: ndarray, alpha: float) -> Tuple[ndarray, float]:
        '''Compute the gradient vector for the log loss with regards to the weights and bias.
        
        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data.
        y: ndarray of shape (n_samples,).
            Target (true) labels.
        p_hat : ndarray, shape (n_samples,)
            Estimated probabilities.
        w : ndarray, shape (n_features,)
            Weight array.
        alpha : float
            Reguralization constant.       
        
        Returns
        -------
        Tuple[ndarray, float]: 
            Tuple with:
            - a numpy array of shape (n_features,) containing the partial derivatives w.r.t. the weights; and
            - a float representing the partial derivative w.r.t. the bias.
        '''
        # X.shape: (n_samples, n_features)
        # y.shape == p_hat.shape: (n_samples,)
        n_samples = X.shape[0]
        
        regularization = alpha * w
        
        error = p_hat - y  # shape (n_samples,)
        grad_w = (np.dot(error, X) / n_samples) + regularization  # shape (n_features,)
        grad_b = error.mean()  # float
        
        return grad_w, grad_b

    def fit(self, X: ndarray, y: ndarray, verbose: int = 0):
        for i,val in enumerate(np.unique(y)):
            self.classes_.append(val)
            self.__indexes_dict[val] = i
            y_copy = np.zeros(y.shape)
            y_copy[y == val] = 1
            self.__fit(X, y_copy, verbose=verbose)
            self.__w_dict[i] = self.__w
            self.__b_dict[i] = self.__b
        
    def __fit(self, X: ndarray, y: ndarray, verbose: int = 0):
        '''Train a Logistic Regression classifier.
        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data.
        y: ndarray of shape (n_samples,).
            Target (true) labels.
        verbose: int, default=0
            Verbose flag. Print training information every `verbose` iterations.
            
        Returns
        -------
        self : object
            Returns self.
        '''
        ### CHECK INPUT ARRAY DIMENSIONS
        assert X.ndim == 2, f'X must be 2D. Passed: {X.ndim}'
        assert y.ndim == 1, f'y must be 1D. Passed: {y.ndim}'
        assert X.shape[0] == y.shape[0], \
            f'X.shape[0] should be equal to y.shape[0], instead: {X.shape[0]} != {y.shape[0]}'
        # alternatively
        # X, y = check_X_y(X, y)

        ### SETTING SEED
        np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape

        ### PARAMETER INITIALIZATION
        # return values from the “standard normal” distribution.
        w = np.random.randn(n_features)  # shape: (n_features,)
        b = 0.0
        
        # array that stores the loss of each epoch
        losses = []
        
        #define batch_size, default n_samples
        if self.optimizer == 'stochastic':
            batch_size = 1
        elif self.optimizer == 'mini':
            batch_size = min (20, n_samples)
        else:
            batch_size = n_samples
        
        # LEARNING ITERATIONS
        for epoch in np.arange(self.n_epochs):
            
            #create mini-batches:
            batches = self.__create_mini_batches(X, y, batch_size)
            
            #iterate on mini-batches
            for batch in batches:
                X_mini, y_mini = batch
                y_mini = y_mini.reshape(y_mini.shape[0],)
                ### ESTIMATION (FORWARD PASS)
                # X.shape == (n_samples, n_features)
                # w.shape == (n_features,)
                z = np.dot(X_mini, w) + b  # shape: (n_samples,)
                p_hat = self.__sigmoid(z)
                
                loss_epoch = self.__log_loss(y_mini, p_hat)
                losses.append(loss_epoch)
                
                ### GRADIENT DESCENT UPDATES (BACKWARD PASS)
                # grad_w.shape: (n_features,)
                # grad_b: float
                grad_w, grad_b = self.__gradient(X_mini, y_mini, p_hat, w, self.alpha)
                w = w - self.learning_rate * grad_w  # shape: (n_features)
                b = b - self.learning_rate * grad_b  # float
            
          ### ASSIGN THE TRAINED PARAMETERS TO THE PRIVATE ATTRIBUTES
        self.__w = w
        self.__b = b

    def __predict_proba(self, X: ndarray, w, b) -> ndarray:
        '''Estimate the probability for the positive class of input samples.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        ndarray of shape (n_samples,)
            The estimated probabilities for the positive class of input samples.
        '''
        assert self.is_fitted(), 'The instance is not fitted yet.'
        assert X.ndim == 2, f'X must b 2D. Passed: {X.ndim}'

        z = np.dot(X, w) + b
        p_hat = self.__sigmoid(z)
        
        return p_hat

    def predict_proba(self, X: ndarray) -> ndarray:
        '''Estimate the probability for the positive class of input samples.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        ndarray of shape (n_samples,)
            The estimated probabilities for the positive class of input samples.
        '''
        assert self.is_fitted(), 'The instance is not fitted yet.'
        assert X.ndim == 2, f'X must b 2D. Passed: {X.ndim}'

        p_hat_dict = {}
        for i,class_name in self.__indexes_dict.items():
            p_hat = self.__predict_proba(X, self.__w_dict[i], self.__b_dict[i])
            p_hat_dict[i] = p_hat
        p_hat_df = pd.DataFrame(p_hat_dict)
        y_prob = np.array([max(p_hat_df.loc[idx, :]) for idx in p_hat_df.index], dtype=float)
        return y_prob

    def predict(self, X: ndarray) -> ndarray:
        '''Predict the labels for input samples.
        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted labels of input samples.
        '''
        assert self.is_fitted(), 'The instance is not fitted yet.'
        assert X.ndim == 2, f'X must b 2D. Passed: {X.ndim}'

        p_hat_dict = {}
        for i,class_name in self.__indexes_dict.items():
            p_hat = self.__predict_proba(X, self.__w_dict[i], self.__b_dict[i])
            p_hat_dict[i] = p_hat
        p_hat_df = pd.DataFrame(p_hat_dict)
        y_hat = np.array([self.classes_[np.argmax(p_hat_df.loc[idx, :])] for idx in p_hat_df.index], dtype=int)
        return y_hat