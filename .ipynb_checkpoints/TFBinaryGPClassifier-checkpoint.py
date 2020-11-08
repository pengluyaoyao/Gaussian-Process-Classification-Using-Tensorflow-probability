import warnings
import math
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
import numpy as np


tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

tf.enable_v2_behavior()

class TFBinaryGPClassifier(object):
    
    def __init__(self, kernel, optimizer='fmin_l_bfgs_b', max_iter_predict=100,
                 warm_start=True, copy_X_train=True, random_state=None):
        
        self.kernel = kernel
        self.optimizer = optimizer
        self.max_iter_predict = max_iter_predict
        self.warm_start = warm_start
        self.copy_X_train = copy_X_train
        self.random_state = random_state
        
    def fit(self, X, y):
              
        self.X_train_ = X        
        self.y = y
        self.classes = tf.unique(self.y)[0]
        self.start = np.array([0.]*2, dtype='float64')
        
        if tf.math.greater(self.classes.shape, 2):
            raise ValueError('Only supports binary classification, y contain classes %s' % self.classes.shape)
        
        if self.optimizer is not None and self.kernel.feature_ndims>0:
            
            def make_val_and_grad_fn(params):
                val_and_grad = tfp.math.value_and_gradient(self.log_marginal_likelihood, params)
                val_and_grad = (-val_and_grad[0], -val_and_grad[1])
                return val_and_grad
            
            optim_results = tfp.optimizer.lbfgs_minimize(
                make_val_and_grad_fn, initial_position=self.start, f_relative_tolerance=1e-7)

            self.log_marginal_likelihood_value_ = optim_results.objective_value
            self.converged = optim_results.converged
            
            if self.converged==False:
                warnings.warn("L-BFGS is not converged.")
                
            self.kernel._amplitude = tf.math.exp(optim_results.position[0])
            self.kernel._length_scale = tf.math.exp(optim_results.position[1:])
            
        self.K = self.kernel.matrix(self.X_train_, self.X_train_)
        
        _, _, (self.pi_, self.W_sr_, self.L_, _, _) = self.posterior_mode(self.K[0], return_temporaries=True) 
        
        return self
    
    def predict(self, X):
        
        K_star = self.kernel.matrix(tf.constant(self.X_train_, dtype='float64'), tf.constant(X, dtype='float64'))
        f_star = tf.matmul(tf.transpose(K_star[0]), tf.reshape(self.y - self.pi_, [-1,1]))
        f_star_flat = tf.reshape(f_star, [-1])
        
        return tf.where(tf.math.greater(f_star_flat, 0), 1, 0)
    
    def predict_proba(self, X):

        K_star = self.kernel.matrix(tf.constant(self.X_train_, dtype='float64'), tf.constant(X, dtype='float64'))[0]
        f_star = tf.matmul(tf.transpose(K_star), tf.reshape(self.y - self.pi_, [-1,1]))
        
        v = tf.linalg.solve(self.L_, tf.expand_dims(self.W_sr_, -1) * K_star)
        var_f_star = tf.linalg.diag_part(self.kernel.matrix(tf.constant(X, dtype='float64'), tf.constant(X, dtype='float64'))[0])- tf.reduce_sum(tf.transpose(v)**2, 1)
        
        LAMBDAS = np.array([0.41, 0.4, 0.37, 0.44, 0.39])[:, np.newaxis]
        COEFS = np.array([-1854.8214151, 3516.89893646, 221.29346712,
                          128.12323805, -2010.49422654])[:, np.newaxis]

        alpha = 1 / (2 * var_f_star)
        gamma = LAMBDAS * tf.reshape(f_star, -1)
        integrals = tf.math.sqrt(tf.constant(math.pi, dtype='float64') / alpha) \
                    * tf.math.erf(gamma * tf.math.sqrt(alpha / (alpha + LAMBDAS**2))) \
                    / (2 * tf.math.sqrt(var_f_star * 2 * tf.constant(math.pi, dtype='float64')))
        pi_star = tf.reduce_sum(COEFS * integrals, axis=0) + .5 * tf.reduce_sum(COEFS)
        probs_preds = tf.stack([1 - pi_star, pi_star], axis=1)

        return probs_preds#, np.where(f_star > 0, 1, 0)

    def log_marginal_likelihood(self, params):
    
        amplitude = tf.math.exp(params[0])
        length_scale = tf.math.exp(params[1:])

        self.kernel._amplitude = amplitude
        self.kernel._length_scale = length_scale
        K = self.kernel.matrix(tf.constant(self.X_train_, dtype='float64'), tf.constant(self.X_train_, dtype='float64'))

        f_mode, Z, _ = self.posterior_mode(K=K[0], return_temporaries=False)

        return Z[0,0] 
    
    def posterior_mode(self, K, return_temporaries=False):
        
        n = self.X_train_.shape[0]    
        if self.warm_start and hasattr(self, "f_cached"):
            f = self.f_cached
        else:
            f = tf.zeros(n, dtype=np.float64)
        log_marginal_likelihood = tf.constant(-np.inf, dtype='float64')

        for i in range(self.max_iter_predict):

            pi = tf.sigmoid(f)
            W = pi*(1-pi)
            W_sr = tf.sqrt(W)
            W_sr_K = tf.reshape(W_sr, [-1,1])*K
            B = tf.eye(W.shape[0], dtype='float64') + W_sr_K * W_sr
            L = tf.linalg.cholesky(B)
            b = W * f + (self.y - pi)
            a = b - W_sr * tf.reshape(tf.linalg.cholesky_solve(L, tf.matmul(W_sr_K, tf.reshape(b, (-1,1)))), [-1])
            f = tf.matmul(K, tf.reshape(a, [-1,1]))

            lml = -0.5*tf.matmul(tf.reshape(a, [1,-1]), f)-tf.reduce_sum(tf.math.log(1+tf.math.exp(-(self.y*2.0-1.0)*tf.reshape(f, [-1]))))- tf.reduce_sum(tf.math.log(tf.linalg.tensor_diag_part(L)))
            f = np.reshape(f, [-1])

            if lml[0,0]-log_marginal_likelihood < 1e-10:
                break
            log_marginal_likelihood = lml

        self.f_cached = f
        if return_temporaries:
            return f, lml, (pi, W_sr, L, b, a)
        else:
            return f, lml, i