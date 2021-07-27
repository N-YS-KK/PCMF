import numpy as np
import tensorflow as tf

class Positive_Collective_Matrix_Factorization:

    """
    Our proposed model PCMF.

    Attributes
    ----------
    X : numpy.ndarray
    Y : numpy.ndarray

    alpha : int
        Y weight of loss function.

    d_hidden : int
        Number of potential topics.
        This is the number of columns in the matrix U.
        This is the number of rows in the matrix V.
        This is the number of rows in the matrix Z.

    lamda : int
        Regularization weight.

    link_X : str
        Link function of matrix X.
        This is supposed to take 'sigmoid', 'linear' and 'log'.

    link_Y : str
        Link function of matrix Y.
        This is supposed to take 'sigmoid', 'linear' and 'log'.

    weight_X : numpy.ndarray
        Weight of each element of X of loss function.

    weight_Y : numpy.ndarray
        Weight of each element of Y of loss function.

    optim_steps : int
        Number of repetitions of Adam.

    verbose : int
        How many steps to output the progress.

    lr : int
        Learning rate.

    #We'll skip the rest.
    """

    def __init__(self, X, Y, alpha=1, d_hidden=12, lamda=0.1):

        """
        Parameters
        ----------
        X : numpy.ndarray
        Y : numpy.ndarray

        alpha : int
            Y weight of loss function.

        d_hidden : int
            Number of potential topics.
            This is the number of columns in the matrix U.
            This is the number of rows in the matrix V.
            This is the number of rows in the matrix Z.

        lamda : int
            Regularization weight.
        """

        self.X = X
        self.Y = Y
        self.alpha = alpha
        self.d = d_hidden
        self.lamda = lamda

    def train(
        self,
        link_X="sigmoid",
        link_Y="sigmoid",
        weight_X=None,
        weight_Y=None,
        optim_steps=401,
        verbose=100,
        lr=0.005,
    ):

        """
        Trained matrix U, matrix V, and matrix Z.

        Parameters
        ----------
        link_X : str
            Link function of matrix X.
            This is supposed to take 'sigmoid', 'linear' and 'log'.

        link_Y : str
            Link function of matrix Y.
            This is supposed to take 'sigmoid', 'linear' and 'log'.

        weight_X : numpy.ndarray
            Weight of each element of X of loss function.

        weight_Y : numpy.ndarray
            Weight of each element of Y of loss function.

        optim_steps : int
            Number of repetitions of Adam.

        verbose : int
            How many steps to output the progress.

        lr : int
            Learning rate.

        Returns
        -------
        U_ : numpy.ndarray
            Trained matrix U.

        V_ : numpy.ndarray
            Trained matrix V.

        Z_ : numpy.ndarray
            Trained matrix Z.
        """

        # Initialization by normal distribution.
        U = tf.keras.backend.variable(
            tf.random.truncated_normal([self.X.shape[0], self.d], 0, 1.0),
            dtype=tf.float32,
        )
        V = tf.keras.backend.variable(
            tf.random.truncated_normal([self.d, self.X.shape[1]], 0, 1.0),
            dtype=tf.float32,
        )
        Z = tf.keras.backend.variable(
            tf.random.truncated_normal([self.d, self.Y.shape[1]], 0, 1.0),
            dtype=tf.float32,
        )

        # Definition of correct answer data.
        X = tf.keras.backend.constant(self.X, dtype=tf.float32)
        Y = tf.keras.backend.constant(self.Y, dtype=tf.float32)

        # Positive number subtracted from the argument when the link function = sigmoid.
        if link_X == "sigmoid":
            sig_biasX = tf.keras.backend.variable(12, dtype=tf.float32)
        if link_Y == "sigmoid":
            sig_biasY = tf.keras.backend.variable(12, dtype=tf.float32)
    
        # Weight of each element of X and Y of loss function. If not specified, all will be 1.0.
        if weight_X is None:  # Range is {0, 1} or [0, 1]
            weight_X = np.ones_like(X)
        if weight_Y is None:  # Range is {0, 1} or [0, 1]
            weight_Y = np.ones_like(Y)

        def loss(alpha=self.alpha, lamda=self.lamda):

            """
            Calculate the loss function.

            Parameters
            ----------
            alpha : int
                Y weight of loss function.

            lamda : int
                Regularization weight.

            Returns
            -------
            loss_all : tensorflow.python.framework.ops.EagerTensor
            """

            U_ = tf.nn.softplus(U)
            V_ = tf.nn.softplus(V)
            Z_ = tf.nn.softplus(Z)

            # For X, calculate loss according to the set link function.
            if link_X == "sigmoid":  # Range is {0, 1} or [0, 1]
                X_ = (
                    tf.matmul(U_, V_) - sig_biasX
                )  # Inner product> 0, subtract only sig_biasX.
                loss_X = tf.math.reduce_mean(
                    weight_X
                    * (X * tf.math.softplus(-X_) + (1 - X) * tf.math.softplus(X_))
                )  # sigmoid + cross_entropy
            elif link_X == "linear":  # Range is (0, ∞)
                X_ = tf.matmul(U_, V_)
                loss_X = tf.math.reduce_mean(weight_X * tf.square(X - X_))
            elif link_X == "log":  # Range is (-∞, ∞)
                X_ = tf.math.log(tf.matmul(U_, V_))
                loss_X = tf.math.reduce_mean(weight_X * tf.square(X - X_))
            
            # For Y, calculate loss according to the set link function.
            if link_Y == "sigmoid":  # Range is {0, 1} or [0, 1]
                Y_ = (
                    tf.matmul(U_, Z_) - sig_biasY
                )  # Inner product> 0, subtract only sig_biasX.
                loss_Y = tf.math.reduce_mean(
                    weight_Y
                    * (Y * tf.math.softplus(-Y_) + (1 - Y) * tf.math.softplus(Y_))
                )  # sigmoid + cross_entropy
            elif link_Y == "linear":  # Range is (0, ∞)
                Y_ = tf.matmul(U_, Z_)
                loss_Y = tf.math.reduce_mean(weight_Y * tf.square(Y - Y_))
            elif link_Y == "log":  # Range is (-∞, ∞)
                Y_ = tf.math.log(tf.matmul(U_, Z_))
                loss_Y = tf.math.reduce_mean(weight_Y * tf.square(Y - Y_))
            
            # Norm
            norm = (
                tf.math.reduce_euclidean_norm(U_)
                + tf.math.reduce_euclidean_norm(V_)
                + tf.math.reduce_euclidean_norm(Z_)
            )

            # Loss function
            loss_all = loss_X + alpha * loss_Y + lamda * norm
            return loss_all

        # Actual calculation from here.
        opt = tf.optimizers.Adam(learning_rate=lr)
        loss_record = []
        for times in range(optim_steps):
            loss_ = lambda: loss()
            loss_record.append(loss_().numpy())
            # Change the combination of variables to update depending on whether each link function is sigmoid.
            if link_X == "sigmoid":
                if link_Y == "sigmoid":
                    opt.minimize(loss_, var_list=[U, V, Z, sig_biasX, sig_biasY])
                else:
                    opt.minimize(loss_, var_list=[U, V, Z, sig_biasX])
            else:
                if link_Y == "sigmoid":
                    opt.minimize(loss_, var_list=[U, V, Z, sig_biasY])
                else:
                    opt.minimize(loss_, var_list=[U, V, Z])
            if verbose > 0:
                if times % verbose == 0:
                    print(
                        "[Info] At time-step {}, loss is {}".format(
                            times, loss_record[-1]
                        )
                    )
        # Apply softplus when outputting.
        U_ = tf.nn.softplus(U).numpy()
        V_ = tf.nn.softplus(V).numpy()
        Z_ = tf.nn.softplus(Z).numpy()

        return U_, V_, Z_