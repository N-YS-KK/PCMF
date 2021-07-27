![](https://raw.githubusercontent.com/N-YS-KK/PCMF/main/images/PCMF_logo.PNG) 

# Positive Collective Matrix Factorization (PCMF)
We propose Positive Collective Matrix Factorization (PCMF). PCMF is a model that combines the interpretability of NMF and the extensibility of CMF.

# Description of PCMF
Non-Negative Matrix Factorization (NMF) and Collective matrix Factorization (CMF) exist as methods of matrix factorization. The features of each are as follows.

## Non-Negative Matrix Factorization（NMF）
Predict the original matrix by the product of two nonnegative matrices.

![](https://raw.githubusercontent.com/N-YS-KK/PCMF/main/images/NMF.PNG) 

- Advantages  

Since it is non-negative, a highly interpretable feature representation can be obtained.

- Disadvantages  

Low extensibility because multiple relationships cannot be considered.

## Collective matrix Factorization（CMF）
This is a method of factoring two or more relational data (matrix) at the same time when a set has multiple relations.

![](https://raw.githubusercontent.com/N-YS-KK/PCMF/main/images/CMF.PNG) 

- Advantages  

In addition to being able to consider multiple relationships, flexible output is possible (link function), so it is highly extensible.

- Disadvantages  

The interpretability is low because positive and negative values appear in the elements of the matrix.

## Positive Collective Matrix Factorization (PCMF)
PCMF is a model that combines the advantages of NMF, "interpretability," and the advantages of CMF, "extensibility." Specifically, for each matrix, interpretability is achieved by converting the elements of the matrix into positive values using a softplus function. The backpropagation method is used as the learning method.

![](https://raw.githubusercontent.com/N-YS-KK/PCMF/main/images/PCMF.PNG) 

# Installation
coming soon!

# Usage
coming soon!

# Training
coming soon!

# License
MIT Licence

# Joint research
coming soon!

# Citation
coming soon!

# Reference
[1] Daniel D. Lee and H. Sebastian Seung. “Learning the parts of objects by non-negative matrix factorization.” Nature 401.6755 (1999): 788-791.

[2] Daniel D. Lee and H. Sebastian Seung. “Algorithms for non-negative matrix factorization.” Advances in neural information processing systems 13 (2001): 556-562.

[3] Ajit P. Singh and Geoffrey J. Gordon. Relational learning via collective matrix factorization. Proceedings of the 14th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining: 650-658, 2008.

[4] David E. Rumelhart, Geoffrey E. Hinton and Ronald J. Williams. “Learning representations by back-propagating errors.” Nature 323.6088 (1986): 533-536

[5] Diederik P. Kingma and Jimmy Ba. “Adam: A method for stochastic optimization.” arXiv preprint arXiv:1412.6980 (2014).

[6] Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfel-low, Andrew Harp, Geoffrey Irving, Michael Isard, Yangqing Jia, Rafal Jozefowicz, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mane, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Mike Schuster, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viegas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu and Xiaoqiang Zheng. “Tensor-flow: Large-scale machine learning on heterogeneous distributed systems.” arXiv preprint arXiv:1603.04467 (2016)
