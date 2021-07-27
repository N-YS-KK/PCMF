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