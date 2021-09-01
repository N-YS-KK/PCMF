![](https://raw.githubusercontent.com/N-YS-KK/PCMF/main/images/PCMF_logo.PNG) 

# Positive Collective Matrix Factorization (PCMF)
We propose Positive Collective Matrix Factorization (PCMF). PCMF is a model that combines the interpretability of NMF and the extensibility of CMF.

# Description of PCMF

## Problem setting
When there are two relational data (matrix ![](https://latex.codecogs.com/gif.latex?X), ![](https://latex.codecogs.com/gif.latex?Y)) that share one set, and you want to predict the relational data (matrix ![](https://latex.codecogs.com/gif.latex?\hat{X}), ![](https://latex.codecogs.com/gif.latex?\hat{Y})) and extract feature representations (matrix ![](https://latex.codecogs.com/gif.latex?U), ![](https://latex.codecogs.com/gif.latex?V^T), ![](https://latex.codecogs.com/gif.latex?Z^T)) at the same time.

### Example
- Two relational data (matrix ![](https://latex.codecogs.com/gif.latex?X), ![](https://latex.codecogs.com/gif.latex?Y))

![](https://latex.codecogs.com/gif.latex?X): Patient-disease matrix  
![](https://latex.codecogs.com/gif.latex?Y): Patient-patient attribute matrix  

At this time, the patient set is shared.

![](https://raw.githubusercontent.com/N-YS-KK/PCMF/main/images/two_relational_data.PNG) 

- Feature representations

![](https://latex.codecogs.com/gif.latex?U): Patient matrix  
![](https://latex.codecogs.com/gif.latex?V^T): Disease matrix  
![](https://latex.codecogs.com/gif.latex?Z^T): Patient attributes matrix

![](https://raw.githubusercontent.com/N-YS-KK/PCMF/main/images/feature_representations.PNG) 

## Detailed description of PCMF
PCMF is a model that combines the advantages of NMF, "interpretability," and the advantages of CMF, "extensibility." Specifically, for each matrix, interpretability is achieved by converting the elements of the matrix into positive values using a softplus function. The backpropagation method is used as the learning method.

The illustration of PCMF is as follows.

![](https://raw.githubusercontent.com/N-YS-KK/PCMF/main/images/PCMF.PNG) 

### Example  
This will be described using the previous example.

- The patient matrix ![](https://latex.codecogs.com/gif.latex?U^T) with the softplus function applied is the patient matrix ![](https://latex.codecogs.com/gif.latex?U^{'T}).  
- The disease matrix ![](https://latex.codecogs.com/gif.latex?V^T) with the softplus function applied is the disease matrix ![](https://latex.codecogs.com/gif.latex?V^{'T}).  
- The patient attribute matrix ![](https://latex.codecogs.com/gif.latex?Z^T) with the softplus function applied is the patient attribute matrix ![](https://latex.codecogs.com/gif.latex?Z^{'T}).  
- Applying the link function ![](https://latex.codecogs.com/gif.latex?f_1) to the product of the patient matrix ![](https://latex.codecogs.com/gif.latex?U^{'}) and the disease matrix ![](https://latex.codecogs.com/gif.latex?V^{'T}) yields the predicted value of the patient-disease matrix ![](https://latex.codecogs.com/gif.latex?X^{'}).  
- Applying the link function ![](https://latex.codecogs.com/gif.latex?f_2) to the product of the patient matrix ![](https://latex.codecogs.com/gif.latex?U^{'}) and the patient attributes ![](https://latex.codecogs.com/gif.latex?Z^{'T}) yields the predicted value of the patient-patient attributes matrix ![](https://latex.codecogs.com/gif.latex?Y^{'}).

### Softplus function
![](https://latex.codecogs.com/gif.latex?\zeta(x)=\mathrm{log}(1+e^x))

The softplus function is a narrowly monotonically increasing function that takes a positive value for all real numbers ![](https://latex.codecogs.com/gif.latex?x\in\mathrm{R}). It is applied to each element of the matrix, and it is assumed that a matrix of the same size is output.

### Link function
Note that due to the influence of the Softplus function, the input value of the PCMF link function is always positive. Choose a link function depending on the nature and purpose of the matrix you are predicting.

- When the value of the matrix to be predicted is (-∞, ∞)  
Log function.

- When the value of the matrix to be predicted is (0, ∞)  
Linear function.

- When the value of the matrix to be predicted is {0,1}  
Sigmoid function. (Since the output value of the sigmoid function is 0.5 or more when the input value is 0 or more, the operation of subtracting a common positive number uniformly for the input is performed.)

### Feature representations analysis
Feature representations analysis can be performed by analyzing the feature representations (matrix ![](https://latex.codecogs.com/gif.latex?U), ![](https://latex.codecogs.com/gif.latex?V^T), ![](https://latex.codecogs.com/gif.latex?Z^T)) extracted by PCMF. (Note that PCMF outputs the matrix ![](https://latex.codecogs.com/gif.latex?U^{'}), ![](https://latex.codecogs.com/gif.latex?V^{'T}), ![](https://latex.codecogs.com/gif.latex?Z^{'T})), which is the format to which the softplus function is applied, as the final output.)

## CMF and NMF (reference)
Non-Negative Matrix Factorization (NMF) and Collective matrix Factorization (CMF) exist as methods of matrix factorization. The features of each are as follows.

### Non-Negative Matrix Factorization（NMF）[1][2]
Predict the original matrix by the product of two nonnegative matrices.

- Advantages  
Since it is non-negative, a highly interpretable feature representation can be obtained.

- Disadvantages  
Low extensibility because multiple relationships cannot be considered.

### Collective matrix Factorization（CMF）[3]
This is a method of factoring two or more relational data (matrix) at the same time when a set has multiple relations.

- Advantages  
In addition to being able to consider multiple relationships, flexible output is possible (link function), so it is highly extensible.

- Disadvantages  
The interpretability is low because positive and negative values appear in the elements of the matrix.

# Installation
You can get PCMF from PyPI. Our project in PyPI is [here](https://pypi.org/project/pcmf/). 
```bash
pip install pcmf
```

# Usage
For more detail, please read `examples/How_to_use_PCMF.ipynb`. If it doesn't render at all in github, please click [here](https://nbviewer.jupyter.org/github/N-YS-KK/PCMF/blob/main/examples/How_to_use_PCMF.ipynb).

## Training

```python
cmf = Positive_Collective_Matrix_Factorization(X, Y, alpha=0.5, d_hidden=12, lamda=0.1)
cmf.train(link_X = 'sigmoid', link_Y = 'sigmoid', 
          weight_X = None, weight_Y =wY, 
          optim_steps=501, verbose=50, lr=0.05)
```

# License
MIT Licence

# Citation
You may use our package(PCMF) under MIT License. If you use this program in your research then please cite:

**PCMF Package**
```bash
@misc{sumiya2021pcmf,
  author = {Yuki, Sumiya and Ryo, Matsui and Kensho, Kondo and Kazuhide, Nakata},
  title = {PCMF},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {https://github.com/N-YS-KK/PCMF}
}
```

**PCMF Paper**[ [link](https://www.jstage.jst.go.jp/article/pjsai/JSAI2021/0/JSAI2021_2G3GS2e03/_pdf/-char/ja) ](Japanese)
```bash
@article{sumiya2021pcmf,
  title={Patient Disease Prediction and Medical Feature Extraction using Matrix Factorization},
  author={Yuki, Sumiya and Atsuyoshi, Matsuda and Kenji, Araki and Kazuhide, Nakata},
  journal={The Japanese Society for Artifical Intelligence},
  year={2021}
}
```

# Reference
[5] [6] [7] are used in the code.  

[1] Daniel D. Lee and H. Sebastian Seung. “Learning the parts of objects by non-negative matrix factorization.” Nature 401.6755 (1999): 788-791.

[2] Daniel D. Lee and H. Sebastian Seung. “Algorithms for non-negative matrix factorization.” Advances in neural information processing systems 13 (2001): 556-562.

[3] Ajit P. Singh and Geoffrey J. Gordon. Relational learning via collective matrix factorization. Proceedings of the 14th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining: 650-658, 2008.

[4] Yuki Sumiya, Kazuhide Nakata, Atsuyoshi Matsuda, Kenji Araki. "Patient Disease Prediction and Relational Data Mining using Matrix Factorization." The 40th Joint Conference on Medical Informatics, 2020.

[5] David E. Rumelhart, Geoffrey E. Hinton and Ronald J. Williams. “Learning representations by back-propagating errors.” Nature 323.6088 (1986): 533-536

[6] Diederik P. Kingma and Jimmy Ba. “Adam: A method for stochastic optimization.” arXiv preprint arXiv:1412.6980 (2014).

[7] Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfel-low, Andrew Harp, Geoffrey Irving, Michael Isard, Yangqing Jia, Rafal Jozefowicz, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mane, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Mike Schuster, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viegas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu and Xiaoqiang Zheng. “Tensor-flow: Large-scale machine learning on heterogeneous distributed systems.” arXiv preprint arXiv:1603.04467 (2016)