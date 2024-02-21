
#### One-Hot-Encoding 

One Hot Encoding is the way to represent each desition as a vector, for example, we have three fruits, such as {"banana", "orange" , "apple"}, then the vector that represents the desitions is $y = \{(1,0,0),(0,1,0)(0,0,1)\}$  where each touple represents each fruit

#### Linear Model

We will need a model with multiples outputs, where each output represent each touple of $y$, one per class. For example, in the previous vector, the figure would be 

![[Pasted image 20240221095829.png]]
 and the corresponding neural network diagram would be 
 ![nn_classification](https://d2l.ai/_images/softmaxreg.svg)

for a better notation, we use vectors and matrices $o = Wx+b$. for example, in the previous example, we have a $3x4$ $$ \begin{bmatrix}
w_{11} & w_{12} & w_{13} & w_{14} \\
w_{21} & w_{22} & w_{24} & w_{24} \\
w_{31} & w_{32} & w_{33} & w_{34} 
\end{bmatrix}  $$
and a vector of biases in $\mathbb{R}^3$ $b=\{b_1, b_2, b_3\}$  


#### The softmax

we can directly to minimize de difference between $o$  and he labels $y$ but
1. there is no guarantee that the outputs $o_1$ are nonnegative 
2. there is no guarantee that the outputs $o_1$ dont exceed 1

the idea of softmax comes from Boltzmann in statistical modern physics

#### Softmax and cross entropy loss

Since the softmax function and the corresponding cross entropy loss are so common, using the definition of the softmax we obtain: 
![[Pasted image 20240221123648.png]]

To understand better 