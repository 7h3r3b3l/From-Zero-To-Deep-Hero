
## Generalization
- The goal of ML and DL is to understand patterns in the data, not learn the data, otherwise, the model will be **underfitted** 

## Training Error and generalization

- When doing machine learning, one of the problems that may be is the assumption that the data have identical distributions, what if we assume that the distribution is $P(X,Y)$ and is really $Q(X,Y)$ ? 

- There is a **Training error** $R_{emp}$ 
- There is a generalization error $R$ that respect the distribution

- $R_{emp}[X,y,f] = \frac{1}{n} \sum_{i=1} ^n l(x^i, y^i , f(x^i))$ 
- $R(p,f) = E_{(x,y)} ~ P(l(x,y,f(x)))$ 


- we can never calculate $R$ exactly
- we must estimate $R$ testing $X' , y'$ 

## Model complexity

- when more data, statistically, we can approximate better to a good solution

#### Cross validation

- is an algorithm when data is split into $k$ non overlapping subsets, then the model training and validation is executed $k$ times each training on $k-1$ subsets and validating different subsets