

## Binary classification

- the goal of machine learning is to label new data


#### The perceptron

- the perceptron is a model based on neurons from the brain
- recibes inputs, that goes to a black box, that returns an output

- there are $x_1,x_2,x_3 , ... , x_n$ that corresponds to the **inputs**
- there are also $w_1,w_2, ... , w_n$ that correspond to the **weights**
- there is also a unit $b$ that is known as the **bias**
- $z = b +  \sum_{i=1}^n x_i w_i$  

#### The training process

- we have $z = b+ \sum_{i=1}^n x_i w_i$
- the idea is to find all the $w_i$ and $b$ 
1. define the training set
2. initialize model weights and bias to 0
3. for every training epoch
	- for every training example $x^{[i]}, y^{[i]}$ 
			1. make a prediction
			2. compute the error
			3. update weights


