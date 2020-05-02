
# geneticalgorithm

geneticalgorithm is a Python library of standard and elitist genetic algorithm (GA) in Python.
geneticalgorithm solves optimization problems with continuous, integers, or mixed variables in python. 
geneticalgorithm provides an easy implementation of Elitist Genetic Algorithm (GA) in Python.   
    

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install geneticalgorithm in Python.

```bash
pip install geneticalgorithm
```


#A simpel example
Assume we want to find a set of X=(x1,x2,x3) that minimizes function f(X)=x1+x2+x3 where X can be any real number between zero and ten (\[0,10\]).
This is a trivial problem and we easily know that the answer is X=(0,0,0) where f(X)=0. 
We just use this simple example to see how to define our function, how to link it to geneticalgorithm, how to implement GA, and finally see how the output looks like.
Let's start with defining our function f that we want to minimize; we also import geneticalgorithm and [numpy](https://numpy.org) as below:

```python
import numpy as np
from geneticalgorithm import geneticalgorithm as GA

def f(X):
    return np.sum(X)
```
Notice that we define the function so that the output of the function is the objective function that we want to minimize where the input is the set of X, decision variables. 
Now that we have our simple function it is time to call geneticalgorithm and setup its parameters but before that lets define the boundary of the decision variables. So we continue the code as below:
```python
import numpy as np
from geneticalgorithm import geneticalgorithm as GA

def f(X):
    return np.sum(X)
    
    
varbound=np.array([[0,10]]*3)

```
Notice that we need to define an numpy array and for each variable we need a separate boundary. 
Here I have three variables and all of them have the same boundary so I multiply the boundary by 3 and it gives me an array of length three with the same entries. For the case the boundaries are different see other examples.
Also, note that varbound is NOT defined inside the function f.
Now it is time to call geneticalgorithm to solve the defined optimization problem:

```python
import numpy as np
from geneticalgorithm import geneticalgorithm as GA

def f(X):
    return np.sum(X)
    
    
varbound=np.array([[0,10]]*3)

model=GA(function=f,dimension=3,variable_type='real',variable_boundaries=varbound)

```
GA needs some input: 
Obviously the first input is the function f we already defined. 
Our problem has three variables so we set dimension equal three. 
Variables are real (continuous) so we use string real to notify the type of variables (geneticalgorithm accepts other types including Boolean, Integers and Mixed; see other examples).
Finally we input varboound which includes the boundaries of the variables. Note that the length of varbound must match dimension)
Now we are ready to run the GA. So we add another line to our Python code as below:

```python
import numpy as np
from geneticalgorithm import geneticalgorithm as GA

def f(X):
    return np.sum(X)
    
    
varbound=np.array([[0,10]]*3)

model=GA(function=f,dimension=3,variable_type='real',variable_boundaries=varbound)

model.run()
```
Then we run the code and you should see a progress bar that shows progress of the GA and then the solution, objctive function value and a graph as below:


![](https://github.com/rmsolgi/geneticalgorithm/blob/master/gaimpl.gif)




# Methods and Outputs:
        
methods:
    run(): implements the genetic algorithm (GA)
                
outputs:
    output_dict:  a dictionary including the best set of variables
    found and the value of the given function associated to it.
    {'variable': , 'function': }
            
    report: a list including the history of the progress of the
    algorithm over iterations


## Usage

```python
import foobar

foobar.pluralize('word') # returns 'words'
foobar.pluralize('goose') # returns 'geese'
foobar.singularize('phenomena') # returns 'phenomenon'
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
