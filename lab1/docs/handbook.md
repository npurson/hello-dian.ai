# Handbook

## 1. Helloworld

### Learning Outline

* Linear model
* Optimization & gradient descent

Refer to ***Reference part of [README](../README.md)*** for learning materials.

All the parts that need to be implemented are marked as follows, 
look for all TODO blocks by search "TODO".

```Python
# TODO xxxx

...

# End of todo
```

### Objectives

Implment the `Linear` Class in `nn/modules.py` with `forward()` and `backward()` methods. 
Then pass the test of `module_test.py`, the usage is as follows:

```bash
$ python module_test.py Linear Sigmoid  # Test the specified module
Linear: forward ................ pass
        backward ............... pass
Sigmoid: forward ............... pass
         backward .............. pass
$ python module_test.py  # Test all the modules
Linear: forward ................ pass
        backward ............... pass
...
```

Run `helloworld.py` to perform a simple classification task on binary-class coordinates. 
Plot the decision boundary of the classifier by `plot_clf.py`. Use `plt.savefig()` if your environment does not support showing images.

Tips: [Optional] VSCode user can use `# %%` to turn a Python source file into Jupyter Notebook.

## 2. MNIST classification

### Learning Outline

* Backward propagation
* NN modules: BatchNorm, Convolution, Pooling, Activation, etc.
* Loss function

### Objectives:

1. Implement the rest TODOs in `modules.py` & `functional.py` and pass the test with `module_test.py`.
2. Implement the TODOs in `optim.py`
3. Design a classifier and train on MNIST or other harder datasets, *e.g.*, Cifar-10.
4. Once you have completed the whole NN framework, its source code and our official solutions are worth reading!

* 😵 We still haven't figure how to test the implementation of `BatchNorm1d`, submit PRs if you have any better ideas!
