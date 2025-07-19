import numpy as np
import random

class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _prev=(), _op='', label=''):
        self.data = data
        self.grad = 0
        self.label = label
        self._backward = lambda: None
        self._prev = set(_prev)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        t = (np.exp(2*x) - 1)/(np.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value({self.label}={self.data}, grad={self.grad})"


class ValueList:
    def __init__(self, values):
        self.values = values

    def backward(self):
        for v in self.values:
            v.backward()

    def __getitem__(self, i):
        return self.values[i]

    def __iter__(self):
        return iter(self.values)

    def __repr__(self):
        return f"ValueList({self.values})"
    
class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1), label='w') for _ in range(nin)]
        self.b = Value(random.uniform(-1,1), label='b')

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)))
        act += self.b
        act.label = 'act'
        out = act.tanh()
        out.label = 'out'
        return out

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"Neuron({len(self.w)})"

class Layer(Module):
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else ValueList(outs)

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    def __init__(self, nin, nouts):
        szs = [nin] + nouts
        self.nin = nin
        self.layers = [Layer(szs[i], szs[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of {self.nin} Inputs and {len(self.layers)} Layers: [{', '.join(str(layer) for layer in self.layers)}]"

    def loss(self, xs, ys):
        ypred = [self(x) for x in xs]
        loss = sum((Value(ygt)-ypred)**2 for ygt, ypred in zip(ys, ypred))
        return loss

    def grad_desc(self, xs, ys, step_size=0.01, iter=1):
        for k in range(iter):
            loss = self.loss(xs, ys)
            self.zero_grad() #reset grad to 0 for all parameters
            loss.backward() #calculate new grads

            for p in self.parameters():
                p.data += (-1.0 * step_size * p.grad)
            loss = self.loss(xs, ys)
            print("Loss: ", loss.data)