"""
LightNet is a very subtle attempt at implementing TensorFlow/PyTorch like neural networks which use Rev_autodiff
and computational graphs. LightNet is only for educational purposes and can be used to implement and train nets
over small and moderately large datasets (maxsize after tiling can be of the order 10^6) in a reasonable time frame.

To dive deep into how LightNet was built refer the Jupyter Notebook 'Reverse_autodiff_py.ipynb'.
"""

import numpy as np
import graphviz

class Node:
    def __init__(self, data, label='', children=(), _op=[]):
        self.data=np.array(data, dtype=float)
        self.label=label
        self.children=children
        self._op=_op
        self._grad=np.zeros(shape=self.data.shape, dtype=float)
        self._backward= lambda: None
    
    def __repr__(self):
        return f"Node: data={np.array2string(self.data, precision=4, floatmode='fixed')}, label='{self.label}', op='{self._op}'"
    
    def __add__(self, other):
        new_node=Node(self.data+other.data, children=(self, other), _op=["+"])
        
        def _backward():
            new_node.children[0]._grad+=new_node._grad
            new_node.children[1]._grad+=new_node._grad
            
            for child in new_node.children:
                if(child._op):
                    child._backward()
                    
        new_node._backward=_backward
        return new_node
    
    def __mul__(self, other):
        new_node=Node(self.data*other.data, children=(self, other), _op=['*'])
        
        def _backward():
            new_node.children[0]._grad+=(new_node._grad)*(new_node.children[1].data)
            new_node.children[1]._grad+=(new_node._grad)*(new_node.children[0].data)
            
            for child in new_node.children:
                if(child._op):
                    child._backward()
                    
        new_node._backward=_backward
        return new_node
    
    def dotpr(self, other):
        new_node=Node(np.sum((self.data)*(other.data)), children=(self, other), _op=['dot'])
        
        def _backward():
            new_node.children[0]._grad+=(new_node._grad)*(new_node.children[1].data)
            new_node.children[1]._grad+=(new_node._grad)*(new_node.children[0].data)
            new_node.children[0]._backward()
            new_node.children[1]._backward()
            
        new_node._backward=_backward
        return new_node
    
    def sig(self):
        new_node=Node(1/(1+pow(np.e, -self.data)), children=(self, ),_op=['sig'])
        
        def _backward():
            new_node.children[0]._grad+=(new_node.data)*(1-new_node.data)*new_node._grad
            
            if(new_node.children[0]._op):
                new_node.children[0]._backward()
                    
        new_node._backward=_backward
        return new_node
    
    def BinCrossEntropy(self, y):
        new_node=Node((-y*np.log(self.data)-(1-y)*np.log(1-self.data)),children=(self,), _op=["BCE"])

        def _backward():
            new_node.children[0]._grad+=(((1-y)/(1-new_node.children[0].data))-(y/new_node.children[0].data))*new_node._grad

            if(new_node.children[0]._op):
                new_node.children[0]._backward()

        new_node._backward=_backward
        return new_node
        
    @staticmethod
    def buffer(Nodes:list):
        new_node=Node([node.data for node in Nodes], children=tuple(Nodes), _op=['buf'])
        
        def _backward():
            i=0
            for child in new_node.children:
                child._grad+=new_node._grad[i]
                i+=1
            
            for child in new_node.children:
                if(child._op):
                    child._backward()
                    
        new_node._backward=_backward
        return new_node

    @staticmethod
    def backward(root:'Node'):
        root._grad=np.ones(shape= root.data.shape, dtype=float)
        root._backward()
        
def draw_graph(root: Node)-> graphviz.graphs.Digraph:
    graph=graphviz.Digraph(format='svg', name="Comp_graph", graph_attr={"rankdir":"LR"}, comment="Computational graph")
    
    uidr=str(id(root))
    graph.node(uidr, label=f"{root.label} | data: {np.array2string(root.data, precision=4, floatmode='fixed')} | grad: {np.array2string(root._grad, precision=4, floatmode='fixed')}", shape="record")
    
    if root._op:
        uidr_op=str(id(root._op))
        graph.node(uidr_op, label=f"{root._op}")
        graph.edge(uidr_op, uidr)
    else:
        return graph
    
    check=set((uidr))
    edges={(uidr, uidr_op)}
    
    def rec(curr: Node, parent: Node):
        uid=str(id(curr))
        if uid not in check:
            graph.node(uid, label=f"{curr.label} | data: {np.array2string(curr.data, precision=4, floatmode='fixed')} | grad: {np.array2string(curr._grad, precision=4, floatmode='fixed')}", shape="record")
            check.add(uid)
        if tuple((uid, str(id(parent._op)))) in edges:
            return
        graph.edge(uid, str(id(parent._op)))
        edges.add((uid, str(id(parent))))
            
        if curr._op:
            uid_op=str(id(curr._op))
            graph.node(uid_op, label=f"{curr._op}")
            if tuple((uid_op, uid)) in edges:
                return
            graph.edge(uid_op, uid)
            edges.add((uid_op,uid))
        else:
            return
        
        for child1 in curr.children:
            rec(child1, curr)
        
    for child in root.children:
        rec(child, root)
        
    return graph

class Neuron:
    def __init__(self, num_par, activation, label=''):
        self.label=label
        self.activation = activation
        self.num_par=num_par
        self.weights=Node(np.random.randn(num_par+1), label=f"W: {self.label}")
        
    def __call__(self, input:Node):
        wdoti=self.weights.dotpr(input); wdoti.label=f"D: {self.label}"
        sigz=wdoti.sig(); sigz.label=f"S: {self.label}"
        self.wdoti=wdoti
        self.sigz=sigz
        return self.sigz
        
class Layer:
    def __init__(self, units:int, activation='',num_inp=0, label=''):
        self.units=units
        self.activation=activation
        self.num_inp=num_inp
        self.label=label
        self.neuron_list=[]
        self.inputs=Node(np.zeros((self.num_inp+1)))
        if(num_inp!=0):     
            """
            IF A LAYER IS EXPLICITLY DECLARED i.e not through the network class.
            """
            self.neuron_list=[Neuron(self.num_inp,self.activation,label=f"{self.label}, N{i}") for i in range(units)]
    
    def create_neurons(self):
        """
        INCASE A LAYER IS IMPLICITLY DECLARED i.e. through the network class.
        """
        self.neuron_list=[Neuron(self.num_inp,self.activation,label=f"{self.label}, N{i}") for i in range(self.units)]
    
    def __call__(self, input:Node):
        self.inputs=input
        buffer_node=Node.buffer([neuron(self.inputs) for neuron in self.neuron_list]); buffer_node.label=f"I: L{int(self.label[1])+1}"
        return buffer_node
    
    def get_weights(self):
        weights=np.array([neuron.weights.data for neuron in self.neuron_list])
        return weights
               
class Network:
    def __init__(self, layers:list):
        self.layers=layers
        self.inp_par=layers[0][0]
        
        # 1) INITIALISING NUM_INP FOR ALL LAYERS OF THE NETWORK
        layers[1].num_inp=self.inp_par
        layers[1].create_neurons()
        for i in range(2,len(layers)):
            layers[i].num_inp=layers[i-1].units            
            # 2) CREATE NEURON OBJECTS FOR EACH LAYER
            layers[i].create_neurons()
            
        # 3) CONNECTING THE LAYERS: to connect the layers we'll have to fwd pass through the network once
        self.root=self(np.random.rand(self.inp_par))
            
        
    def __call__(self, input):
        input=np.array(input).reshape((self.inp_par))
        input=np.r_[input, 1]
        output=Node(input,label=f"I: L1")
        for layer in self.layers[1:]:
            output=layer(output)
            output.data=np.r_[output.data, 1]
            output._grad=np.r_[output._grad, 0]
        output.data=np.array([output.data[0]], dtype=float)
        output._grad=np.array([output._grad[0]], dtype=float)
        self.root=output
        return output
    
    def backward(self):
        Node.backward(self.root)
    
    def train(self, X, Y, epoch, alpha):
        #l=[]
        X=np.array(X, dtype=float)
        Y=np.array(Y, dtype=float)
        for _ in range(epoch):
            for i in range(X.shape[0]):
                x=X[i]
                y=Y[i]
                # 1) fwd pass and calculating loss for the datapoint assuming activation to be sigmoid
                y_p=self(x)
                self.root=y_p.BinCrossEntropy(y); self.root.label='Loss'
                # 2) backward pass
                self.backward()
                
                # 3) Weight update step: data=data-(alpha*_grad) for every neuron
                for layer in self.layers[1:]:
                    for neuron in layer.neuron_list:
                        neuron.weights.data = neuron.weights.data-(alpha*neuron.weights._grad)                        
                        neuron.weights._grad=np.zeros(neuron.weights.data.shape)
                        neuron.wdoti._grad=np.zeros(neuron.wdoti.data.shape)
                        neuron.sigz._grad=np.zeros(neuron.sigz.data.shape)
                    layer.inputs._grad=np.zeros(layer.inputs._grad.shape)
                self.root.children[0]._grad=np.array([0], dtype=float)
                self.root._grad=np.array([0], dtype=float)
                
                # l.append(self.draw())
                # 'l' contains graphviz.Digraph objects, each is a graph made after one weight update step
        self.root=self.root.children[0]
        #return l
    
    def score(self, X,Y):
        i=0
        count=0
        for x in X:
            c=float(self(x).data[0])>.5
            if Y[i]==c:
                count+=1
            i+=1
        return count/Y.shape[0]             
    
    def get_weights(self):
        for layer in self.layers[1:]:
            print(f"{layer.label}\nWeights: \n{layer.get_weights()}")
        
    def draw(self):
        return draw_graph(self.root)