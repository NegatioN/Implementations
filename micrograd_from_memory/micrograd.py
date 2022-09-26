
def cast(other):
    if not isinstance(other, Value):
        other = Value(other)
    return other

class Value:
    def __init__(self, data, children=()) -> None:
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self.prev = set(children)
    
    def __repr__(self) -> str:
        return f'Value({self.data})' 
    
    def __mul__(self, other):
        out = Value(self.data * cast(other).data, children=(self, other))
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = _backward
        return out
    
    def __add__(self, other):
        out = Value(self.data + cast(other).data, children=(self, other))
        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
        out._backward = _backward
        return out

    def backward(self):
        ''' Traverse the graph of nodes backwards.'''
        nodes = [self]
        seen = set()
        
        # is this function identical to topological ordering? You should make a test.
        for n in nodes:
            if n not in seen:
                n._backward()
                seen.add(n)
                nodes.extend(n.prev)
            
    def __sub__(self, other):
        return self + (other*-1)
    def __rsub__(self, other): return cast(other) - self
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other



a = Value(5)
b = Value(3)

print(a*b)
print(a+b)
print(a+2)
print(2+a)
print(2*a)
print(a*3)
print(a-3)
print(6-a)

a = Value(5)
b = Value(3)
c = Value(3)

d = a + b
e = c * d
e.grad = 1.0
e.backward()
print(a.grad, b.grad, c.grad, d.grad, e.grad)
