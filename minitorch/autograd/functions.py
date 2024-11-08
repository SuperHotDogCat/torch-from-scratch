
class AddBackward:
    def __init__(self, x, y):
        self.input = (x, y)
    
    def backward(self, gradient):
        return (gradient, gradient)

class SubBackward:
    def __init__(self, x, y):
        self.input = (x, y)
    
    def backward(self, gradient):
        return (gradient, -gradient)

class ElementwiseMulBackward:
    def __init__(self, x, y):
