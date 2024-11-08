
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
        self.input = (x, y)
    
    def backward(self, gradient):
        x = self.input[0]
        y = self.input[1]
        return (y * gradient, x * gradient)
