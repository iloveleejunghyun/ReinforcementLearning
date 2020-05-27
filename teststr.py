
class A:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        super().__init__()

    def __str__(self):
        return f"{self.x}, {self.y}"


a = A(1, 2)
print(a)
