class Counter:
    def __init__(self,value):
        self.value=value
    def increment(self):
        self.value+=1
    def decrement(self):
        self.value-=1
    def reset(self):
        self.value=0
    def get_value(self):
        return self.value

x=Counter(5)
print(x.get_value())
x.increment()
print(x.get_value())