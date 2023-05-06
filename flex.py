import random
from typing import Union, Tuple, Any

class A:

    def __init__(self) -> None:
        self.value = random.randint(1, 10)

    def __repr__(self) -> str:
        return str(self.value)
    
    def __add__(self, other: Any) -> 'A':
        if not isinstance(other, A):
            raise ArithmeticError("bad right operand")

        result = A()
        result.value = self.value + other.value
        return result


a = A()
b = A()

print(f'{a=}, {b=}')

print(f'sum={a + b}')