class Function:
    # func
    # args
    def __init__(self):
        raise NotImplementedError

    def eval(self, at):
        return self.func(*self.args)

    def diff(self, with_respect_to, at):
        # At is a dictionary containing values of all variables
        # e.g. {'x':2,'y':10}
        raise NotImplementedError


class Sum(Function):
    def __init__(self, a, b):
        self.func = lambda x, y: x + y
        self.args = (a, b)

    def diff(self, with_respect_to, at):
        print(
            f"d/d{with_respect_to}({self.args[0]}) + d/d{with_respect_to}({self.args[1]}) at {at}"
        )
        return self.args[0].diff(with_respect_to, at) + self.args[1].diff(
            with_respect_to, at
        )


class Prod(Function):
    def __init__(self, a, b):
        self.func = lambda x, y: x * y
        self.args = (a, b)

    def diff(self, with_respect_to, at):
        print(
            f"d/d{with_respect_to}({self.args[0]})*{self.args[1]} + d/d{with_respect_to}({self.args[1]})*{self.args[0]}, at {at}"
        )
        return self.args[0].diff(with_respect_to, at) * self.args[1].eval(
            at
        ) + self.args[1].diff(with_respect_to, at) * self.args[0].eval(at)


class Const(Function):
    def __init__(self, a):
        # Only allowed numbers
        self.func = lambda x: x
        self.args = (a,)

    def diff(self, with_respect_to, at):
        print(f"Derivative of a constant ({self.args[0]}) is 0")
        return 0


class Var:
    def __init__(self, name):
        self.name = name

    def eval(self, at):
        print(f"Value of variable {self.name} is {at[self.name]}")
        return at[self.name]

    def diff(self, with_respect_to, at):
        deriv = 1 if with_respect_to == self.name else 0
        print(f"Derivative of {self.name} w.r.t. {with_respect_to} is {deriv}")
        return deriv


ex = Sum(Prod(Var("x"), Const(5)), Prod(Var("x"), Var("x")))
# We want:
# ex -> 5x + x^2
# ex.diff('x',2) -> 5 + 2x -> 5 + 2*2 = 9
print(ex.diff("x", {"x": 2}))

ex2 = Sum(Prod(Var("x"), Const(5)), Prod(Var("x"), Var("y")))
# We want:
# ex2 -> 5x + xy
# ex2.diff("y", {"x": 7, "y": 10}) -> x -> 7
print(ex2.diff("y", {"x": 7, "y": 10}))
