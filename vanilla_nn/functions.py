# Dad: think about defining Name object and likewise +,-,/,x
import logging
from math import log

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.FileHandler("debug.log"))


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
        # This needs to return a function
        # How do we make it build its own function?!!?
        raise NotImplementedError


class Sum(Function):
    def __init__(self, a, b):
        def fn(x, y):
            return x + y

        self.func = fn
        self.args = (a, b)

    def __str__(self):
        return f"Sum{{ {self.args[0]}, {self.args[1]} }}"

    def eval(self, at):
        logger.debug(f"Evaluating {self.args[0]}+{self.args[1]}")
        return self.args[0].eval(at) + self.args[1].eval(at)

    def diff(self, with_respect_to, at):
        logger.debug(
            f"d/d{with_respect_to}({self.args[0]}) + d/d{with_respect_to}({self.args[1]})"
        )
        return self.args[0].diff(with_respect_to, at) + self.args[1].diff(
            with_respect_to, at
        )


class Prod(Function):
    def __init__(self, a, b):
        # Never actually use the self.func property
        self.func = lambda x, y: x * y
        self.args = (a, b)

    def __str__(self):
        return f"Prod{{ {self.args[0]},{self.args[1]} }}"

    def eval(self, at):
        logger.debug(f"Evaluating {self.args[0]}*{self.args[1]}")
        return self.args[0].eval(at) * self.args[1].eval(at)

    def diff(self, with_respect_to, at):
        logger.debug(
            f"d/d{with_respect_to}({self.args[0]})*{self.args[1]} + d/d{with_respect_to}({self.args[1]})*{self.args[0]}"
        )
        return self.args[0].diff(with_respect_to, at) * self.args[1].eval(
            at
        ) + self.args[1].diff(with_respect_to, at) * self.args[0].eval(at)


class Neg(Function):
    def __init__(self, a):
        self.func = lambda x: -x
        self.args = (a,)

    def __str__(self):
        return f"-{self.args[0]}"

    def eval(self, at):
        return -1 * self.args[0].eval(at)

    def diff(self, with_respect_to, at):
        return -1 * self.args[0].diff(with_respect_to, at)


class Exp(Function):
    def __init__(self, y, x):
        self.func = lambda y, x: y ** x
        self.args = (y, x)

    def __str__(self):
        return f"Exp{{ {self.args[0]},{self.args[1]} }}"

    def eval(self, at):
        logger.debug(f"Evaluating {self.args[0]}^{self.args[1]}")
        return self.args[0].eval(at) ** self.args[1].eval(at)

    def diff(self, with_respect_to, at):
        wrt_s = f"d/d{with_respect_to}"
        logger.debug(
            f""
        )
        return self.eval(at)*(
            log(self.args[0].eval(at))*self.args[1].diff(with_respect_to, at)
            + 
            self.args[1].eval(at)/self.args[0].eval(at)*self.args[1].diff(with_respect_to,at)
            )


class Const:
    def __init__(self, a):
        # Only allowed numbers
        self.constant = a

    def __str__(self):
        return f"Const({self.constant})"

    def eval(self, at):
        logger.debug(f"Constant value is {self.constant} at {at}")
        return self.constant

    def diff(self, with_respect_to, at):
        logger.debug(f"Derivative of a constant ({self.constant}) is 0")
        return 0


class Var:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"Var({self.name})"

    def eval(self, at):
        logger.debug(f"Value of variable {self.name} is {at[self.name]}")
        return at[self.name]

    def diff(self, with_respect_to, at):
        deriv = 1 if with_respect_to == self.name else 0
        logger.debug(f"Derivative of {self.name} w.r.t. {with_respect_to} is {deriv}")
        return deriv

class ReLU(Function):
    def __init__(self, x):
        def func(x):
            if x > 0:
                return x
            else:
                return Const(0)
        self.func = func
        self.args = (x,)
    
    def __str__(self) -> str:
        return f"Relu({self.args[0]})"

    def eval(self, at):
        if self.args[0].eval(at) > 0:
            return self.args[0].eval(at)
        else:
            return Const(0).eval(at)

    def diff(self, wrt, at):
        return 1 if self.eval(at) > 0 else 0



MSE = lambda prediction, label: Prod(
    Sum(Const(label), Neg(prediction)), Sum(Const(label), Neg(prediction))
)
