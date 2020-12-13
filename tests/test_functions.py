import pytest

from vanilla_nn.functions import Sum
from vanilla_nn.functions import Prod
from vanilla_nn.functions import Const
from vanilla_nn.functions import Var


@pytest.mark.parametrize(
    "expression, wrt, at, expected_val",
    [
        (
            # 5x + x^2
            Sum(Prod(Var("x"), Const(5)), Prod(Var("x"), Var("x"))),
            "x",
            {"x": 5},
            15,
        ),
        (
            # 5x + xy
            Sum(Prod(Var("x"), Const(5)), Prod(Var("x"), Var("y"))),
            "y",
            {"x": 7, "y": 10},
            7,
        ),
        (
            # (2+x)y
            Prod(Sum(Var("x"), Const(2)), Var("y")),
            "y",
            {"x": 9, "y": 3},
            11,
        ),
        (
            # x^2y
            Prod(Prod(Var("x"), Var("x")), Var("y")),
            "x",
            {"x": 7, "y": 4},
            56,
        ),
    ],
)
def test_differentiation(expression, wrt, at, expected_val):
    assert expression.diff(wrt, at) == expected_val