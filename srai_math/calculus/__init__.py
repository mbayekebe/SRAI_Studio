from .limits import (
    approach_sequence,
    epsilon_delta_check,
    is_continuous_at,
    numerical_limit,
    one_sided_limits,
    removable_extension,
)

__all__ = [
    "approach_sequence",
    "numerical_limit",
    "one_sided_limits",
    "is_continuous_at",
    "epsilon_delta_check",
    "removable_extension",
]

from .differentiation import (
    derivative,
    directional_derivative,
    gradient,
    hessian,
    jacobian,
    second_derivative,
    taylor_first_order,
)

from .integration import (
    cumulative_trapezoid,
    left_riemann,
    midpoint_rule,
    monte_carlo_integral,
    right_riemann,
    simpson_rule,
    trapezoidal_rule,
)

from .autodiff import (
    Dual,
    autodiff_derivative,
    autodiff_gradient,
    dual_cos,
    dual_exp,
    dual_log,
    dual_sin,
    gradient_check,
    relative_gradient_error,
)
