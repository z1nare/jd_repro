"""NOT IMPLEMENTED -- a *measured* cost model to replace the router's rule.

:mod:`jdgram.engine.router` currently decides with one analytic inequality,
``tfirst iff m*T^2 < P_layer``, which compares the two routes' **workspace**.
Profiling at vocabulary scale showed that rule making the wrong call, and the
reason is that it optimises a quantity that has stopped mattering:

  * At V=50257 the LM head has ``m*T^2 = 2.1M`` against ``P_layer = 12.9M``, so
    the rule selects T-first.
  * T-first on a vocabulary head measures 12.4x slower per kernel than d-first.
  * The workspace it saves (24 MiB vs 393 MiB) is invisible at model scale,
    where peak is set by the forward activations and the held tied captures.
    Measured peak was identical on both routes -- 3526 MiB either way -- while
    the time difference was 122 ms per Gramian.

So the rule is faithfully minimising bytes that are not the peak, and it is
blind to time, which is the only axis on which the two routes differ here. A
correct model has to weigh both, and the coefficients are hardware-dependent
(kernel efficiency at a given ``(m, T, d_out, d_in)``), so they have to be
measured on the target card rather than derived.

Both routes are exactly equivalent numerically -- level L5 of
``bench/profile_suite.py`` measures them agreeing to 0.000e+00 at every
vocabulary tested -- so changing the rule can only change how long a run takes,
never what it produces.
"""

from __future__ import annotations


def crossover(*args, **kwargs):
    raise NotImplementedError(
        "cost model: not measured yet. jdgram.engine.router falls back to the "
        "analytic rule 'tfirst iff m*T^2 < P_layer', which is known to pick the "
        "slow route on a large vocabulary head -- see this module's docstring."
    )
