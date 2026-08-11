import pytest
import numpy as np
from kalman import predire, corriger, variance_regime_permanent


def test_predire_etale():
    assert np.allclose(predire(5.0, 0.5, 2.0, 0.3), (7.0, 0.8))

def test_corriger_resserre():
    gain = 25.0 / 26.0               
    assert np.allclose(corriger(20.0, 25.0, 24.0, 1.0), (20*(1-gain) + gain*24.0, (1.0 - gain) * 25.0))
    variance_finale = corriger(20.0, 25.0, 24.0, 1.0)[1]
    assert variance_finale < min(25.0, 1.0)

def test_variance_regime_permanent():
    q, r = 0.04, 1.0
    p_eq = variance_regime_permanent(q, r)
    # vérifier que la variance est stable à l'équilibre
    p_eq2 = predire(p_eq, p_eq, 0.0, q)[1]
    s_eq2 = corriger(p_eq2, p_eq2, 0.0, r)[1]
    assert np.allclose(s_eq2, p_eq)

