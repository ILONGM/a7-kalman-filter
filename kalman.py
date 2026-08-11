"""Filtre de Kalman scalaire (1D).
La croyance sur la position est décrite par deux nombres seulement : une
estimation (le centre de la cloche) et une variance (sa largeur au carré).
Deux opérations alternent, exactement comme dans le filtre bayésien discret
d'A6 : prédire étale la cloche, corriger la resserre.
"""

import numpy as np


def predire(estimation, variance, deplacement, variance_mouvement):
    """Étape de prédiction : le robot bouge, l'incertitude augmente.
    On additionne deux quantités aléatoires — la position courante, déjà
    incertaine, et le déplacement, lui aussi incertain. Les erreurs s'empilent
    et les variances de termes indépendants s'additionnent, donc le résultat
    est nécessairement moins bien connu que le point de départ.
    Notation usuelle : estimation = mu, variance = sigma^2,
    deplacement = u, variance_mouvement = Q.
    """
    return estimation + deplacement, variance + variance_mouvement


def corriger(estimation, variance, mesure, variance_mesure):
    """Étape de correction : fusion de deux avis sur la même quantité.

    Contrairement à la prédiction, rien ne s'additionne ici : l'estimation et
    la mesure portent toutes deux sur la position actuelle. Elles se
    corroborent, et le résultat est plus sûr que chacune des deux prise
    séparément — y compris plus sûr que la meilleure.

    Le gain est le poids qui MINIMISE la variance du mélange. En pondérant
    l'estimation par w et la mesure par (1-w), la variance du résultat vaut
    w^2 * variance + (1-w)^2 * variance_mesure ; l'annuler en dérivée donne
    w = variance_mesure / (variance + variance_mesure), donc un poids sur la
    mesure de variance / (variance + variance_mesure). C'est ce qu'on appelle
    le gain de Kalman, et c'est le sens précis du mot « optimal ».

    Notation usuelle : estimation = mu, variance = sigma^2, mesure = z,
    variance_mesure = R, gain = K.
    """
    gain = variance / (variance + variance_mesure)
    innovation = mesure - estimation      # seule information nouvelle apportée
    return estimation + gain * innovation, (1.0 - gain) * variance


def variance_regime_permanent(variance_mouvement, variance_mesure):
    """Variance vers laquelle le filtre converge, quelles que soient les
    conditions initiales.

    À l'équilibre, ce que la prédiction ajoute est exactement ce que la
    correction retire. En notant p la variance après prédiction :

        p = s + Q  et  s = p*R / (p + R)

    d'où p^2 - Q*p - Q*R = 0, dont la racine positive est
    p = [Q + sqrt(Q^2 + 4*Q*R)] / 2. La variance après correction s'en déduit.

    Conséquence pratique : la précision atteinte ne dépend que du rapport
    entre bruit de mouvement et bruit de mesure — ni du temps écoulé, ni de
    l'incertitude de départ.
    """
    q = float(variance_mouvement)
    r = float(variance_mesure)
    if q < 0 or r < 0:
        raise ValueError("les variances doivent être positives ou nulles")
    apres_prediction = (q + np.sqrt(q**2 + 4 * q * r)) / 2
    return apres_prediction * r / (apres_prediction + r)