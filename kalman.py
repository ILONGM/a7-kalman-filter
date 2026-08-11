def predire(mu, sigma2, u, q):
    """Le robot avance de u ; son mouvement a une variance q.
    Retourne (nouvelle moyenne, nouvelle variance).
    """
    # deux lignes
    mu = mu + u
    sigma2 = sigma2 + q
    return mu, sigma2

def corriger(mu, sigma2, z, r):
    """Fusionne l'estimation (mu, sigma2) avec une mesure z de variance r.
    Retourne (nouvelle moyenne, nouvelle variance).
    """
    K = sigma2 / (sigma2 + r)
    mu_nouveau = mu + K * (z - mu) #nouvelle moyenne pondérée par K (calcul factirisé à développer)
    sigma2_nouveau = 1/ (1/sigma2 + 1/r)
    return mu_nouveau, sigma2_nouveau
