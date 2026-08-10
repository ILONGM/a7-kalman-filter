def predire(mu, sigma2, u, q):
    """Le robot avance de u ; son mouvement a une variance q.
    Retourne (nouvelle moyenne, nouvelle variance).
    """
    # deux lignes
    mu = mu + u
    sigma2 = sigma2 + q
    return mu, sigma2