import numpy as np
import matplotlib.pyplot as plt
from kalman import predire, corriger

DT_PAS = 1.0          # déplacement commandé par pas
Q = 0.04              # variance du bruit de mouvement
R = 1.0               # variance du bruit de capteur
N_PAS = 50


def simuler(rng, q=Q, r=R, q_filtre=None, r_filtre=None):
    """Retourne (vraies, estimees, sigmas, mesures). q, r : le bruit RÉEL du monde
    q_filtre, r_filtre : ce que le filtre CROIT (par défaut, la vérité)
    """
    q_filtre = q if q_filtre is None else q_filtre
    r_filtre = r if r_filtre is None else r_filtre

    position = 0.0
    mu, sigma2 = 0.0, 100.0
    vraies, estimees, sigmas, mesures = [], [], [], []

    for _ in range(N_PAS):
        # 1. le monde bouge vraiment, avec son bruit
        position += DT_PAS + rng.normal(0, np.sqrt(q))
        # 2. le capteur mesure, avec son bruit
        z = position + rng.normal(0, np.sqrt(r))
        # 3. le filtre : prédire puis corriger
        mu, sigma2 = predire(mu, sigma2, DT_PAS, q_filtre)
        mu, sigma2 = corriger(mu, sigma2, z, r_filtre)
        # ... deux lignes à écrire ...

        vraies.append(position)
        estimees.append(mu)
        sigmas.append(np.sqrt(sigma2))
        mesures.append(z)

    return map(np.array, (vraies, estimees, sigmas, mesures))


def tracer(vraies, estimees, sigmas, mesures, titre, fichier):
    t = np.arange(len(vraies))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(t, mesures, ".", color="grey", alpha=0.6, label="mesures brutes")
    ax1.plot(t, vraies, "-", linewidth=2, color="k", label="position vraie")
    ax1.plot(t, estimees, "-", linewidth=2, color="tab:red", label="estimée (Kalman)")
    ax1.fill_between(t, estimees - 2*sigmas, estimees + 2*sigmas,
                     color="tab:red", alpha=0.2, label="±2σ")
    ax1.set_ylabel("position (m)")
    ax1.set_title(titre)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, np.abs(estimees - vraies), "-", color="tab:red", label="erreur réelle")
    ax2.plot(t, 2*sigmas, "--", color="k", label="±2σ annoncé par le filtre")
    ax2.set_xlabel("pas")
    ax2.set_ylabel("erreur (m)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.savefig(fichier, dpi=150)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    v, e, s, m = simuler(rng)
    print(f"erreur moyenne des mesures brutes : {np.abs(m - v).mean():.3f} m")
    print(f"erreur moyenne du filtre          : {np.abs(e - v).mean():.3f} m")
    print(f"sigma final annoncé par le filtre : {s[-1]:.3f} m")
    tracer(v, e, s, m, "Filtre de Kalman 1D", "kalman.png")
    plt.show()