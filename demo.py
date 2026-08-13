import numpy as np
import matplotlib.pyplot as plt
from kalman import predire, corriger

DEPLACEMENT = 1.0              # ce qu'on commande à chaque pas (m)
N_PAS = 50

# le bruit RÉEL du monde — sert uniquement à tirer les nombres aléatoires
VARIANCE_MOUVEMENT_REELLE = 0.04    # les roues dérapent de ±0.2 m par pas
VARIANCE_MESURE_REELLE = 1.0        # le capteur se trompe de ±1 m


def simuler(rng,
            variance_mouvement_reelle=VARIANCE_MOUVEMENT_REELLE,
            variance_mesure_reelle=VARIANCE_MESURE_REELLE,
            variance_mouvement_supposee=None,
            variance_mesure_supposee=None,
            n_pas=N_PAS):
    """Deux mondes en parallèle, sans communication entre eux.

    Les variances RÉELLES fabriquent les tirages aléatoires : elles décrivent
    la physique, que personne ne connaît sur un vrai robot.
    Les variances SUPPOSÉES sont les deux réglages du filtre : ce qu'on lui
    déclare sur la fiabilité de son mouvement et de son capteur. Par défaut,
    on lui dit la vérité — situation qui n'existe jamais en pratique.
    """
    if variance_mouvement_supposee is None:
        variance_mouvement_supposee = variance_mouvement_reelle
    if variance_mesure_supposee is None:
        variance_mesure_supposee = variance_mesure_reelle

    position = 0.0                      # la vérité, invisible pour le filtre
    estimation, variance = 0.0, 100.0   # la croyance : où, et à quel point j'en suis sûr

    positions, estimations, ecarts_types = [], [], []
    mesures, innovations_normalisees = [], []

    for _ in range(n_pas):
        # le monde bouge vraiment, avec son bruit
        position += DEPLACEMENT + rng.normal(0, np.sqrt(variance_mouvement_reelle))
        # le capteur lit la position vraie, avec son bruit
        mesure = position + rng.normal(0, np.sqrt(variance_mesure_reelle))

        # le filtre : d'abord le mouvement (la cloche s'élargit)
        estimation, variance = predire(estimation, variance,DEPLACEMENT, variance_mouvement_supposee)

        # diagnostic : la surprise observée, rapportée à la surprise annoncée.
        # Le filtre prédit la lecture à venir : centrée sur estimation, d'écart
        # typique sqrt(variance + variance_mesure_supposee). Si ses réglages sont
        # cohérents, ce rapport a une variance de 1. Ce calcul n'utilise jamais
        # la position vraie — il est donc faisable sur un vrai robot.
        innovation = mesure - estimation
        innovations_normalisees.append(innovation / np.sqrt(variance + variance_mesure_supposee))

        # puis la mesure (la cloche se resserre)
        estimation, variance = corriger(estimation, variance,mesure, variance_mesure_supposee)

        positions.append(position)
        estimations.append(estimation)
        ecarts_types.append(np.sqrt(variance))
        mesures.append(mesure)

    return {
        "positions": np.array(positions),
        "estimations": np.array(estimations),
        "ecarts_types": np.array(ecarts_types),
        "mesures": np.array(mesures),
        "innovations_normalisees": np.array(innovations_normalisees),
    }


def resumer(nom, run):
    """Trois chiffres : la précision obtenue, l'incertitude annoncée, et la
    cohérence entre les deux."""
    erreur = np.abs(run["estimations"] - run["positions"]).mean()
    sigma_final = run["ecarts_types"][-1]
    coherence = run["innovations_normalisees"][5:].var()   # on saute le transitoire
    print(f"{nom:34s} | erreur {erreur:6.3f} m | sigma annoncé {sigma_final:6.3f} m "
          f"| cohérence {coherence:7.3f}")
    return erreur, sigma_final, coherence


def tracer(run, titre, fichier):
    t = np.arange(len(run["positions"]))
    estimations = run["estimations"]
    sigmas = run["ecarts_types"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(t, run["mesures"], ".", color="grey", alpha=0.6, label="mesures brutes")
    ax1.plot(t, run["positions"], "-", linewidth=2, color="k", label="position vraie")
    ax1.plot(t, estimations, "-", linewidth=2, color="tab:red", label="estimée (Kalman)")
    ax1.fill_between(t, estimations - 2 * sigmas, estimations + 2 * sigmas,
                     color="tab:red", alpha=0.2, label="±2σ annoncé")
    ax1.set_ylabel("position (m)")
    ax1.set_title(titre)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, np.abs(estimations - run["positions"]), "-", color="tab:red",
             label="erreur réelle")
    ax2.plot(t, 2 * sigmas, "--", color="k", label="±2σ annoncé par le filtre")
    ax2.set_xlabel("pas")
    ax2.set_ylabel("erreur (m)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.savefig(fichier, dpi=150)


if __name__ == "__main__":
    # (nom, variance de mouvement supposée, variance de mesure supposée)
    configs = [
        ("filtre bien réglé",              None,   None),
        ("croit ses roues parfaites",      0.0,    None),
        ("doute trop de ses roues",        1.0,    None),
        ("croit son capteur excellent",    None,   0.01),
        ("croit son capteur mauvais",      None,   100.0),
    ]

    print("cohérence : variance des innovations normalisées, doit valoir ~1\n")
    reference = None
    for nom, q_sup, r_sup in configs:
        rng = np.random.default_rng(42)      # même monde pour tous les runs
        run = simuler(rng,
                      variance_mouvement_supposee=q_sup,
                      variance_mesure_supposee=r_sup)
        resumer(nom, run)
        if reference is None:
            reference = run

    tracer(reference, "Filtre de Kalman 1D — réglages corrects", "kalman.png")
    plt.show()