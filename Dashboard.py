# %%
import plotly.express as px
import streamlit as st
import numpy as np
import pandas as pd
from nelson_siegel_svensson.calibrate import calibrate_ns_ols, calibrate_nss_ols
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge
from matplotlib import pyplot as plt
from sklearn.pipeline import make_pipeline


@st.cache
def import_data(path, index_col=0, parse_dates=True):
    try:
        data = pd.read_csv(path, index_col=index_col, parse_dates=parse_dates)
        print("data imported")
    except FileNotFoundError:
        st.write("NotFoundError")
        raise
    return data


@st.cache
def spline_cubique(X_train, y_train, X_plot):
    X_plot = X_plot[:, np.newaxis]
    X_train = X_train[:, np.newaxis]
    model = make_pipeline(SplineTransformer(
        n_knots=4, degree=3), Ridge(alpha=1e-3))
    model.fit(X_train, y_train)
    return model.predict(X_plot)


@st.cache
def nelson_siegel(X_train, y_train, X_plot):
    # starting value of 1.0 for the optimization of tau
    curve, status = calibrate_ns_ols(X_train, y_train, tau0=1.0)
    assert status.success
    params = list(map(lambda x: round(x, 3), (curve.beta0,
                  curve.beta1, curve.beta2, curve.tau)))
    params_txt = r" ($\beta_0$"+f"={params[0]}, "+r"$\beta_1$"+f"={params[1]}, "\
        + r"$\beta_2$"+f"={params[2]}, "+r"$\tau$"+f"={params[3]})"
    return params_txt, curve(X_plot)


@st.cache
def nelson_siegel_sevenson(X_train, y_train, X_plot):
    # starting value of 1.0, 1.0 for the optimization of tau
    curve, status = calibrate_nss_ols(X_train, y_train, tau0=(2.5, 2.5))
    assert status.success
    params = list(map(lambda x: round(x, 3), (curve.beta0,
                  curve.beta1, curve.beta2, curve.tau1, curve.tau2)))
    parameters = r" ($\beta_0$"+f"={params[0]}, "+r"$\beta_1$"+f"={params[1]}, "\
        + r"$\beta_2$"+f"={params[1]}, "+r"$\tau_1$" + \
        f"={params[2]}, "+r"$\tau_2$"+f"={params[3]})"
    return parameters, curve(X_plot)


# %%
st.title("Modélisation des courbes de taux")
st.sidebar.title("Séléctionnez la partie")

sidebar_options = st.sidebar.selectbox(
    "Parties",
    ("Etude théorique", "Etude pratique")
)

if sidebar_options == "Etude théorique":
    st.write("# Introduction")
    st.write("## Définition Courbe des taux")
    st.write("""La courbe des taux d’intérêt d’une obligation financière est une fonction qui, à une date
    donnée et pour chaque maturité, indique le niveau de taux d’intérêt (coût de l’argent qui
    permet de calculer le prix qu’il faut payer pour emprunter de l’argent, ou rémunération
    reçue en cas de prêt d’argent associé).""")
    st.write("""La courbe des taux est représentée sur un repère avec sur l’axe des abscisses la matu-
    rité de l’obligation et sur l’ordonnée le taux d’intérêt. La forme de la courbe dépend en
    particulier des anticipations des agents privés concernant l’évolution de l’inflation et de la
    politique monétaire (car les taux d’intérêt relèvent de mécanismes de marché, la banque
    centrale ne contrôlant directement que les taux directeurs)""")
    st.write("""Ainsi, à une série de flux $F_t$ payés aux dates $T_t$, on associe une série de rendement $r_t$,
    on en déduit la valeur de marché $VM$ calculée à l’aide des facteurs d’actualisation $B_t$.""")
    st.write("""$$
    VM = \sum_{t>0} F_t*B_t \quad avec \quad B_t = 1/(1 + r_t)^T_t
    $$""")

    st.write("## La démarche pour construire la courbe des taux")
    st.write("### L'importance de la modélisation de la courbe des taux")
    st.write("""Construire la courbe des taux zéro-coupon au comptant « spot » est très important en
    pratique car elle permet aux acteurs du marché : """)
    st.write("""
    1. D’évaluer et de couvrir à la date de reconstitution les produits de taux délivrant des
    flux futurs connus (obligation à taux fixe, par exemple).
    2. De réaliser l’analyse «rich and cheap» (bond picking) qui consiste à détecter les
    produits sur-et sous-évalués par le marché pour tenter d’en tirer profit.
    3. De dériver les autres courbes implicites : la courbe des taux forward, la courbe des
    taux de rendement au pair et la courbe des taux de rendement instantanés.""")
    st.write("""La courbe des taux zéro-coupon au comptant permet de mettre en place des
    modèles stochastiques de déformation de cette courbe dans le temps.
    La construction de la courbe est rendue nécessaire par le fait qu’il n’existe pas suffisamment
    d’obligations zéro-coupon cotées sur le marché. Par conséquent, il n’est pas possible
    d’obtenir les taux zéro-coupon pour un continuum de maturité. En outre, les obligations
    zéro-coupon ont souvent une moindre liquidité que les obligations à coupons.""")

    st.write("### Sélection des titres pour la construction de courbe des taux")
    st.write("""Il est primordial d’effectuer une sélection rigoureuse des titres qui servent à la
    construction de la courbe des taux. Pour cela il faut éliminer :""")
    st.write("""
    1. Les titres qui présentent des clauses optionnelles car la présence d’options rend le
    prix de ces titres non homogènes avec ceux qui n’en contiennent pas.
    2. Les titres qui représentent des erreurs de prix causées par des erreurs de saisie.
    3. Les titres qui sont soit illiquides, soit surliquides, qui présentent des prix qui ne sont
    pas dans le marché.
    4. Les segments de maturité dans lesquels on ne dispose pas de titres.""")

    st.write("## Les modèles de courbe des taux")
    st.write("""Pour un panier d’obligations à coupons, il s’agit de la
    minimisation de l’écart au carré entre les prix de marché et les
    prix reconstitués à l ’aide d’une forme a priori spécifiée des taux
    zéro-coupon ou de la fonction d’actualisation.""")
    st.write("Soit un panier constitué de $n$ titres. On note à la date $t$:")
    st.write("""
    - $P_i^j$: prix de marché du j-ème titre.
    - $\hat{P}_i^j$: prix théorique du j-ème titre
    - $F_i^j$: flux futur du j-ème titre tombant à la date s (s > t)
    """)
    st.write("L’idée consiste à trouver le vecteur des paramètres tel que")
    st.write("""$$
    \min_{β} \sum_{j=1}^{n} (\hat{P}_i^j - P_i^j)^2
    $$""")
    st.write("""On distingue deux grandes classes de modèles:""")
    st.write("""- Les modèles type Nelson et Siegel fondés sur une modélisation
    des taux zéro-coupon. Le prix théorique s’écrit:
    $$
    \hat{P}_t^j = \sum_{s} F_s^j * B(t, s) = \sum_{s} F_s^j * \exp(-(s-t)*g(s-t;β))
    $$
    $g$ est la fonctionnelle des taux zéro-coupon. Le prix de l’obligation est une
    fonction non linéaire des paramètres d’estimation.""")
    st.write("""- Les modèles à splines fondés sur une modélisation de la
    fonction d’actualisation.
    $$
    \hat{P}_t^j = \sum_{s} F_s^j * B(t, s) = \sum_{s} F_s^j * f(s-t;β)
    $$
    $f$ est une fonction linéaire des paramètres d’estimation. Par
    conséquent, le prix de l’obligation est également une fonction
    linéaire des paramètres d’estimation""")

    st.write("### Le modèle de Nelson-Siegel")
    st.write("""La méthode de Nelson et Siegel est l’une des plus intéressante car
    ils partent de l’hypothèse où le taux instantané $r_t(θ)$ est la solution d’une équation
    différentielle, dans le cas d’une racine double. C’est-à-dire $r_t(θ)$ est de la forme :""")
    st.write(r"""
    $$
    r_t(θ) = β_0 + β_1 \exp(− \frac{θ}{τ}) + β_2(− \frac{θ}{τ}) \exp(− \frac{θ}{τ})
    $$
    """)
    st.write("""Puisque des équations différentielles peuvent générer des courbes de taux en S.
    Sachant que le taux d’intérêt est défini par :
    """)
    st.write(r"""
    $$
    R_t(θ) = \frac{1}{θ} \int_{0}^{θ} r_t(s) \,ds
    $$""")
    st.write("celui-ci a pour expression :")
    st.write(r"""
    $$
    R_t(θ) = β_0 + β_1 \left[ \frac{1 - \exp(− \frac{θ}{τ_1})}{- \frac{θ}{τ_1}} \right]
    + β_2 \left[ \frac{1 - \exp(− \frac{θ}{τ_1})}{- \frac{θ}{τ_1}} - \exp(- \frac{θ}{τ_1}) \right]
    $$""")
    st.write("""
    - $β_0$ : facteur de niveau; il s ’agit du taux long.
    - $β_1$ : facteur de rotation; il s’agit de l’écart entre le taux court et le taux long.
    - $β_2$ : paramètre d’échelle destiné à rester fixe au cours du temps.
    """)
    st.write("""Il est aisé d’exprimer les dérivées partielles de par rapport à chacun des
    paramètres béta, ce que l’on appelle les sensibilités des taux zéro-coupon aux paramètres béta.
    """)

    st.write("### Le modèle de Nelson-Siegel augmenté")
    st.write("""Le modèle de Nelson et Siegel ne permet pas de reconstituer
    toutes les formes de courbes de taux que l’on peut rencontrer
    sur le marché, en particulier les formes à une bosse et un creux.
    En outre, il manque de souplesse d’ajustement pour les
    maturités supérieures à 7 ans si bien que les obligations de
    telles maturités sont parfois mal évaluées par le modèle.
    Le premier inconvénient peut être levé en utilisant le modèle de
    Svensson ou modèle de Nelson-Siegel augmenté.""")
    st.write("La fonctionnelle s’écrit maintenant :")
    st.write(r"""
    $$
    R_t(θ) = β_0 + β_1 \left[ \frac{1 - \exp(− \frac{θ}{τ_1})}{- \frac{θ}{τ_1}} \right]
    + β_2 \left[ \frac{1 - \exp(− \frac{θ}{τ_1})}{- \frac{θ}{τ_1}} - \exp(- \frac{θ}{τ_1}) \right]
    + β_3 \left[ \frac{1 - \exp(− \frac{θ}{τ_2})}{- \frac{θ}{τ_2}} - \exp(- \frac{θ}{τ_2}) \right]
    $$""")
    st.write("""
    - $β_3$ : paramètre de courbure supplémentaire qui a surtout une
    influence sur la partie courte de la courbe
    - $τ_2$ : paramètre d’échelle
    """)
    st.write("""Cette extension donne plus de flexibilité à la courbe sur le
    secteur court terme.""")

    st.write("### Les modèles à splines")
    st.write("""Ils sont fondés sur une modélisation de la fonction d’actualisation.
    Les plus célèbres sont les splines polynomiaux  et les splines exponentielles.
    Leur avantage tient à leur grande flexibilité qui leur permet de
    reconstruire toutes les formes de courbe rencontrées sur le marché.""")
    st.write("""Il est commun de considérer l’écriture standard comme dans
    l’exemple qui suit:""")
    st.write(r"""
    $$
    B(0, s) =
    \begin{cases}
        B_0(s) = d_0 + c_0s + b_0 s^2 + a_0 s^2 + a_0 s^3 , s \in [0, 5]  \\
        B_5(s) = d_1 + c_1s + b_1 s^2 + a_1 s^2 + a_1 s^3 , s \in [5, 10]  \\
        B_{10}(s) = d_2 + c_2s + b_2 s^2 + a_2 s^2 + a_2 s^3 , s \in [10, 20]
    \end{cases}
    $$
    """)
    st.write("""La fonction d’actualisation compte ici $12$ paramètres.
    On rajoute des contraintes de régularité sur cette fonction qui
    garantissent la continuité, la continuité de la dérivée première et
    de la dérivée seconde de cette fonction aux points de raccord 5
    et 10.""")
    st.write(r"""
    $$
    \text { Pour } i = 0, 1, \text { et }2 \quad:
    \begin{cases}
        B_0^{(i)}(5) = B_5^{(i)}(5) \\
        B_{10}^{(i)}(10) = B_5^{(i)}(10)
    \end{cases}
    $$
    """)
    st.write("""Et la contrainte qui porte sur le facteur d’actualisation: $B_0(0) = 1$
    En utilisant l’ensemble de ces $7$ contraintes, le nombre de
    paramètres à estimer tombe à $5$:""")
    st.write(r"""
    $$
    B(0, s)=\left\{\begin{array}{c}
        B_{0}(s)=d_{0}+c_{0} s+b_{0} s^{2}+a_{0} s^{3}, s \in[0,5] \\
        B_{5}(s)=1+c_{0} s+b_{0} s^{2}+a_{0}\left[s^{3}-(s-5)^{3}\right]+a_{1}(s-5)^{3}, s \in[5,10] \\
        B_{10}(s)=1+c_{0} s+b_{0} s^{2}+a_{0}\left[s^{3}-(s-5)^{3}\right] \\
                +a_{1}\left[(s-5)^{3}-(s-10)^{3}\right]+a_{2}(s-10)^{3}, s \in[10,20]
    \end{array}\right.
    $$
    """)
    st.write("Le système précédent peut être écrit sous la forme suivante:")
    st.write(r"""
    $$
    B_{0}(s) = 1 + d_{0} + c_{0} s + b_{0} s^{2} + a_{0} s^{3} + (a_{1}-a_{0}).(s-5)_{+}^3 + (a_{2}-a_{1}).(s-10)_{+}^3 , \text { Pour } s \in[0,20]
    $$
    """)
    st.write("""Il y a une autre écriture de cette équation dans la base des B- splines. Cette écriture est devenu extrêmement classique.""")
    st.write("""Les B-splines sont des fonctions linéaires de fonctions bornées de puissance. On écrit alors:""")
    st.write(r"""
    $$
    B(0, s)=\sum_{l=-3}^{2} c_{l} B_{l}^{3}(s)=\sum_{l=-3}^{2} c_{l}\left(\sum_{j=l}^{l+4}\left[\prod_{i=l}^{l+4} \frac{1}{\lambda_{i}-\lambda_{j}}\right]\left(s-\lambda_{j}\right)_{+}^{3}\right) \\
    $$
    Où les coefficients lambda sont définis comme suit:
    $$
    \lambda_{-3}<\lambda_{-2}<\lambda_{-1}<\lambda_{0}=0<\lambda_{1}=5<\lambda_{2}=10<\lambda_{3}=20<\lambda_{4}<\lambda_{5}<\lambda_{6}
    $$
    $$
    B(0,0)=\sum_{l=-3}^{2} c_{l} B_{l}^{3}(0)=\sum_{l=-3}^{-1} c_{l} B_{l}^{3}(0)=1
    $$
    """)

    st.write("### Le Modèle de Vasicek")
    st.write("""Dans ce modèle, le taux sans risque est modélisé par un processus d'Ornstein-Uhlenbeck, sous la probabilité neutre au risque $Q$ :""")
    st.write(r"""$$
    dr_t  = k(θ-r_t)dt +σdW_t^Q
    $$""")
    st.write("""Avec θ la moyenne à long terme du processus, k le taux de retour à la moyenne,
    σ est le coefficient de diffusion de la volatilité et r est le taux d'intérêt sans risque.
    Les taux d'intérêt sont normalement distribués avec espérance et variance :
    """)
    st.write(r"""$$
    E[r_t|F_t] = r_s e^{-k(t-s)}+θ(1-e^{-k(t-s)})
    $$""")
    st.write(r"""
    $$
    Var[r_t│F_s]=\frac{σ^2}{2k} (1-e^{2k(t-s)})
    $$""")
    st.write(r"""
    Étant donné $s = 0$ et à tout instant $t$, on a
    $$
    r_t \sim N(θ+(r_0-θ) e^{-kt},\frac{σ^2}{2k} (1-e^{2k(t-s))})
    $$
    """)
    st.write(r"""
    On peut montrer qu'une obligation à coupon zéro avec une échéance au moment $T$ peut être trouvée en calculant
    l'espérance suivante sous la mesure de risque neutre $B(0, T)=E^{Q} [e^{-\int_{0}^{T} r_{u} d u}]$
    Où le processus de taux court $r_{u}$ peut-être à peu près n'importe quel processus. Nous pourrions estimer
    cette espérance en utilisant la simulation de Monte Carlo, mais le modèle de Vasicek nous permet de calculer
    la valeur de l'obligation à coupon zéro en utilisant la propriété de Markov et la fonction de densité.
    En utilisant cette méthode, nous pouvons évaluer l'obligation par les équations suivantes.
    """)
    st.write(r"""
    $B(0, T)=e^{-a(T)-b(T) r_{0}} \quad$ où $\quad b(\tau)=\frac{1-e^{\tau k}}{k} \quad$
    et $\quad a(\tau)=\left(\theta-\frac{\sigma^{2}}{2 k}\right)(\tau-b(\tau)) \frac{\sigma^{2}}{4 k} b(\tau)^{2}$
    """)
    st.write(r"""
    On peut montrer qu'une obligation à coupon zéro avec une échéance au moment $T$ peut être trouvée en calculant
    l'espérance suivante sous la mesure de risque neutre $B(0, T)=E^{Q} [e^{-\int_{0}^{T} r_{u} d u}]$
    Où le processus de taux court $r_{u}$ peut-être à peu près n'importe quel processus. Nous pourrions estimer
    cette espérance en utilisant la simulation de Monte Carlo, mais le modèle de Vasicek nous permet de calculer
    la valeur de l'obligation à coupon zéro en utilisant la propriété de Markov et la fonction de densité.
    En utilisant cette méthode, nous pouvons évaluer l'obligation par les équations suivantes.
    """)

# %%
else:
    # %%
    st.write("## Présentation des données")

    st.write("""Pour l'appliquation et la comparaison des différents modèles de courbes de taux,
    on va utiliser les taux zéro coupons basées sur un grand nombre d'obligations du Trésor US
    en circulation, et sont basées sur une convention de composition continue. Les valeurs sont
    des estimations anuelles des rendements à partir de 2000 pour toute la plage d'échéances couverte
    par les titres du Trésor en circulation.""")

    data = import_data("interest_rates.csv", index_col=0, parse_dates=True)
    st.write("#### Représentation tabulaire des données :")
    st.dataframe(data.set_index(data.index.year, drop=True))

    df_long = pd.melt(import_data("interest_rates.csv",
                      None, None), id_vars=['Date'])
    df_long.columns = ["Date", "Maturité", "Taux Z.C"]
    df_long["Maturité"] = df_long["Maturité"].map(lambda x: int(x[-2:]))
    plot_range = (df_long["Taux Z.C"].max()-df_long["Taux Z.C"].min())*0.1
    st.write("#### Le développement des taux Zéro coupons avec les années :")
    fig = px.scatter(df_long[::-1], x="Maturité", y="Taux Z.C",
                     animation_frame="Date", 
                     range_y=[df_long["Taux Z.C"].min()-plot_range,
                              df_long["Taux Z.C"].max()+plot_range])
    st.plotly_chart(fig)

    st.write("## Application des modèles :")
    st.write("#### Sélectionner l'année :")
    year = st.selectbox('Année', options=data.index, index=0,
                        format_func=lambda d: d.year)
    st.write(
        """#### Sélectionner la durée de maturité :""")
    maturity = st.slider("Maturité", min_value=5, max_value=30,
                         value=10)
    st.write("""#### Sélectionner le(s) modèle(s) à appliquer""")

    models = {'Splines Cubiques': spline_cubique, 'Nelson-Siegel': nelson_siegel,
              'Nelson-Siegel Augmenté': nelson_siegel_sevenson}
    model_names = st.multiselect(
        'Modèle', options=models.keys(), default=['Nelson-Siegel'])
    t = np.array(range(1, maturity+1))
    y = data.loc[year][:maturity]
    assert len(t) == len(y)
    time_line = np.linspace(1, maturity, 100)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.suptitle('Calibrated Nelson-Siegel Curve')
    ax.plot(t, y, 'r+',  label="Points réelles")
    for model_name in model_names:
        try:
            model_curve = models[model_name](t, y, time_line)
            parameters = ""
            if len(model_curve) == 2:
                parameters = model_curve[0]
                model_curve = model_curve[1]
        except AssertionError:
            st.warning(f"le modèle {model_name} n'a pas pu convergé")
            continue
        ax.plot(time_line, model_curve, label=model_name+parameters)

    ax.set_xlabel('Maturité')
    ax.set_ylabel('Taux Z.C')
    ax.legend(loc="lower right", prop={'size': 7.2})
    st.pyplot(fig)
    
    st.write("""## Comparaison et conclusions""")
    st.write("""Le reproche formulé à l’encontre des modèles de type
    Nelson-Siegel est leur insuffisante flexibilité, en l'addition des problèmes
    de convergence, qui sont souvent rencontré dans le modèle Nelson-Siegel augmenté, 
    pour estimer les paramétres supplémentaires. En revanche les 
    variables de ces modèles sont interprétables financièrement.""")
    st.write("""D'autre part, les modèles à splines sont beaucoup plus 
    flexibles, ce qui leur permet de reconstruire toutes les formes de courbe 
    rencontrées sur le marché. Mais présentent au contraire des
    paramètres qui ne sont pas interprétables d’un point de vue financier.""")
for _ in range(15):
    st.sidebar.write("")
st.sidebar.write("""
#### Réalisé Par :  
### Abdelghafour Ait Bennassar
#### Encadré par : 
### Dr. El Asri Brahim""")
