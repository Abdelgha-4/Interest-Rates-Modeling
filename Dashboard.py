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
st.title("Mod??lisation des courbes de taux")
st.sidebar.title("S??l??ctionnez la partie")

sidebar_options = st.sidebar.selectbox(
    "Parties",
    ("Etude th??orique", "Etude pratique")
)

if sidebar_options == "Etude th??orique":
    st.write("# Introduction")
    st.write("## D??finition Courbe des taux")
    st.write("""La courbe des taux d???int??r??t d???une obligation financi??re est une fonction qui, ?? une date
    donn??e et pour chaque maturit??, indique le niveau de taux d???int??r??t (co??t de l???argent qui
    permet de calculer le prix qu???il faut payer pour emprunter de l???argent, ou r??mun??ration
    re??ue en cas de pr??t d???argent associ??).""")
    st.write("""La courbe des taux est repr??sent??e sur un rep??re avec sur l???axe des abscisses la matu-
    rit?? de l???obligation et sur l???ordonn??e le taux d???int??r??t. La forme de la courbe d??pend en
    particulier des anticipations des agents priv??s concernant l?????volution de l???inflation et de la
    politique mon??taire (car les taux d???int??r??t rel??vent de m??canismes de march??, la banque
    centrale ne contr??lant directement que les taux directeurs)""")
    st.write("""Ainsi, ?? une s??rie de flux $F_t$ pay??s aux dates $T_t$, on associe une s??rie de rendement $r_t$,
    on en d??duit la valeur de march?? $VM$ calcul??e ?? l???aide des facteurs d???actualisation $B_t$.""")
    st.write("""$$
    VM = \sum_{t>0} F_t*B_t \quad avec \quad B_t = 1/(1 + r_t)^T_t
    $$""")

    st.write("## La d??marche pour construire la courbe des taux")
    st.write("### L'importance de la mod??lisation de la courbe des taux")
    st.write("""Construire la courbe des taux z??ro-coupon au comptant ?? spot ?? est tr??s important en
    pratique car elle permet aux acteurs du march?? : """)
    st.write("""
    1. D?????valuer et de couvrir ?? la date de reconstitution les produits de taux d??livrant des
    flux futurs connus (obligation ?? taux fixe, par exemple).
    2. De r??aliser l???analyse ??rich and cheap?? (bond picking) qui consiste ?? d??tecter les
    produits sur-et sous-??valu??s par le march?? pour tenter d???en tirer profit.
    3. De d??river les autres courbes implicites : la courbe des taux forward, la courbe des
    taux de rendement au pair et la courbe des taux de rendement instantan??s.""")
    st.write("""La courbe des taux z??ro-coupon au comptant permet de mettre en place des
    mod??les stochastiques de d??formation de cette courbe dans le temps.
    La construction de la courbe est rendue n??cessaire par le fait qu???il n???existe pas suffisamment
    d???obligations z??ro-coupon cot??es sur le march??. Par cons??quent, il n???est pas possible
    d???obtenir les taux z??ro-coupon pour un continuum de maturit??. En outre, les obligations
    z??ro-coupon ont souvent une moindre liquidit?? que les obligations ?? coupons.""")

    st.write("### S??lection des titres pour la construction de courbe des taux")
    st.write("""Il est primordial d???effectuer une s??lection rigoureuse des titres qui servent ?? la
    construction de la courbe des taux. Pour cela il faut ??liminer :""")
    st.write("""
    1. Les titres qui pr??sentent des clauses optionnelles car la pr??sence d???options rend le
    prix de ces titres non homog??nes avec ceux qui n???en contiennent pas.
    2. Les titres qui repr??sentent des erreurs de prix caus??es par des erreurs de saisie.
    3. Les titres qui sont soit illiquides, soit surliquides, qui pr??sentent des prix qui ne sont
    pas dans le march??.
    4. Les segments de maturit?? dans lesquels on ne dispose pas de titres.""")

    st.write("## Les mod??les de courbe des taux")
    st.write("""Pour un panier d???obligations ?? coupons, il s???agit de la
    minimisation de l?????cart au carr?? entre les prix de march?? et les
    prix reconstitu??s ?? l ???aide d???une forme a priori sp??cifi??e des taux
    z??ro-coupon ou de la fonction d???actualisation.""")
    st.write("Soit un panier constitu?? de $n$ titres. On note ?? la date $t$:")
    st.write("""
    - $P_i^j$: prix de march?? du j-??me titre.
    - $\hat{P}_i^j$: prix th??orique du j-??me titre
    - $F_i^j$: flux futur du j-??me titre tombant ?? la date s (s > t)
    """)
    st.write("L???id??e consiste ?? trouver le vecteur des param??tres tel que")
    st.write("""$$
    \min_{??} \sum_{j=1}^{n} (\hat{P}_i^j - P_i^j)^2
    $$""")
    st.write("""On distingue deux grandes classes de mod??les:""")
    st.write("""- Les mod??les type Nelson et Siegel fond??s sur une mod??lisation
    des taux z??ro-coupon. Le prix th??orique s?????crit:
    $$
    \hat{P}_t^j = \sum_{s} F_s^j * B(t, s) = \sum_{s} F_s^j * \exp(-(s-t)*g(s-t;??))
    $$
    $g$ est la fonctionnelle des taux z??ro-coupon. Le prix de l???obligation est une
    fonction non lin??aire des param??tres d???estimation.""")
    st.write("""- Les mod??les ?? splines fond??s sur une mod??lisation de la
    fonction d???actualisation.
    $$
    \hat{P}_t^j = \sum_{s} F_s^j * B(t, s) = \sum_{s} F_s^j * f(s-t;??)
    $$
    $f$ est une fonction lin??aire des param??tres d???estimation. Par
    cons??quent, le prix de l???obligation est ??galement une fonction
    lin??aire des param??tres d???estimation""")

    st.write("### Le mod??le de Nelson-Siegel")
    st.write("""La m??thode de Nelson et Siegel est l???une des plus int??ressante car
    ils partent de l???hypoth??se o?? le taux instantan?? $r_t(??)$ est la solution d???une ??quation
    diff??rentielle, dans le cas d???une racine double. C???est-??-dire $r_t(??)$ est de la forme :""")
    st.write(r"""
    $$
    r_t(??) = ??_0 + ??_1 \exp(??? \frac{??}{??}) + ??_2(??? \frac{??}{??}) \exp(??? \frac{??}{??})
    $$
    """)
    st.write("""Puisque des ??quations diff??rentielles peuvent g??n??rer des courbes de taux en S.
    Sachant que le taux d???int??r??t est d??fini par :
    """)
    st.write(r"""
    $$
    R_t(??) = \frac{1}{??} \int_{0}^{??} r_t(s) \,ds
    $$""")
    st.write("celui-ci a pour expression :")
    st.write(r"""
    $$
    R_t(??) = ??_0 + ??_1 \left[ \frac{1 - \exp(??? \frac{??}{??_1})}{- \frac{??}{??_1}} \right]
    + ??_2 \left[ \frac{1 - \exp(??? \frac{??}{??_1})}{- \frac{??}{??_1}} - \exp(- \frac{??}{??_1}) \right]
    $$""")
    st.write("""
    - $??_0$ : facteur de niveau; il s ???agit du taux long.
    - $??_1$ : facteur de rotation; il s???agit de l?????cart entre le taux court et le taux long.
    - $??_2$ : param??tre d?????chelle destin?? ?? rester fixe au cours du temps.
    """)
    st.write("""Il est ais?? d???exprimer les d??riv??es partielles de par rapport ?? chacun des
    param??tres b??ta, ce que l???on appelle les sensibilit??s des taux z??ro-coupon aux param??tres b??ta.
    """)

    st.write("### Le mod??le de Nelson-Siegel augment??")
    st.write("""Le mod??le de Nelson et Siegel ne permet pas de reconstituer
    toutes les formes de courbes de taux que l???on peut rencontrer
    sur le march??, en particulier les formes ?? une bosse et un creux.
    En outre, il manque de souplesse d???ajustement pour les
    maturit??s sup??rieures ?? 7 ans si bien que les obligations de
    telles maturit??s sont parfois mal ??valu??es par le mod??le.
    Le premier inconv??nient peut ??tre lev?? en utilisant le mod??le de
    Svensson ou mod??le de Nelson-Siegel augment??.""")
    st.write("La fonctionnelle s?????crit maintenant :")
    st.write(r"""
    $$
    R_t(??) = ??_0 + ??_1 \left[ \frac{1 - \exp(??? \frac{??}{??_1})}{- \frac{??}{??_1}} \right]
    + ??_2 \left[ \frac{1 - \exp(??? \frac{??}{??_1})}{- \frac{??}{??_1}} - \exp(- \frac{??}{??_1}) \right]
    + ??_3 \left[ \frac{1 - \exp(??? \frac{??}{??_2})}{- \frac{??}{??_2}} - \exp(- \frac{??}{??_2}) \right]
    $$""")
    st.write("""
    - $??_3$ : param??tre de courbure suppl??mentaire qui a surtout une
    influence sur la partie courte de la courbe
    - $??_2$ : param??tre d?????chelle
    """)
    st.write("""Cette extension donne plus de flexibilit?? ?? la courbe sur le
    secteur court terme.""")

    st.write("### Les mod??les ?? splines")
    st.write("""Ils sont fond??s sur une mod??lisation de la fonction d???actualisation.
    Les plus c??l??bres sont les splines polynomiaux  et les splines exponentielles.
    Leur avantage tient ?? leur grande flexibilit?? qui leur permet de
    reconstruire toutes les formes de courbe rencontr??es sur le march??.""")
    st.write("""Il est commun de consid??rer l?????criture standard comme dans
    l???exemple qui suit:""")
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
    st.write("""La fonction d???actualisation compte ici $12$ param??tres.
    On rajoute des contraintes de r??gularit?? sur cette fonction qui
    garantissent la continuit??, la continuit?? de la d??riv??e premi??re et
    de la d??riv??e seconde de cette fonction aux points de raccord 5
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
    st.write("""Et la contrainte qui porte sur le facteur d???actualisation: $B_0(0) = 1$
    En utilisant l???ensemble de ces $7$ contraintes, le nombre de
    param??tres ?? estimer tombe ?? $5$:""")
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
    st.write("Le syst??me pr??c??dent peut ??tre ??crit sous la forme suivante:")
    st.write(r"""
    $$
    B_{0}(s) = 1 + d_{0} + c_{0} s + b_{0} s^{2} + a_{0} s^{3} + (a_{1}-a_{0}).(s-5)_{+}^3 + (a_{2}-a_{1}).(s-10)_{+}^3 , \text { Pour } s \in[0,20]
    $$
    """)
    st.write("""Il y a une autre ??criture de cette ??quation dans la base des B- splines. Cette ??criture est devenu extr??mement classique.""")
    st.write("""Les B-splines sont des fonctions lin??aires de fonctions born??es de puissance. On ??crit alors:""")
    st.write(r"""
    $$
    B(0, s)=\sum_{l=-3}^{2} c_{l} B_{l}^{3}(s)=\sum_{l=-3}^{2} c_{l}\left(\sum_{j=l}^{l+4}\left[\prod_{i=l}^{l+4} \frac{1}{\lambda_{i}-\lambda_{j}}\right]\left(s-\lambda_{j}\right)_{+}^{3}\right) \\
    $$
    O?? les coefficients lambda sont d??finis comme suit:
    $$
    \lambda_{-3}<\lambda_{-2}<\lambda_{-1}<\lambda_{0}=0<\lambda_{1}=5<\lambda_{2}=10<\lambda_{3}=20<\lambda_{4}<\lambda_{5}<\lambda_{6}
    $$
    $$
    B(0,0)=\sum_{l=-3}^{2} c_{l} B_{l}^{3}(0)=\sum_{l=-3}^{-1} c_{l} B_{l}^{3}(0)=1
    $$
    """)

    st.write("### Le Mod??le de Vasicek")
    st.write("""Dans ce mod??le, le taux sans risque est mod??lis?? par un processus d'Ornstein-Uhlenbeck, sous la probabilit?? neutre au risque $Q$ :""")
    st.write(r"""$$
    dr_t  = k(??-r_t)dt +??dW_t^Q
    $$""")
    st.write("""Avec ?? la moyenne ?? long terme du processus, k le taux de retour ?? la moyenne,
    ?? est le coefficient de diffusion de la volatilit?? et r est le taux d'int??r??t sans risque.
    Les taux d'int??r??t sont normalement distribu??s avec esp??rance et variance :
    """)
    st.write(r"""$$
    E[r_t|F_t] = r_s e^{-k(t-s)}+??(1-e^{-k(t-s)})
    $$""")
    st.write(r"""
    $$
    Var[r_t???F_s]=\frac{??^2}{2k} (1-e^{2k(t-s)})
    $$""")
    st.write(r"""
    ??tant donn?? $s = 0$ et ?? tout instant $t$, on a
    $$
    r_t \sim N(??+(r_0-??) e^{-kt},\frac{??^2}{2k} (1-e^{2k(t-s))})
    $$
    """)
    st.write(r"""
    On peut montrer qu'une obligation ?? coupon z??ro avec une ??ch??ance au moment $T$ peut ??tre trouv??e en calculant
    l'esp??rance suivante sous la mesure de risque neutre $B(0, T)=E^{Q} [e^{-\int_{0}^{T} r_{u} d u}]$
    O?? le processus de taux court $r_{u}$ peut-??tre ?? peu pr??s n'importe quel processus. Nous pourrions estimer
    cette esp??rance en utilisant la simulation de Monte Carlo, mais le mod??le de Vasicek nous permet de calculer
    la valeur de l'obligation ?? coupon z??ro en utilisant la propri??t?? de Markov et la fonction de densit??.
    En utilisant cette m??thode, nous pouvons ??valuer l'obligation par les ??quations suivantes.
    """)
    st.write(r"""
    $B(0, T)=e^{-a(T)-b(T) r_{0}} \quad$ o?? $\quad b(\tau)=\frac{1-e^{\tau k}}{k} \quad$
    et $\quad a(\tau)=\left(\theta-\frac{\sigma^{2}}{2 k}\right)(\tau-b(\tau)) \frac{\sigma^{2}}{4 k} b(\tau)^{2}$
    """)
    st.write(r"""
    On peut montrer qu'une obligation ?? coupon z??ro avec une ??ch??ance au moment $T$ peut ??tre trouv??e en calculant
    l'esp??rance suivante sous la mesure de risque neutre $B(0, T)=E^{Q} [e^{-\int_{0}^{T} r_{u} d u}]$
    O?? le processus de taux court $r_{u}$ peut-??tre ?? peu pr??s n'importe quel processus. Nous pourrions estimer
    cette esp??rance en utilisant la simulation de Monte Carlo, mais le mod??le de Vasicek nous permet de calculer
    la valeur de l'obligation ?? coupon z??ro en utilisant la propri??t?? de Markov et la fonction de densit??.
    En utilisant cette m??thode, nous pouvons ??valuer l'obligation par les ??quations suivantes.
    """)

# %%
else:
    # %%
    st.write("## Pr??sentation des donn??es")

    st.write("""Pour l'appliquation et la comparaison des diff??rents mod??les de courbes de taux,
    on va utiliser les taux z??ro coupons bas??es sur un grand nombre d'obligations du Tr??sor US
    en circulation, et sont bas??es sur une convention de composition continue. Les valeurs sont
    des estimations anuelles des rendements ?? partir de 2000 pour toute la plage d'??ch??ances couverte
    par les titres du Tr??sor en circulation.""")

    data = import_data("interest_rates.csv", index_col=0, parse_dates=True)
    st.write("#### Repr??sentation tabulaire des donn??es :")
    st.dataframe(data.set_index(data.index.year, drop=True))

    df_long = pd.melt(import_data("interest_rates.csv",
                      None, None), id_vars=['Date'])
    df_long.columns = ["Date", "Maturit??", "Taux Z.C"]
    df_long["Maturit??"] = df_long["Maturit??"].map(lambda x: int(x[-2:]))
    plot_range = (df_long["Taux Z.C"].max()-df_long["Taux Z.C"].min())*0.1
    st.write("#### Le d??veloppement des taux Z??ro coupons avec les ann??es :")
    fig = px.scatter(df_long[::-1], x="Maturit??", y="Taux Z.C",
                     animation_frame="Date", 
                     range_y=[df_long["Taux Z.C"].min()-plot_range,
                              df_long["Taux Z.C"].max()+plot_range])
    st.plotly_chart(fig)

    st.write("## Application des mod??les :")
    st.write("#### S??lectionner l'ann??e :")
    year = st.selectbox('Ann??e', options=data.index, index=0,
                        format_func=lambda d: d.year)
    st.write(
        """#### S??lectionner la dur??e de maturit?? :""")
    maturity = st.slider("Maturit??", min_value=5, max_value=30,
                         value=10)
    st.write("""#### S??lectionner le(s) mod??le(s) ?? appliquer""")

    models = {'Splines Cubiques': spline_cubique, 'Nelson-Siegel': nelson_siegel,
              'Nelson-Siegel Augment??': nelson_siegel_sevenson}
    model_names = st.multiselect(
        'Mod??le', options=models.keys(), default=['Nelson-Siegel'])
    t = np.array(range(1, maturity+1))
    y = data.loc[year][:maturity]
    assert len(t) == len(y)
    time_line = np.linspace(1, maturity, 100)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.suptitle('Calibrated Nelson-Siegel Curve')
    ax.plot(t, y, 'r+',  label="Points r??elles")
    for model_name in model_names:
        try:
            model_curve = models[model_name](t, y, time_line)
            parameters = ""
            if len(model_curve) == 2:
                parameters = model_curve[0]
                model_curve = model_curve[1]
        except AssertionError:
            st.warning(f"le mod??le {model_name} n'a pas pu converg??")
            continue
        ax.plot(time_line, model_curve, label=model_name+parameters)

    ax.set_xlabel('Maturit??')
    ax.set_ylabel('Taux Z.C')
    ax.legend(loc="lower right", prop={'size': 7.2})
    st.pyplot(fig)
    
    st.write("""## Comparaison et conclusions""")
    st.write("""Le reproche formul?? ?? l???encontre des mod??les de type
    Nelson-Siegel est leur insuffisante flexibilit??, en l'addition des probl??mes
    de convergence, qui sont souvent rencontr?? dans le mod??le Nelson-Siegel augment??, 
    pour estimer les param??tres suppl??mentaires. En revanche les 
    variables de ces mod??les sont interpr??tables financi??rement.""")
    st.write("""D'autre part, les mod??les ?? splines sont beaucoup plus 
    flexibles, ce qui leur permet de reconstruire toutes les formes de courbe 
    rencontr??es sur le march??. Mais pr??sentent au contraire des
    param??tres qui ne sont pas interpr??tables d???un point de vue financier.""")
for _ in range(15):
    st.sidebar.write("")
st.sidebar.write("""
#### R??alis?? Par :  
### Abdelghafour Ait Bennassar
#### Encadr?? par : 
### Dr. El Asri Brahim""")
