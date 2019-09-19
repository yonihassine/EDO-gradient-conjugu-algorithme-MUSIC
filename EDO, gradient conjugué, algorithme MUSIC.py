#!/usr/bin/env python
# coding: utf-8

# # Différences finies et gradient conjugué

# In[2]:


get_ipython().run_line_magic('pylab', 'inline')


# # 1. Différences finis et gradient conjugué
# 
# Dans cet exercice, nous illustrons comment résoudre des équations différentielles ordinaire linéaire (EDO linéaire) en résolvant un système linéaire. Notre but est de trouver une solution **$1$-périodique** à l'équation
# $$
#     \boxed{x''(t) + x(t) = \exp(\sin(2 \pi t)).}
# $$

# ### 1./ Reformulation avec des matrices creuses
# 
# Pour cela, nous découpons le segment $[0, 1]$ en $L$ points régulièrement espacés $[t_0, t_2, \ldots, t_{L-1}] \in [0, 1)$ avec $t_0 = 0$ et $t_L = 1$ (attention aux indices). On pose donc $t_i = i/L$. Nous représentons une fonction $x : [0, 1] \to \mathbb{R}$ par le vecteur
# $$
#     x := \begin{pmatrix} x(t_0) \\ \vdots \\ x(t_{L-1}) \end{pmatrix} \in \mathbb{R}^{L}.
# $$
# De plus, nous approchons la dérivée seconde par des différences finies (avec les conditions de périodicité). C'est à dire qu'on pose $\varepsilon = 1/L$, puis
# $$
#     x''(t_i) \approx \dfrac{\big( x(t_{i+1}) + x(t_{i-1}) - 2 x(t_i) \big)}{\varepsilon^2}.
# $$
# On pose
# $$
#     (Dx)[i] = \dfrac{\big( x(t_{i+1}) + x(t_{i-1}) - 2 x(t_i) \big)}{\varepsilon^2}.
# $$
# 
# **Exercice** : Montrer que $D$ peut-être vu comme une matrice de $\mathcal{M}_{L}(\mathbb{R})$, puis écrire une fonction `getD(L)` qui renvoie cette matrice.
# 
# - On pourra soit utiliser la fonction modulo de python (`a%b` pour a modulo b), soit regarder la doc de la fonction `eye`.

# In[3]:


def getD(L):
    D = zeros((L, L))
    for i in range(L):
        D[i,i] = -2
        D[(i+1)%L,i] = 1
        D[(i-1)%L,i] = 1
    return D*L**2


# In[4]:


# Vérification, ce code ne doit pas faire d'erreur
D6 = getD(6)
assert D6[0,0] == -2*6*6
assert norm(dot(D6, ones(6))) < 1e-8
D6 # pour l'affichage


# En pratique, on prendra des très grandes valeurs de $L$. Il n'est pas donc pas pratique de **stocker** la matrice $D$ (presque tous ses coefficients sont nuls). On préfère utiliser une fonction *multiplier par D*.
# 
# **Exercice** : Ecrire une fonction `dd(x)` qui prend un `array` x de taille $L$, et renvoie l'`array` $D x$ de taille $L$, *sans construire D*. Autrement dit `dd(x)` renvoie directement le vecteur $x''$.
# - On pourra regarder la fonction `roll` de python.

# In[62]:


def dd(x):
    L = len(x)
    return L**2*(roll(x,1) + roll(x,-1) - 2*x)


# In[63]:


# Vérification, Ce code ne doit pas faire d'erreur
x = rand(6)
assert norm(dd(x) - dot(D6,x)) < 1e-8


# **Exercice** : Compiler la cellule suivante, et expliquer les résultats !

# In[64]:


L = 10000
x = rand(L)

print("\nCalcul de ddx en construisant la matrice :")
get_ipython().run_line_magic('time', 'v1 = dot(getD(L),x)')

print("\nCalcul de ddx directement :")
get_ipython().run_line_magic('time', 'v2 = dd(x)')

print("\nErreur entre les deux calculs : ", norm(v1 - v2))


# **Exercice**: Montrer que l'équation initiale peut s'écrire sous la forme $A x = b$, avec $A = D+1$ et $b$ le vecteur de taille $L$ qui contient les valeurs $\exp(\sin(2\pi t_i))$.

# **Exercice** : Ecrire une fonction `getb(L)` qui renvoie le vecteur `b`.
# - (Attention que dans la fonction `linspace(0,1,K)` de Python, le dernier élément de la liste créée est $1$ !)

# In[46]:


def getb(L):
    tt = linspace(0, 1-1/L, L)
    return exp(sin(2*pi*tt))


# In[47]:


# Vérification, ce code ne doit pas faire d'erreurs
assert norm(getb(5) - exp(sin(2*pi*array([0,1/5, 2/5, 3/5, 4/5])))) < 1e-6


# **Exercice** : Ecrire une fonction `A(x)` qui renvoie le résultat de la multiplication de $x$ par $A = D + 1$.

# In[12]:


def A(x):
    return dd(x) + x


# ### 2. Résolution du système avec l'algorithme du gradient conjugué
# 
# D'après l'exercice précédent, on peut résoudre l'équation différentielle en résolvant le système linéaire $Ax = b$. Pour cela, on utilise l'algorithme du gradient conjugué.
# 
# On rappelle que le gradient conjugué est défini par l'initialisation
# $$
#     x_0 = 0_{\mathbb{R}^L}, \ p_0 = b, \ r_0 = b
# $$
# puis
# \begin{align*}
#     \alpha_{n+1} & = \dfrac{r_n^T r_n}{p_n^T A p_n}, \\
#     x_{n+1} & = x_n + \alpha_{n+1} p_n, \\
#     r_{n+1} & = r_n - \alpha_{n+1} A p_n, \\
#     \beta_{n+1} & = \dfrac{r_{n+1}^T r_{n+1}}{r_n^T r_n}, \\
#     p_{n+1} & = r_{n+1} + \beta_{n+1} p_{n}.
# \end{align*}
# De plus, l'agorithme s'arrête dès que $n > L+2$, ou que $\| r_n \|$ est plus petit qu'une certaine tolérance.

# **Exercice** : Ecrire une fonction `solveGC(A,b,tol=1e-6)` qui trouve la solution de $Ax = b$ avec l'algorithme du gradient conjugué.
# 
# - On fera attention que dans notre cas, $A$ n'est pas une matrice, mais une fonction linéaire... On écrira donc l'algorithme dans ce cas

# In[13]:


def solveGC(A, b, tol=1e-6): 
    d = len(b)
    xn, pn, rn = zeros(d), b, b #Initialisation
    for n in range(d+1):
        if norm(rn) < tol: #Condition de sortie "usuelle"
            return xn
        Apn = A(pn) #une seule multiplication matrice/vecteur 
        alphan = dot(rn, rn)/dot(pn, Apn)
        xn, rnp1 = xn + alphan*pn, rn - alphan*Apn
        pn, rn = rnp1 + dot(rnp1, rnp1)/dot(rn, rn)*pn, rnp1
    print("Probleme , l’algorithme n’a pas convergé après",n,"itérations")
    return xn


# In[14]:


## Vérification, ce code ne doit pas faire d'erreurs
N = 20
b, B = rand(N), rand(N, N)
Cmatrix = dot(B.transpose(), B)+eye(N)
def Cfct(x): return dot(Cmatrix,x)

x1 = solveGC(Cfct, b) # votre fonction
x2 = solve(Cmatrix, b) # la fonction python
assert norm(x1 - x2) < 1e-6


# **Exercice** : Résoudre l'équation initiale avec l'algorithme du gradient conjugué avec $L = 1000$. Vérifier que votre solution $x$ vérifie bien $dd(x) + x = b$.

# In[16]:


L = 1000
tt = linspace(0, 1, L+1)
tt = tt[0:L]
    
b = getb(L)
x = solveGC(A,b)

plot(tt, x, 'r')
plot(tt, dd(x)+x-b, 'b')
print(norm(dd(x)+x-b))


# **Félicitations**, vous savez résoudre des EDOs (simples) avec un ordinateur. Pour ceux qui suivront le cours d'EDO numérique l'année prochaine, voici certaines questions en suspens :
# - Pourquoi les différences finies ?
# - Comment faire si $x(t)$ est un vecteur ?
# - L'algorithme est-il stable numériquement ?
# 
# etc. etc.

# ## 2. L'algorithme MUSIC
# 
# Dans l'algorithme MUSIC, on cherche la position de $N$ sources placés en $s_i$ avec $M$ récepteurs placés en $r_j$. Dans cet exercice, on regarde le problème dans le plan $\mathbb{R}^2$, et les positions des sources et des récepteurs seront sous la forme d'`array` de taille $N \times 2$ et $M \times 2$ respectivement.
# 
# ### 1/ Mise en place
# 
# 
# **Exercice** : Ecrire une fonction `getConfiguration(N,M)` qui renvoie les arrays `sources` et `recepteurs`. On pourra choisir :
# * les sources aléatoirement dans le carré $[-1, 1]^2$ (voir la fonction `rand`) ;
# * les récepteurs disposés régulièrement sur le cercle de centre $(0,0)$ et de rayon $2$.

# In[50]:


def getConfiguration(N,M):
    sources = 2*rand(N,2)-1 # La position des sources à trouver
    tt = linspace(0, 2*pi, M)
    recepteurs = array([2*cos(tt), 2*sin(tt)]).transpose() # La position des recepteurs (en cercle)
    return sources, recepteurs


# In[51]:


# Vérification, ce code doit afficher les récepteurs en cerle autour des sources
sources, recepteurs = getConfiguration(4,30)
axis('equal')
plot(recepteurs[:,0], recepteurs[:,1], '.b')
plot(sources[:,0], sources[:,1], 'xr')


# On suppose que chaque source émet un signal indépendant dans le temps. Ici, on prendra des signaux aléatoires entre $-1$ et $1$. De plus, comme il n'y a pas d'échelle de temps dans le problème, on peut supposer sans perte de généralité que les signaux émettent un signal à fréquence $1$, et les signaux émettent pendant un temps $T \in \mathbb{N}^*$.
# 
# **Exercice** : Ecrire une fonction `getInitialSignals(N,T=1000`) qui renvoie un array `F` de taille $N \times T$ de signaux aléatoires indépendants.

# In[52]:


def getInitialSignals(N,T=1000):
    F = rand(N,T)*2-1
    return F


# In[53]:


# Affichage des signaux
T = 1000
F = getInitialSignals(2,T)
subplot(211)
plot(range(T), F[0,:])
subplot(212)
plot(range(T), F[1,:])


# On peut vérifier que les signaux sont indépendants en calculant la matrice de corrélation
# $$ C_F = \frac{1}{T} \int_0^T F(t) F(t)^T dt.$$
# Si les signaux sont indépendants, cette matrice doit être quasi diagonale.
# 
# **Exercice** : Vérifiez que les signaux que vous avez générés sont indépendants. On pourra prendre $N = 4$.

# In[54]:


F = getInitialSignals(4, T)
CF = dot(F, F.transpose())/T
print(CF)
# On observe que les éléments sur la diagonale sont de l'ordre de 0.3, 
# alors que les autres sont de l'ordre de 0.01. Les signaux sont donc indépendants


# En pratique, on n'a pas accès à $F(t)$, mais seulement à ce qu'enregistrent les récepteurs. Comme nous l'avons vu en TD, le récepteur $j$ enregistre
# $$ g_j(t) = \sum_{i=1}^N f_i(t) \frac{1}{\| r_j - s_i \|}.$$
# ou encore
# $$ G(t) = \Phi(s_1; \ldots ; s_N) F(t)
# \quad \text{avec} \quad
# \Phi(x_1; \ldots ; x_N) = \begin{pmatrix} \Phi(x_1) & \Phi(x_2) & \ldots \Phi(x_N) \end{pmatrix}
# \quad \text{où} \quad
# \Phi(x) := \begin{pmatrix}\dfrac{1}{\| r_1 - x \|} \\ \vdots \\ \dfrac{1}{\| r_M - x \|} \end{pmatrix}.
# $$

# **Exercice** : Ecrire la fonction `Phi(X, recepteurs)` qui renvoie l'array $\Phi(x_1, \ldots, x_d)$ de taille $M \times d$, où $X$ est un `array` de taille $d \times 2$, et où les $(x_i)$ sont les lignes de $X$.
# * **Remarque** On pourra calculer $d$ avec `d = shape(X)[0]`, et $M$ avec `M = shape(recepteurs)[0]`.

# In[55]:


def Phi(X, recepteurs):
    d = shape(X)[0]
    M = shape(recepteurs)[0]
    phi = zeros((M,d))
    for i in range(M):
        for j in range(d):
            phi[i,j] = 1/norm(X[j,:] - recepteurs[i,:])
    return phi


# **Exercice** : Ecrire une fonction `getRecordedSignals(sources, recepteurs, F)` qui renvoie $G(t)$.

# In[56]:


def getRecordedSignals(sources, recepteurs, F):
    Phi_sources = Phi(sources, recepteurs)
    G = dot(Phi_sources,F)
    return G


# ### 2 Retrouver les positions des sources
# 
# On suppose maintenant qu'on n'a pas accès à l'array `source`, et qu'on ne connait seulement la position des récepteurs et $G$.
# 
# On a donc la configuration **fixée** suivante

# In[57]:


N,M,T = 4,30,1000
sources, recepteurs = getConfiguration(N,M)
F = getInitialSignals(N,T)
G = getRecordedSignals(sources, recepteurs, F)


# **Exercice** : Calculer la matrice de corrélation $C$ de $G$, et vérifier que $C$ n'a que $N$ grandes valeurs propres.

# In[60]:


C = dot(G, G.transpose())/T
D,U = eigh(C)
print("Il y a %d grandes valeurs propres"%len(D[D > 0.001]))


# **Exercice** : Calculer la matrice de projection `P*` sur les $N$ vecteurs propres de $C$ correspondants aux plus grandes valeurs propres de $C$. 
# * On rappelle que si $C u_i = \lambda_i u_i$, alors la matrice de projection sur $Vect(u_i)$ est $u_i u_i^T$.

# In[27]:


delta = 1e-3
vap = D[D > delta]
Ustar = U[:, D>delta]
Pstar = dot(Ustar, Ustar.transpose())


# **Exercice** : Ecrire une fonction `distanceMUSIC(x)` qui prend un point $x \in \mathbb{R}^2$, et renvoie la distance $\| P*(\Phi(x)) - \Phi(x) \|$

# In[28]:


def distanceMUSIC(x):
    return norm(dot(Pstar,Phi(x, recepteurs)) - Phi(x, recepteurs))


# **Exercice** : Tracer les courbes de niveau de la fonction `log(distanceMUSIC)` sur $[-1, 1] \times [-1, 1]$, ainsi que la vrai position des sources.
# - On pourra utiliser prendre 300 courbes de niveau, et utiliser la fonction `contourf`.

# In[29]:


xx = linspace(-1, 1, 100)
Z = [[distanceMUSIC(array([[x1, x2]])) for x1 in xx] for x2 in xx]


# In[30]:


axis('equal')
plot(sources[:,0], sources[:,1], 'xr')
contourf(xx, xx, log(Z), 300)


# In[ ]:




