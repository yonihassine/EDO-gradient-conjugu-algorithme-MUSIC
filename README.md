# EDO-gradient-conjugu-algorithme-MUSIC
MUSIC (MUltiple SIgnal Classification), ou classification multiple des signaux, est un algorithme utilisé pour l'estimation de fréquence1 et la localisation d'émetteurs2. Cet algorithme permet de déterminer la direction des signaux incidents sur un réseau de capteurs même lorsque le rapport signal-à-bruit est très faible.
Ici, on regarde le problème dans le plan  ℝ2 , et les positions des sources et des récepteurs seront sous la forme d'array de taille  𝑁×2 et  𝑀×2 
respectivement.
On suppose que chaque source émet un signal indépendant dans le temps. Ici, on prendra des signaux aléatoires entre  −1  et  1 . De plus, comme il n'y a pas d'échelle de temps dans le problème, on peut supposer sans perte de généralité que les signaux émettent un signal à fréquence  1 , et les signaux émettent pendant un temps  𝑇∈ℕ∗ .
