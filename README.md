# Kmeans

Ce notebook présente une implémentation de l'algorithme du KMeans qui peut être appliqué à des dimensions quelconques. Une version spécifique au traitement d'image sera aussi présenté.

Testé avec :
- Python 3.13.12
- numpy==2.3.4
- scipy==1.16.3
- skimage==0.25.2


```python
# Imports and fixing the seed
import numpy as np
import matplotlib.pyplot as plt
import scipy
import skimage

np.random.seed(789)
```


```python
!python --version
```

    Python 3.13.12



```python
print('\n'.join(f'{m.__name__}=={m.__version__}' for m in globals().values() if getattr(m, '__version__', None)))
```

    numpy==2.3.4
    scipy==1.16.3
    skimage==0.25.2


## Chargement des données de tests

### Génération de clusters quelconques

Génération de clusters suivant une loi gaussienne aux paramètres aléatoires


```python
def generate_random_cluster(nb_cluster, nb_points_per_cluster=50, dimension=2):
    random_generated_data = np.zeros((nb_points_per_cluster * nb_cluster, dimension))
    # Create all the points following the normal law for each cluster
    for i in range(nb_cluster):
        # Create the distribution
        mu = np.random.randint(100)
        sigma = np.random.randint(5)
        distrib = np.random.normal(mu, sigma, (nb_points_per_cluster, dimension))
        # add the genarated points to  
        random_generated_data[i*nb_points_per_cluster: (i+1)*nb_points_per_cluster] = distrib
    np.random.shuffle(random_generated_data) # shuffle to avoid biais when tested
    return random_generated_data
```


```python
random_generated_2_cluster = generate_random_cluster(2)

# Show data on graph
plt.scatter(random_generated_2_cluster[:,0], random_generated_2_cluster[:,1])
plt.show()
```


    
![png](img/output_10_0.png)
    



```python
random_generated_5_cluster= generate_random_cluster(5)

# Show data on graph
plt.scatter(random_generated_5_cluster[:,0], random_generated_5_cluster[:,1])
plt.show()
```


    
![png](img/output_11_0.png)
    


### Import des données du fichier .csv

Import du jeu de test depuis le fichier représentant 6 classes


```python
data_from_file = np.genfromtxt("data/kmeans_6_classes.csv", delimiter=";", skip_header=1)
print(data_from_file)
np.random.shuffle(data_from_file) # shuffle to avoid biais when tested

# Show data on graph
plt.scatter(data_from_file[:,0], data_from_file[:,1])
plt.show()
```

    [[2.38360866 1.46715007]
     [2.20152865 1.63564779]
     [2.36495777 1.26597762]
     ...
     [2.49055014 2.24624783]
     [2.3978519  2.05521342]
     [2.59315747 1.93219502]]



    
![png](img/output_14_1.png)
    


### Importation des données d'image

Import d'une image sous forme de tableau RVB


```python
# Generate RGB array from image
data_from_image = skimage.io.imread("data/fruits.jpg")
print(f"raw data: \n shape:{data_from_image.shape}\n data: {data_from_image}")

# Reshaping the image to flatten all the lines
data_from_image_reshape = data_from_image.reshape(-1, 3)
print(f"flatten lines data: \n shape: {data_from_image_reshape.shape}\n data: {data_from_image_reshape}")

# Show image on RGB figure
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')
ax.scatter(data_from_image_reshape[:,0], data_from_image_reshape[:,1], data_from_image_reshape[:,2])
plt.show()
```

    raw data: 
     shape:(279, 370, 3)
     data: [[[255 255 255]
      [255 255 255]
      [255 255 255]
      ...
      [255 255 255]
      [255 255 255]
      [255 255 255]]
    
     [[255 255 255]
      [255 255 255]
      [255 255 255]
      ...
      [255 255 255]
      [255 255 255]
      [255 255 255]]
    
     [[255 255 255]
      [255 255 255]
      [255 255 255]
      ...
      [255 255 255]
      [255 255 255]
      [255 255 255]]
    
     ...
    
     [[255 255 255]
      [255 255 255]
      [255 255 255]
      ...
      [255 255 255]
      [255 255 255]
      [255 255 255]]
    
     [[255 255 255]
      [255 255 255]
      [255 255 255]
      ...
      [255 255 255]
      [255 255 255]
      [255 255 255]]
    
     [[255 255 255]
      [255 255 255]
      [255 255 255]
      ...
      [255 255 255]
      [255 255 255]
      [255 255 255]]]
    flatten lines data: 
     shape: (103230, 3)
     data: [[255 255 255]
     [255 255 255]
     [255 255 255]
     ...
     [255 255 255]
     [255 255 255]
     [255 255 255]]



    
![png](img/output_17_1.png)
    


## Définition du Kmeans

### Description de l'implémentation

#### Caractéristique
- Fonctionnement sur des dimensions quelconques
- Complexité temporelle de l'entrainementen $O(n*d*k*I)$
- La complexité en espace est donc en $O(n*d*k)$

> $n$ représente le nombre de données d'entrainements, $d$ la dimension du problème, $k$ le nombre de cluster à identifier, $I$ le nombre d'itération avant convergence

#### Structure de données
Cette version de KMeans stocke :
- les données d'entrainements
- l'ensemble des centroïds
- la matrice de distance
- un tableau permettant de faire l'association au cluster

#### Principe de l'entrainement
1. Initialisation du cluster avec des centroids aléatoires
2. Boucle tant que les centroids ne sont pas stables
    1. Calculer les centroides
    2. Mettre à jour la matrice de distance
    3. Mettre à jour l'association des points à leur cluster

#### Piste d'amélioration
- Choisir le type de distance à utiliser
- Ajouter une fonction de prédiction pour tester une valeur en dehors du jeu de test
- Pouvoir laisser le choix à l'utilisateur d'utiliser une version avec une empreinte mémoire plus faible (exemple : ne pas stocker la matrice de distance)
- Généraliser les fonctions d'affichages pour choisir les dimensions à afficher dans le cas où on est sur plus de 2 dimensions
- Mise en place d'heuristique pour favoriser une meilleure initialisation des centroïds

### L'implémentation de Kmeans


```python
class Kmeans():
    def __init__(self, x_train, k):
        self.k = k
        self.x_train = x_train
        self.centroids = np.zeros((self.k, self.x_train.shape[1]))
        self.dist_matrix = np.zeros((len(self.x_train), self.k))
        self.cluster = np.zeros(len(self.x_train))

    def init_centroids(self):
        """
        Choose random centroids from the data
        """
        inds = np.random.choice(len(self.x_train), self.k)
        self.centroids[:] = self.x_train[inds]

    def dist_calc(self):
        """
        Update the distance matrix
        """
        self.dist_matrix = scipy.spatial.distance.cdist(self.x_train, self.centroids)

    def agglomerate(self):
        """
        Update the association of point with the closest cluster
        """
        # Probablement une meilleure écriture plus Python (liste en comprehension ?, version numpy plus opti ?)
        for i, e in enumerate(self.dist_matrix):
            self.cluster[i] = np.argmin(e)

    def get_point_of_cluster(self, cluster_index):
        in_matrix = (self.cluster == cluster_index)
        return self.x_train[in_matrix]

    def plot_cluster(self):
        for ci in range(self.k):
            x_cluster = self.get_point_of_cluster(ci)
            plt.scatter(x_cluster[:,0], x_cluster[:,1])
            plt.scatter(self.centroids[ci][0], self.centroids[ci][1], marker='x', linewidths=10)
        plt.show()

    def new_centroids_calc(self):
        for ci in range(self.k):
            in_matrix = self.get_point_of_cluster(ci)
            if in_matrix.size != 0:
                self.centroids[ci] = np.mean(in_matrix, axis=0)

    def get_centroid_from_cluster(self, ci):
        return self.centroids[ci]

    def init_cluster(self):
        """
        Choose random centroid and calculate the first association
        """
        self.init_centroids()
        self.dist_calc()
        self.agglomerate()

    def fit(self, debug=False, ceil=0.05, fake_init=None):
        """
        Apply the full Kmean algorithm to fit with the data.
        @debug: bool - choose if the cluster is plot after each step
        """
        if fake_init is None:
            self.init_cluster()
        else:
            # Pas très propre - sert uniquement à piper les tests pour montrer des cas d'initialisations spécifique
            self.centroids = fake_init
            self.dist_calc()
            self.agglomerate()
            if debug: self.plot_cluster()
        old_centroids = np.zeros(self.centroids.shape)
        while not(np.all(abs(old_centroids-self.centroids) < ceil)):
            old_centroids = self.centroids.copy()
            self.new_centroids_calc()
            self.dist_calc()
            self.agglomerate()
            if debug: self.plot_cluster()
            

```

Dans le cas d'ume image, on s'assure de pouvoir réutiliser la classe ci dessus. Pour ce faire, il faut avoir les données en 2 dimensions et non 3. Pour ce faire, on détruit la dimension représentant les lignes de l'image pour n'avoir qu'une seule ligne. On prendra le soit de stocker la dimension initiale de l'image pour être en capacité de la reconstruire.

Une adaptation est nécessaire pour l'affichage avec un besoin d'affichage sur les 3 dimensions de couleurs. De la même façon, une méthode de construction d'image à partir des clusters est incluse.


```python
class Kmeans_image(Kmeans):
    def __init__(self, x_train, k):
        self.img_shape = x_train.shape # Store the shape to reconstruct the image after applying the clustering
        x_train = x_train.reshape(-1, 3) # Flatten the RGB into one line only
        super().__init__(x_train, k)

    def plot_cluster(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('Red')
        ax.set_ylabel('Green')
        ax.set_zlabel('Blue')
        for ci in range(self.k):
            x_cluster = self.get_point_of_cluster(ci)
            centroid = self.centroids[ci][0], self.centroids[ci][1], self.centroids[ci][2]
            # Colorier avec la couleur moyenne des clusters
            color = [np.array(centroid) / 255]
            ax.scatter(x_cluster[:,0], x_cluster[:,1], x_cluster[:,2], c=color)
            plt.scatter(*centroid, marker='x', linewidths=10, colorizer=color)
        plt.show()
        
    def construct_image_from_cluster(self):
        """
        Construct the image based on the color of the centroid of each cluster.
        """
        img_cluster = np.zeros((self.img_shape))
        for i in range(self.img_shape[0]):
            for j in range(self.img_shape[1]):
                ci = self.cluster[i*self.img_shape[1] + j]
                img_cluster[i,j] = self.get_centroid_from_cluster(int(ci))
        return img_cluster.astype(np.uint8)
        
        
```

### Tests successifs des méthodes - vocation à du débogage

#### Tests successifs des méthodes - Kmeans classique


```python
k = 3
cluster = Kmeans(random_generated_2_cluster, k)
cluster.init_centroids()
print(cluster.centroids)
```

    [[ 52.00236553  51.82994982]
     [ 86.41095648  96.86750397]
     [103.90219716  92.30580894]]



```python
cluster.dist_calc()
print(cluster.dist_matrix)
```

    [[ 1.31243827 57.8083531  67.07866446]
     [64.13864809 12.46318322  6.09897646]
     [ 1.62368143 56.56576589 65.29482153]
     [63.98482775 15.16848184  2.91128462]
     [ 2.91997797 59.40736194 68.20174117]
     [ 2.01207133 57.87861947 66.57023788]
     [ 4.51745006 59.24696846 67.36039715]
     [60.93212523  9.49302205  8.58365032]
     [ 0.         56.67744186 65.81707754]
     [62.8906703  13.91148096  4.21633423]
     [58.58782858  8.89643728  9.73802749]
     [67.75409133 17.07021361  3.63469602]
     [ 1.04153561 56.35827705 65.75850947]
     [59.13172924  6.93903531 11.18525373]
     [65.90717718 12.11407271  8.39727485]
     [ 2.36312446 58.97076439 68.18013945]
     [60.67513315  4.59482463 14.64453952]
     [ 0.81758048 56.65994182 66.0062229 ]
     [ 3.68394532 60.14600732 68.8680182 ]
     [ 3.0224605  55.2210438  65.07538311]
     [ 4.68871667 58.38562166 68.53913524]
     [59.12612098  6.56063338 11.53939757]
     [ 5.12484296 61.78310879 70.65893627]
     [61.63483293  9.94452827  8.15942163]
     [ 1.99118434 55.63482382 64.37255684]
     [ 2.99550126 59.63143521 68.5548007 ]
     [ 3.16261451 58.43106235 68.16511179]
     [53.96440183 10.88084477 12.47355229]
     [ 0.23766183 56.53970253 65.63469489]
     [58.57385736  2.87851266 15.35297379]
     [ 2.19073476 55.09493101 64.67128731]
     [ 0.94363771 57.15078961 66.48032453]
     [59.13827014 11.28651246  7.82925873]
     [ 3.64118873 57.60549918 67.5846021 ]
     [65.81707754 18.07629835  0.        ]
     [58.01554417 12.00531186  8.33768751]
     [59.99520415  6.37379329 11.7544116 ]
     [ 2.16937463 58.79448976 67.98538905]
     [58.90255946  9.06624464  9.46200835]
     [63.93609687  7.74130633 13.81282102]
     [ 1.63187185 57.68036543 67.10956835]
     [ 1.33716775 57.29386251 66.71102549]
     [62.66115576  6.27207746 14.72887206]
     [ 0.59610262 56.10058449 65.29733674]
     [ 1.63950436 55.35717988 64.78744111]
     [59.62523226  8.98014259  9.26132148]
     [ 2.18986599 58.15056962 66.8327615 ]
     [ 2.19408591 58.80804552 67.74893317]
     [61.29993507 11.76694944  6.43643078]
     [ 1.26767904 56.62886095 65.44690173]
     [ 1.06585641 57.35570265 66.26510109]
     [58.59799828  5.83990421 12.28884531]
     [ 5.43769078 58.92583407 69.19809114]
     [64.59641481 17.50735181  1.23357116]
     [63.90327213 10.91795767  8.00582914]
     [ 4.03359751 58.11631133 68.13642635]
     [61.88624493  6.24564252 13.20481332]
     [ 6.84613655 60.70697402 71.05081844]
     [61.83869455 15.84392648  3.98117063]
     [ 2.96405467 59.30702501 68.02050441]
     [ 1.05637169 57.70380572 66.87333345]
     [57.29177274  6.46843022 12.23268176]
     [62.6989077  12.7097234   5.36657522]
     [ 2.05232097 58.27952909 67.68769909]
     [67.97897447 15.8142414   5.62123186]
     [56.62821414  6.56945727 12.54546405]
     [ 4.49939855 60.41809717 68.81432444]
     [67.18886448 15.81766254  4.48412539]
     [ 1.81793251 58.49477152 67.58845689]
     [ 2.35219233 58.33624732 67.00101494]
     [ 2.26347697 55.42979758 65.0874725 ]
     [60.13510844 11.533535    7.08520527]
     [ 2.4997546  55.4950456  65.22848902]
     [ 2.91870517 58.83106919 67.40472881]
     [55.13185995  5.99853525 24.05850001]
     [57.91460444 11.18373472  8.77326608]
     [ 1.13218827 56.87740067 65.72688779]
     [ 4.33314024 59.63489301 67.88041894]
     [61.84771563 12.35295011  5.79878096]
     [48.59526314 13.02462025 17.5651427 ]
     [ 4.32358285 60.0243691  68.3703017 ]
     [71.47818752 16.26102861 11.74536861]
     [56.67744186  0.         18.07629835]
     [ 1.93919157 54.84876846 64.21428217]
     [63.24440018 11.72610888  6.56214107]
     [ 2.06210961 57.67703126 67.23409907]
     [63.47725996  9.21051918 10.11669537]
     [ 4.43080421 61.03080675 70.23999091]
     [59.57048769  9.38485606  8.94264692]
     [59.47361337 11.38491219  7.55734486]
     [ 3.13907859 57.48081352 67.34642372]
     [58.7382762   9.87390933  8.9643569 ]
     [ 1.78168606 58.10850134 67.46744728]
     [61.88318589  5.47016942 18.51290022]
     [58.50295434 10.0256491   9.00341767]
     [57.24158775  8.96807034 10.49968113]
     [66.34910804 12.27873362  8.7725358 ]
     [61.2228116   4.7218097  15.53600112]
     [57.46585923 11.68330325  8.93009426]
     [ 3.77878918 56.56278955 64.72883951]]



```python
cluster.agglomerate()
print(cluster.cluster)
```

    [0. 2. 0. 2. 0. 0. 0. 2. 0. 2. 1. 2. 0. 1. 2. 0. 1. 0. 0. 0. 0. 1. 0. 2.
     0. 0. 0. 1. 0. 1. 0. 0. 2. 0. 2. 2. 1. 0. 1. 1. 0. 0. 1. 0. 0. 1. 0. 0.
     2. 0. 0. 1. 0. 2. 2. 0. 1. 0. 2. 0. 0. 1. 2. 0. 2. 1. 0. 2. 0. 0. 0. 2.
     0. 0. 1. 2. 0. 0. 2. 1. 0. 2. 1. 0. 2. 0. 1. 0. 2. 2. 0. 2. 0. 1. 2. 1.
     2. 1. 2. 0.]



```python
cluster.plot_cluster()
```


    
![png](img/output_30_0.png)
    



```python
print(cluster.centroids)
cluster.new_centroids_calc()
print(cluster.centroids)
```

    [[ 52.00236553  51.82994982]
     [ 86.41095648  96.86750397]
     [103.90219716  92.30580894]]
    [[51.05835347 51.09689391]
     [91.30774957 95.37917818]
     [98.41490618 93.85563806]]


#### Tests successifs des méthodes - Kmeans image


```python
k = 2
cluster_image = Kmeans_image(data_from_image, k)
cluster_image.init_centroids()
print(cluster_image.centroids)
```

    [[249. 246. 241.]
     [255. 255. 255.]]



```python
cluster_image.dist_calc()
print(cluster_image.dist_matrix)
```

    [[17.69180601  0.        ]
     [17.69180601  0.        ]
     [17.69180601  0.        ]
     ...
     [17.69180601  0.        ]
     [17.69180601  0.        ]
     [17.69180601  0.        ]]



```python
cluster_image.agglomerate()
print(cluster_image.cluster)
```

    [1. 1. 1. ... 1. 1. 1.]



```python
cluster_image.plot_cluster()
```


    
![png](img/output_36_0.png)
    



```python
print(cluster_image.centroids)
cluster_image.new_centroids_calc()
print(cluster_image.centroids)
```

    [[249. 246. 241.]
     [255. 255. 255.]]
    [[172.50948895 121.10660221  65.38254144]
     [254.71955887 254.74842686 254.54807006]]


### Entrainements complets sur les jeux de données 

#### Recherche de 2 clusters parmis 2 clusters - problème à 2 dimensions


```python
k = 2
cluster_to_fit = Kmeans(random_generated_2_cluster, k)
cluster_to_fit.fit(debug=True)
```


    
![png](img/output_40_0.png)
    



    
![png](img/output_40_1.png)
    



```python
k = 2
cluster_to_fit = Kmeans(random_generated_2_cluster, k)
fake_centroids = np.array([[50.7837158, 51.34272601], [ 51.1937917, 49.02415587]]) # Pick two close point from one of the distribution
cluster_to_fit.fit(debug=True, fake_init=fake_centroids)
```


    
![png](img/output_41_0.png)
    



    
![png](img/output_41_1.png)
    



    
![png](img/output_41_2.png)
    



    
![png](img/output_41_3.png)
    


Dans le premier cas, on remarque que l'on a une initialisation favorable à la convergence. La convergence est alors très rapide et les deux distibutions sont correctements identifiés.

Dans le second cas où l'on force une "mauvaise" initialisation (on force les deux centroïdes à être dans le même cluster). On arrive tout de même à retrouver le résultat attendu.

#### Recherche de 2 clusters parmis 5 clusters - problème à 2 dimensions


```python
k = 2
cluster_to_fit = Kmeans(random_generated_5_cluster, k)
cluster_to_fit.fit(debug=True)
```


    
![png](img/output_44_0.png)
    



    
![png](img/output_44_1.png)
    



```python
k = 2
cluster_to_fit = Kmeans(random_generated_5_cluster, k)
cluster_to_fit.fit(debug=True)
```


    
![png](img/output_45_0.png)
    



    
![png](img/output_45_1.png)
    



    
![png](img/output_45_2.png)
    



    
![png](img/output_45_3.png)
    



    
![png](img/output_45_4.png)
    



    
![png](img/output_45_5.png)
    



    
![png](img/output_45_6.png)
    



    
![png](img/output_45_7.png)
    



    
![png](img/output_45_8.png)
    



    
![png](img/output_45_9.png)
    



    
![png](img/output_45_10.png)
    



    
![png](img/output_45_11.png)
    



    
![png](img/output_45_12.png)
    



    
![png](img/output_45_13.png)
    


Dans ce cas à 5 distributions (2 étant entremêlés), on voit que les distributions sont individuellement dans un même cluster. Ce qui est un résultat souhaitable.

Pour autant, si l'on compare les deux exécutions, on voit que l'initialisation change le regroupement des distributions.

#### Recherche des 6 clusters parmis le jeu de données imposé


```python
k = 6
cluster_data_from_file = Kmeans(data_from_file, k)
cluster_data_from_file.fit(debug=True)
```


    
![png](img/output_48_0.png)
    



    
![png](img/output_48_1.png)
    



    
![png](img/output_48_2.png)
    



```python
k = 6
cluster_data_from_file = Kmeans(data_from_file, k)
cluster_data_from_file.fit(debug=True)
```


    
![png](img/output_49_0.png)
    



    
![png](img/output_49_1.png)
    



    
![png](img/output_49_2.png)
    


Cette exemple nous montre bien la sensibilité de l'initialisation. A partir des mêmes données d'entrées, les clusters résultant sont différents.

On remarque aussi du fait de la proximité relative des clusters, quelques difficultés sont rencontrés pour identifier correctement les clusters.

#### Partionnement d'une image en 2


```python
k = 2
cluster_image_fit = Kmeans_image(data_from_image, k)
cluster_image_fit.fit(debug=False)

img_arr_reconstruct = cluster_image_fit.construct_image_from_cluster()
print(img_arr_reconstruct)
plt.imshow(img_arr_reconstruct)
plt.show()
skimage.io.imsave("out/reconstructed_fruit.jpg", img_arr_reconstruct)
```

    [[[250 247 240]
      [250 247 240]
      [250 247 240]
      ...
      [250 247 240]
      [250 247 240]
      [250 247 240]]
    
     [[250 247 240]
      [250 247 240]
      [250 247 240]
      ...
      [250 247 240]
      [250 247 240]
      [250 247 240]]
    
     [[250 247 240]
      [250 247 240]
      [250 247 240]
      ...
      [250 247 240]
      [250 247 240]
      [250 247 240]]
    
     ...
    
     [[250 247 240]
      [250 247 240]
      [250 247 240]
      ...
      [250 247 240]
      [250 247 240]
      [250 247 240]]
    
     [[250 247 240]
      [250 247 240]
      [250 247 240]
      ...
      [250 247 240]
      [250 247 240]
      [250 247 240]]
    
     [[250 247 240]
      [250 247 240]
      [250 247 240]
      ...
      [250 247 240]
      [250 247 240]
      [250 247 240]]]



    
![png](img/output_52_1.png)
    


Appliqué à une image, l'algorithme de KMeans peut aider à différencier des contours. Pour autant, quelques artefacts persistent comme sur les objets ayant une couleurs trop similaires au fond comme l'ail au milieu ou le citron vert.

L'un des points notables, est la stabilité des résultats malgré une initialisation aléatoire.


```python
k = 3
cluster_3_image_fit = Kmeans_image(data_from_image, k)
cluster_3_image_fit.fit(debug=False, ceil=1)

cluster_3_image_reconstruct = cluster_3_image_fit.construct_image_from_cluster()
plt.imshow(cluster_3_image_reconstruct)
plt.show()
```


    
![png](img/output_54_0.png)
    



```python
k = 4
cluster_4_image_fit = Kmeans_image(data_from_image, k)
cluster_4_image_fit.fit(debug=False, ceil=2)

cluster_4_image_reconstruct = cluster_4_image_fit.construct_image_from_cluster()
plt.imshow(cluster_4_image_reconstruct)
plt.show()
```


    
![png](img/output_55_0.png)
    

