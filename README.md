\# K-Deep-Simplex-KDS-algorithm
===============================

This is an algorithm for dictionary learning. Formally, given a set of n data points that lie in a d-dimensional space, KDS learns sparse representation coefficients 
and m dictionary atoms such that each data point is represented a convex combination of local atoms. The optimization over the atoms and sparse coefficients can be solved using alternating minimization. KDS employs algorithm unrolling to design a structured neural network that solves the problem. 

\# Baseline algorithms
======================

In our manuscript, we also compare our paper with other baseline algorithms. The baseline algorithms are noted below.

1. **SMCE**: SMCE is an abbreviation for sparse manifold clustering and embedding. The algorithm was proposed in the paper referenced below:
>Elhamifar, E., & Vidal, R. (2011), Sparse manifold clustering and embedding. Advances in neural information processing systems, 24, 55-63.
2. **LLL**: LLL is an abbreviation for Landmark based LLE algorithm. The algorithm was proposed in the paper referenced below:
> Vladymyrov, Max, and Miguel Á. Carreira-Perpinán. "Locally linear landmarks % for large-scale manifold learning." Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2013, Prague, Czech Republic, September 23-27, 2013, Proceedings, Part III 13. Springer Berlin Heidelberg, 2013.
3. **ESC**: ESC is an abbreviation for exemplar based subspace clustering. The algorithm was proposed in the paper referenced below:
>You, C., Li, C., Robinson, D. P., & Vidal, R. (2018). Scalable exemplar-based subspace clustering on class-imbalanced data. In Proceedings of the European Conference on Computer Vision (ECCV) (pp. 67-83).

 

MATLAB files description
------------------------

`demo_globalrecon.m`: This is the main file which loads different kinds of data
file from the data folder and reads the (x,y,z) coordinates of the data. Using
these coordinates, the full distance matrix for the data is constructed. Then,
some entries of the distance matrix are selected uniformly at random. Once these
information and some parameters for the main algorithm are set, the script calls
the main algorithms.

`alternating_completion.m`: This is the main algorithm for the Euclidean
Distance Geometry Problem for the case of exact partial information.

`alternating_completion_noisy.m`: This is the main algorithm for the Euclidean
Distance Geometry Problem for the case of noisy partial information. The noise
is assumed to be additive Gaussian.

`BBGradient.m`: This implements the BB gradient method coupled with nonmonotone
line search.

`ReadOFF.m, ViewMesh.m, ViewPC.m`: These scripts are used to read different
types of the data file and generate points, mesh. They are also used in viewing
the outputs of the main algorithms.

List of data files tested
-------------------------

-   `1k.off`: Sphere with 1002 points

-   `cow.off`: Cow with 2601 points

-   `ptswiss.mat`: Swiss roll data with 2048 points

-   `UScities.mat`: Data of 2920 US cities

The first and second data sets are taken from
[here](http://visionair.ge.imati.cnr.it/ontologies/shapes/search.jsp). The third
data was obtained by simply plotting the parametric equations of a Swiss roll.
The last data uses Latitude and Longitude of US cities, in different zip codes,
to generate the point coordinates.

Instructions
------------

The starting script is `demo_globalrecon.m`. Choose a data, sampling rate, set
algorithm parameters and call either the exact or noisy solver. \#\# References

Abiy Tasissa and Rongjie Lai, "Exact Reconstruction of Euclidean Distance
Geometry Problem Using Low-rank Matrix Completion," in IEEE Transactions on
Information Theory, 2018. doi: 10.1109/TIT.2018.2881749
[URL](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8537996&isnumber=4667673)

Feedback
--------

Email your feedback to Abiy Tasissa.
