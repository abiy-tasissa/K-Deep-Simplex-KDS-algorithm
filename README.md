\# K-Deep-Simplex-KDS-algorithm
===============================

This is an algorithm for dictionary learning. Formally, given a set of n data points that lie in a d-dimensional space, KDS learns sparse representation coefficients and m dictionary atoms such that each data point is represented a convex combination of local atoms. The optimization over the atoms and sparse coefficients can be solved using alternating minimization. KDS employs algorithm unrolling to design a structured neural network that solves the problem. 

\# Baseline algorithms
======================

In our manuscript, we also compare our paper with other baseline algorithms. The baseline algorithms are noted below.

1. **SMCE**: SMCE is an abbreviation for sparse manifold clustering and embedding. The algorithm was proposed in the paper referenced below:
>Elhamifar, E., & Vidal, R. (2011), Sparse manifold clustering and embedding. Advances in neural information processing systems, 24, 55-63.
2. **LLL**: LLL is an abbreviation for Landmark based LLE algorithm. The algorithm was proposed in the paper referenced below:
> Vladymyrov, Max, and Miguel Á. Carreira-Perpinán. "Locally linear landmarks % for large-scale manifold learning." Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2013, Prague, Czech Republic, September 23-27, 2013, Proceedings, Part III 13. Springer Berlin Heidelberg, 2013.
3. **ESC**: ESC is an abbreviation for exemplar based subspace clustering. The algorithm was proposed in the paper referenced below:
>You, C., Li, C., Robinson, D. P., & Vidal, R. (2018). Scalable exemplar-based subspace clustering on class-imbalanced data. In Proceedings of the European Conference on Computer Vision (ECCV) (pp. 67-83).

All the baseline algorithms are in MATLAB. We used implementation of the original authors. 

Python files description
------------------------
The KDS folder contains:
* a PyTorch implementation of KDS in `src/model.py`
* a script `src/clustering_experiment.py` that uses KDS to cluster various real-world data sets
* a script `src/synthetic_experiment.py` that evaluates the performance of KDS as a function of dictionary size and cluster separation
* a data folder `data` that contains the Salinas-A hyperspectral dataset and the Yale B dataset.
  - The Salinas-A dataset is a single aerial-view hyperpspectral image
    of the Salinas valley in California with 224 bands and 6 regions corresponding to different crops. The Salinas-A dataset is taken from
    [here](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Salinas-A_scene).
  - The cropped version of the Extended Yale Face Dataset B consists of 192 × 168 grayscale images of 39 different faces under varying illumination conditions. The 
   dataset is based on the reference below.
     >Lee, Kuang-Chih, Jeffrey Ho, and David J. Kriegman. "Acquiring linear subspaces for face recognition under variable lighting." IEEE Transactions on pattern 
     analysis and machine intelligence 27.5 (2005): 684-698.

### Dependencies
* scipy: 1.11.2
* PyTorch: 2.0.1
* numpy: 1.24.3
* Python: 3.11.4
* Sacred: 0.8.4
* tqdm: 4.66.1
* scikit-learn: 1.3.0
* keras: 2.13.1
* tensorflow: 2.13.0
* torchvision: 0.15.2


### Usage

To run either experiment, navigate to the `src` directory.

**Example usage (clustering experiment):**

`python clustering_experiment.py -F ../results/moons with moons_default`

Other provided parameter settings include `salinas_default`, `yale2_default`, `yale3_default`, and `mnist_default`. This will run the KDS code with optimal deafult parameters reported in our manuscript.

**Example usage (synthetic experiment):**

`python synthetic_experiment.py -F ../results/synthetic`

MATLAB files description
------------------------

`demo_smce_experiments.m`: This is a script which loads different kinds of datasets and calls the SMCE algorithm. 

`demo_lll.m`: This is a script which loads different kinds of datasets and calls the LLL algorithm. 

`demo_esc.m`: This is a script which loads different kinds of datasets and calls the LLL algorithm. 

`mnist_clustering.m`: This loads up the optimized sparse coefficients of SMCE for MNIST-5 dataset SMCE under different parameter choices and
                      computes clustering accuracy. This is included since running and clustering SMCE on MNIST-5 could be computationally
                      expensive. The parameter choices are: (1) lambda = 1, (2) lambda = 10, (3) lambda = 100 and (4) lambda = 200. Lambda is a regularization
                      term that controls locality regularization in SMCE. We also consider three types of pre-processing: (1) Scaling to
                      [0,1], (2) Standardizing and (3) Normalizing. 
                      Example: `mnist_accuracy_3_4.mat` loads the sparse coefficient matrix with pre-processing set to (3) and lambda set to 400. 
`KDS_Complexity_Plots.m` and `SMCE_Complexity_Plots.m`: These scripts will reproduce the complexity plots in the supplementary. 

`SpectralClustering_fast.m`: A faster implementation of spectral clustering. 

`load_salinas.m` and `load_yale.m`: Scripts that load the Salinas-A and Yale B datasets. 

You can download all the necessary additional data from [here](https://www.dropbox.com/scl/fo/im6dmydz5fqgykoe1u6es/h?rlkey=4jlpzn965rt7yjefznrjqvci1&dl=0). To set up the SPAMS package, needed to run demo_esc, refer to the instructions
[here] (https://thoth.inrialpes.fr/people/mairal/spams/). 

Citation
------------

If you use our code or find our paper useful and relevant, we would appreciate if you cite our paper. 
>Tankala, Pranay, et al. "K-deep simplex: Deep manifold learning via local dictionaries.
arXiv preprint arXiv:2012.02134 (2020).

Feedback
--------

If you have any questions about the code or feedback, email Abiy Tasissa.
