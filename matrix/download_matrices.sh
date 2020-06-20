#!/bin/bash

# Structural analysis matrices
STRUCTURAL=("https://suitesparse-collection-website.herokuapp.com/MM/Williams/cant.tar.gz"
            "https://suitesparse-collection-website.herokuapp.com/MM/Simon/olafu.tar.gz"
            "https://suitesparse-collection-website.herokuapp.com/MM/Schenk_AFE/af_1_k101.tar.gz"
            "https://suitesparse-collection-website.herokuapp.com/MM/Janna/Cube_Coup_dt0.tar.gz"
            "https://suitesparse-collection-website.herokuapp.com/MM/Janna/ML_Laplace.tar.gz"
            "https://suitesparse-collection-website.herokuapp.com/MM/HB/bcsstk17.tar.gz")

# Macroeconomic model
MACROECONOMIC=("https://suitesparse-collection-website.herokuapp.com/MM/Williams/mac_econ_fwd500.tar.gz")

# Electromagnetism
ELECTROMAGNETISM=("https://suitesparse-collection-website.herokuapp.com/MM/Bai/mhd4800a.tar.gz"
                  "https://suitesparse-collection-website.herokuapp.com/MM/Williams/cop20k_A.tar.gz")

# CFD Problems
CFD=("https://suitesparse-collection-website.herokuapp.com/MM/Simon/raefsky2.tar.gz"
     "https://suitesparse-collection-website.herokuapp.com/MM/Bai/af23560.tar.gz"
     "https://suitesparse-collection-website.herokuapp.com/MM/Norris/lung2.tar.gz"
     "https://suitesparse-collection-website.herokuapp.com/MM/Janna/StocF-1465.tar.gz"
     "https://suitesparse-collection-website.herokuapp.com/MM/Fluorem/PR02R.tar.gz"
     "https://suitesparse-collection-website.herokuapp.com/MM/Fluorem/RM07R.tar.gz")

# Thermal Diffusion Problems
THERMAL=("https://suitesparse-collection-website.herokuapp.com/MM/Botonakis/FEM_3D_thermal1.tar.gz"
         "https://suitesparse-collection-website.herokuapp.com/MM/Botonakis/FEM_3D_thermal2.tar.gz"
         "https://suitesparse-collection-website.herokuapp.com/MM/Schmid/thermal1.tar.gz"
         "https://suitesparse-collection-website.herokuapp.com/MM/Schmid/thermal2.tar.gz"
         "https://suitesparse-collection-website.herokuapp.com/MM/Botonakis/thermomech_TK.tar.gz"
         "https://suitesparse-collection-website.herokuapp.com/MM/Botonakis/thermomech_dK.tar.gz")

# Nonlinear Optimisation
NONLINEAR=("https://suitesparse-collection-website.herokuapp.com/MM/Schenk/nlpkkt80.tar.gz"
           "https://suitesparse-collection-website.herokuapp.com/MM/Schenk/nlpkkt120.tar.gz")

# Graph Matrices - web connectivity, circuit simulation
GRAPH=("https://suitesparse-collection-website.herokuapp.com/MM/Williams/webbase-1M.tar.gz"
       "https://suitesparse-collection-website.herokuapp.com/MM/IBM_EDA/dc1.tar.gz"
       "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/amazon0302.tar.gz"
       "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/amazon0312.tar.gz"
       "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/roadNet-PA.tar.gz"
       "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/roadNet-CA.tar.gz"
       "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/web-Google.tar.gz"
       "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/wiki-Talk.tar.gz")


MATRICES=(${STRUCTURAL[@]} ${MACROECONOMIC[@]} ${ELECTROMAGNETISM[@]} ${CFD[@]}
          ${THERMAL[@]} ${NONLINEAR[@]} ${GRAPH[@]})

for matrix in "${MATRICES[@]}"
do
  wget $matrix
done

# untar files and delete compressed files
for i in *.tar.gz
do
  pushd `dirname $i`
  tar xf `basename $i` && rm `basename $i`
  popd
done
