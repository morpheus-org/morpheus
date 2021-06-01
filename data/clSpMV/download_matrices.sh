#!/bin/bash

MATRICES=("https://suitesparse-collection-website.herokuapp.com/MM/Williams/pdb1HYS.tar.gz"
          "https://suitesparse-collection-website.herokuapp.com/MM/Williams/consph.tar.gz"
          "https://suitesparse-collection-website.herokuapp.com/MM/Williams/cant.tar.gz"
          "https://suitesparse-collection-website.herokuapp.com/MM/Boeing/pwtk.tar.gz"
          "https://suitesparse-collection-website.herokuapp.com/MM/Bova/rma10.tar.gz"
          # "https://suitesparse-collection-website.herokuapp.com/MM/QCD/conf5_4-8x8-05.tar.gz"
          "https://suitesparse-collection-website.herokuapp.com/MM/DNVS/shipsec1.tar.gz"
          "https://suitesparse-collection-website.herokuapp.com/MM/Williams/mac_econ_fwd500.tar.gz"
          "https://suitesparse-collection-website.herokuapp.com/MM/Williams/mc2depi.tar.gz"
          "https://suitesparse-collection-website.herokuapp.com/MM/Williams/cop20k_A.tar.gz"
          "https://suitesparse-collection-website.herokuapp.com/MM/Hamm/scircuit.tar.gz"
          # "https://suitesparse-collection-website.herokuapp.com/MM/Williams/webbase-1M.tar.gz"
          # "https://suitesparse-collection-website.herokuapp.com/MM/Mittelmann/rail4284.tar.gz"
          )

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
