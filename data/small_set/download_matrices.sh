#!/bin/bash

MATRICES=("https://suitesparse-collection-website.herokuapp.com/MM/Boeing/pcrystk02.tar.gz"
          "https://suitesparse-collection-website.herokuapp.com/MM/Castrillon/denormal.tar.gz"
          "https://suitesparse-collection-website.herokuapp.com/MM/Bai/cryg10000.tar.gz"
          "https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/apache1.tar.gz"
          "https://suitesparse-collection-website.herokuapp.com/MM/AG-Monien/whitaker3_dual.tar.gz"
          "https://suitesparse-collection-website.herokuapp.com/MM/JGD_Homology/ch7-9-b3.tar.gz"
          "https://suitesparse-collection-website.herokuapp.com/MM/JGD_Homology/shar_te2-b2.tar.gz"
          "https://suitesparse-collection-website.herokuapp.com/MM/Chen/pkustk14.tar.gz"
          "https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/crankseg_2.tar.gz"
          "https://suitesparse-collection-website.herokuapp.com/MM/PARSEC/Ga3As3H12.tar.gz"
          "https://suitesparse-collection-website.herokuapp.com/MM/Fluorem/HV15R.tar.gz"
          "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/europe_osm.tar.gz"
          "https://suitesparse-collection-website.herokuapp.com/MM/JGD_Homology/D6-6.tar.gz"
          "https://suitesparse-collection-website.herokuapp.com/MM/Pajek/dictionary28.tar.gz"
          "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/roadNet-CA.tar.gz")

for tarfile in "${MATRICES[@]}"
do
  wget $tarfile
  file=$(basename $tarfile)
  tar xf $file && rm -rf $file
done

