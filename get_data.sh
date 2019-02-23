#!/usr/bin/env bash

mkdir data
git clone https://github.com/DavidGrangier/wikipedia-biography-dataset.git
cd wikipedia-biography-dataset/
cat wikipedia-biography-dataset.z?? > tmp.zip
unzip tmp.zip
rm tmp.zip
mv wikipedia-biography-dataset data/original_data
rm -rf wikipedia-biography-dataset