#!/bin/bash

# Create data directory so we can store data files
mkdir -p src/data

# Navigate to data directory
cd src/data

echo "Downloading MovieLens Dataset...."
curl -O https://files.grouplens.org/datasets/movielens/ml-latest-small.zip

echo "Extracting Data...."
unzip ml-latest-small.zip

# Move files to data directory
mv ml-latest-small/movies.csv ml-latest-small/ratings.csv .
rm -rf ml-latest-small ml-latest-small.zip

echo "Data setup is complete"