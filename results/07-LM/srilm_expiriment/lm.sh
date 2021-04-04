#!/bin/bash

ngram-count -text train.txt -order 3 -write train-3gram.count
#ngram-count -read train-3gram.count -order 3 -lm train-3gram.arpa -interpolate -kndiscount
ngram-count -read train-3gram.count -order 3 -lm train-3gram.arpa -interpolate -wittenbil
ngram -ppl train.txt -order 3 -lm train-3gram.arpa
ngram -ppl dev.txt -order 3 -lm train-3gram.arpa
ngram -ppl test.txt -order 3 -lm train-3gram.arpa
