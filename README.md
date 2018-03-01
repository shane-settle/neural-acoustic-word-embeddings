This repository contains code for replicating work from:

Discriminative Acoustic Word Embeddings: Recurrent Neural Network-Based Approaches (SLT 2016)

as well as

Query-by-Example Search with Discriminative Neural Acoustic Word Embeddings (Interspeech 2017)

code/
- python code (Tensorflow) to create, run, and save the model
- Python 3.6 with Tensorflow 1.5

kaldi/
- modification of Kaldi's swbd/s5c used to extract features and set up data as required
- word/segment list used in prior work (H. Kamper, W. Wang, and K. Livescu, "Deep convolutional acoustic word embeddings using word-pair side information," in Proc. ICASSP, 2016.) can also be found under kaldi/data/kamperh

partitions/
- train/dev/test partitioning into Switchboard conversation sides
