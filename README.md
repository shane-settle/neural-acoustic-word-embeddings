Neural Acoustic Word Embeddings for Switchboard
===============================================

Overview:
---------

This is a recipe for learning neural acoustic word embeddings for a subset of Switchboard. The models are explained in greater detail in [Settle & Livescu, 2016](https://arxiv.org/abs/1611.02550) as well as [Settle et al., 2017](https://arxiv.org/abs/1706.03818):

- S. Settle and K. Livescu, "Discriminative Acoustic Word Embeddings: Recurrent Neural Network-Based Approaches," in Proc. SLT, 2016.
- S. Settle, K. Levin, H. Kamper, and K. Livescu, "Query-by-Example Search with Discriminative Neural Acoustic Word Embeddings," in Proc. Interspeech, 2017.


Contents:
---------

code/
- python code to create, run, and save the model

kaldi/
- modification of Kaldi's swbd/s5c used to extract features and set up data as required
- word/segment list used in prior work [Kamper et al., 2015](http://arxiv.org/abs/1510.01032) can also be found under kaldi/data/kamperh

partitions/
- train/dev/test partitioning into Switchboard conversation sides (these are consistent with prior work, and extracted from the aforementioned word/segment list)

Steps:
------

1. Ensure access to installed dependencies.
    - Python 3.6
    - Tensorflow 1.5 (and numpy/scipy)
    - [kaldi](https://github.com/kaldi-asr/kaldi)
    - [kaldi-io-for-python](https://github.com/vesis84/kaldi-io-for-python)

2. Clone repo.

3. Check that $KALDI\_ROOT variable points to the location of installed/compiled kaldi. This can be set in your ~/.bashrc or in kaldi/path.sh.

3. Update kaldi/run.sh:
    - set $swbd variable to your local switchboard datapath
    - set $nj to number of desired jobs (default=8)
    - set $stage to desired stage in feature creation (default=1)
    - set $min\_word\_length to desired minimum length character sequence allowed for included words (default=6)
    - set $min\_audio\_duration to minimum audio duration (in frames) allowed for included audio (default=50)
    - set $min\_train\_occurrence\_count to limit how common training words must have been (default=2, note: this must be >= 2 or siamese training will not work)

4. Navigate to kaldi directory and run "./run.sh". Now you should have the desired features.

5. Navigate to code directory and run "python main.py". This will train, evaluate, and save models.
