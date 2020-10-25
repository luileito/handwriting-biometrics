# Human or Machine?

This repository allows you to build computational models
that can tell human and machine-generated handwriting apart.

## Datasets

You can train the models from scratch with the following datasets:

- **$1-GDS** (5280 unistroke gestures, 16 classes):
[human data](https://luis.leiva.name/g3/datasets/csv-1dollar-human.zip)
and [synthetic data](https://luis.leiva.name/g3/datasets/csv-1dollar-synth-best.zip).

- **$N-MMG** (9600 unistroke and multistroke gestures, 16 classes):
[human data](https://luis.leiva.name/g3/datasets/csv-ndollar-human.zip)
and [synthetic data](https://luis.leiva.name/g3/datasets/csv-ndollar-synth-best.zip).

- **Chars74k** (3410 unistroke and multistroke gestures, 62 classes):
[human data](https://luis.leiva.name/g3/datasets/csv-chars74k-human.zip)
and [synthetic data](https://luis.leiva.name/g3/datasets/csv-chars74k-synth-best.zip).

You can try other datasets as long as you follow the [expected CSV format](https://luis.leiva.name/g3/#datasets).

## Model training

This is how we trained our GRU classifier over the $1-GDS dataset:
```sh
~$ python3 main.py --human_dir csv-1dollar-human --synth_dir csv-1dollar-synth-best \
  --model_type gru --epochs 400 --patience 40 --batch_size 32 --activation tanh
```

There are many CLI options you might want to specify,
such as `--verbose 1` (to see more output info) or `--out_dir somedir` (to set the output directory).

To see all the available CLI options, run `python3 main.py -h`.

## Model evaluation

If you already trained a model, you can evaluate it this way:
```sh
~$ python3 main.py --human_dir csv-1dollar-human --synth_dir csv-1dollar-synth-best \
  --eval_model path/to/model.h5 --model_type gru
```

Again, run `python3 main.py -h` to see all the available CLI options.

## Paper

A preprint of our ICPR paper is publicly available:
https://arxiv.org/abs/2001.07803

## Citation

Please cite us using the following reference:

- L. A. Leiva, M. Diaz, M. A. Ferrer, R. Plamondon.
**Human or Machine? It Is Not What You Write, But How You Write It.**
*Proc. ICPR, 2020.*

```bib
@InProceedings{Leiva20_biometrics,
  author    = {Luis A. Leiva and Moises Diaz and Miguel A. Ferrer and Réjean Plamondon},
  title     = {Human or Machine? It Is Not What You Write, But How You Write It},
  booktitle = {Proc. ICPR},
  year      = {2020},
}
```
