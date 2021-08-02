# End-to-End Coreference For German

This repository contains the code for [our 2021 publication](#TODO), adapted from [an existing implementation](https://github.com/lxucs/coref-hoi).

You can use our pre-trained models which are available the [github releases](../../releases).
The simplest way to use the models for inference is via torchserve,
download the models and refer to the [torchserve](#torchserve) section of this README.

If you base your work on our models, incremental code or the German dataset processing, please cite our paper:

```bibtex
@inproceedings{
  title = {Neural End-to-end Coreference Resolution for {G}erman in Different Domains},
  author = {Schröder, Fynn and Hatzel, Hans Ole and Biemann, Chris},
  year = {2021},
  booktitle = {Proceedings of the 17th Conference on Natural Language Processing},
  address = {Düsseldorf, Germany}
}
```

Most of the code in this repository is not language specific and can easily be adapted to many other languages.
mBERT models performed on par with some older German-specific models in our experiments, so even if you are dealing with lower resource language it *may* be possible to train a decent model.

## Architecture

The basic end-to-end coreference model [(as found in the original implementation)](https://github.com/lxucs/coref-hoi) is a PyTorch re-implementation based on the TensorFlow model following similar preprocessing (see this [repository](https://github.com/mandarjoshi90/coref)).

The code was extended to handle incremental coreference resolution and separate mention pre-training.

**Files**:
* [run.py](run.py): training and evaluation
* [run_mentions.py](run.py): training and evaluation of mentions
* [model.py](model.py): the coreference model
* [higher_order.py](higher_order.py): higher-order inference modules
* [analyze.py](analyze.py): result analysis
* [preprocess.py](preprocess.py): converting CoNLL files to examples
* [tensorize.py](tensorize.py): tensorizing example
* [conll.py](conll.py), [metrics.py](metrics.py): same CoNLL-related files from the [repository](https://github.com/mandarjoshi90/coref)
* [experiments.conf](experiments.conf): different model configurations
* [split_droc.py](split_droc.py): create train/dev/test splits for the German literature "DROC" dataset
* [split_gerdracor.py](split_gerdracor.py): create train/dev/test splits for the German Drama dataset GerDraCor
* [split_tuebadz_10.py](split_tuebadz_10.py): create train/dev/test splits for the TüBa-D/Z dataset version 10
* [split_tuebadz_11.py](split_tuebadz_11.py): create train/dev/test splits for the TüBa-D/Z dataset version 11


## Basic Setup
Set up environment and data for training and evaluation:
* Install Python3 dependencies: `pip install -r requirements.txt`
* All data and config files are placed relative to the: `base_dir = /path/to/project` in [local.conf](local.conf) so change it to point to the root of this repo
* All splits created using the `split_*` Python scripts will need to be processed using `preprocess.py` to be used as training input for the model, for example, to split the DROC dataset run:
    - `python split_droc.py --type-system-xml /path/to/DROC-Release/droc/src/main/resources/CorefTypeSystem.xml /path/to/DROC-Release/droc/DROC-xmi data/german.droc_gold_conll`
    - `python preprocess.py --input_dir data/droc --output_dir data/droc --seg_len 512 --language german --tokenizer_name german-nlp-group/electra-base-german-uncased --input_suffix droc_gold_conll --input_format conll-2012`


## Evaluation
If you want to use the official evaluator, download and unzip [conll 2012 scorer](https://cs.emory.edu/~lxu85/conll-2012.zip) under this directory.

Evaluate a model on the dev/test set:
* Download the corresponding model directory and unzip it under `data_dir`
* `python evaluate.py [config] [model_id] [gpu_id]`
    * e.g. Attended Antecedent:`python evaluate.py train_spanbert_large_ml0_d2 May08_12-38-29_58000 0`

## Training

`python run.py [config] [gpu_id]`

* [config] can be any **configuration** in [experiments.conf](experiments.conf)
* Log file will be saved at `your_data_dir/[config]/log_XXX.txt`
* Models will be saved at `your_data_dir/[config]/model_XXX.bin`
* Tensorboard is available at `your_data_dir/tensorboard`


## Configurations

Some important configurations in [experiments.conf](experiments.conf):
* `data_dir`: the full path to the directory containing dataset, models, log files
* `coref_depth` and `higher_order`: controlling the higher-order inference module
* `incremental`: if true uses an incremental approach, otherwise the c2f mode is used
* `incremental_singleton`: give the explicit option to discard new mentions, enables the model to output singletons
* `incremental_teacherforcing`: whether to use teachers forcing when creating and updating entity representations, greatly improves convergence speed
* `evict` whether to evict entity representations in the incremental model from active entity pool after a period of no new mentions
* `unconditional_eviction_limit` after how long of a distance with no mentions to evict an entity
* `singleton_eviction_limit` after how long of a distance of no mentions to evict a singleton entity
* `bert_pretrained_name_or_path`: the name/path of the pretrained BERT model ([HuggingFace BERT models](https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained))
* `max_training_sentences`: the maximum segments to use when document is too long; for BERT-Large and SpanBERT-Large, set to `3` for 32GB GPU or `2` for 24GB GPU


## Model Archival

Using the model name from the experiments.conf and the relative path to the model binary, create a model archive.
Optionally supply 'c2f' or 'incremental' as the model type (defaults to incremental).
```

./archive-model.sh <model_name> <path_to_model/model.bin> [MODEL_VARIANT]

```

## Torchserve

First install torchserve which is not part of our requirements.txt: `pip install torchserve`
Using torchserve models saved in this manner can be served, e.g.:

```
torchserve --models droc_incremental=<model_name>.mar
```

Since the native dependencies may cause issues one some systems we have a custom torchserve docker image in `docker/`.

### Torchserve-API

The model handlers essentially provide the http API, there are two modes of operation for our handlers.
* Using raw text
* Using pretokenized text

Raw text is useful for direct visualization (example requests made using [httpie](https://httpie.io/)),
in this context you may also want to try the 'raw' output mode for relatively human-friendly text.
```
http http://127.0.0.1:8080/predictions/<model_name> output_format=raw tokenized_sentences="Die Organisation gab bekannt sie habe Spenden veruntreut."
```

In the context of a larger language pipeline, pretokenization is often desirable:
```
http http://127.0.0.1:8080/predictions/<model_name> output_format=conll tokenized_sentences:='[["Die", "Organisation", "gab", "bekannt", "sie", "habe", "Spenden", "veruntreut", "."], ["Next", "sentence", "goes", "here", "!"]]'
```
