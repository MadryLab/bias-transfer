# When Does Bias Transfer in Transfer Learning?

This repository contains the code of our paper:

**When Does Bias Transfer in Transfer Learning?** </br>
*Hadi Salman\*, Saachi Jain\*, Andrew Ilyas\*, Logan Engstrom\*, Eric Wong, Aleksander Madry*  <br>
[Paper](https://arxiv.org/abs/2207.02842) - [Blog post](http://gradientscience.org/bias-transfer/)


```bibtex
@article{salman2022does,
  title={When does Bias Transfer in Transfer Learning?},
  author={Salman, Hadi and Jain, Saachi and Ilyas, Andrew and Engstrom, Logan and Wong, Eric and Madry, Aleksander},
  journal={arXiv preprint arXiv:2207.02842},
  year={2022}
}
```

<p>
<img src="BiasTransfer.png" width="1000" >
</p>

The major components of our repo are:

* [main.py](main.py): Entry point of our codebase.
* [src/](src): Contains all our code for running full transfer pipeline.
* [dataset_configs/](configs): Contains the config files that `main.py` expects. These config files contain the hyperparams for each transfer tasks.

## Getting started
*Our code relies on our [FFCV Library](https://ffcv.io/). To install this library, along with other dependencies including PyTorch, follow the instructions below.*

```
conda create -n ffcv python=3.9 cupy pkg-config compilers libjpeg-turbo opencv pytorch torchvision cudatoolkit=11.3 numba -c pytorch -c conda-forge 
conda activate ffcv
pip install ffcv
```

You are now ready to run our code!

## Training a source model on ImageNet.

A major component of our paper is training a source model (potentially with some spurious correlation).

* To train standard ImageNet model:

```
python main.py \
             --config-file dataset_configs/imagenet.yaml \
             --training.exp_name clean_imagenet \
             --training.outdir $OUTDIR
```
* To train **spurious** ImageNet model, which adds a backdoor to the ImageNet model (either a yellow square, or a fixed Gaussian patter, or hats):

```
python main.py \
            --config-file dataset_configs/imagenet.yaml \
            --spurious.spurious_type square \ #or "gaussian" or "hat"
            --spurious.spurious_perc 1 \
            --training.exp_name spurious_imagenet_square \
            --training.outdir $OUTDIR
```

where `$OUTDIR` is the output directory of your choice.

## Transferring to a downstream task.
Now given a pretrained ImageNet model (either clean or spurious), we can transfer this model to various downstream tasks by running:

```
python main.py --config-file dataset_configs/${DATASET}.yaml \
            --training.exp_name transfer \
            --training.outdir $OUTDIR \
            --model.transfer $TRANSFER_TYPE \
            --model.checkpoint ${CHECKPOINT_PATH}/version_0/checkpoints/checkpoint_latest.pt
```
where `$DATASET` is the downstream dataset which we want to transfer to (see next section), `$OUTDIR` is the output directory of your choice, `$TRANSFER_TYPE` is either `FIXED` or `FULL` denoting either fixed-feature transfer or full-network transfer, and `$CHECKPOINT_PATH` is the path of the saved checkpoint of the source ImageNet model from above.

This code also evaluates the trained model at the end of training to check for any bias. The results (accuracy, predictions, etc.) can be found in `$OUTDIR/results.pt`.

## Datasets that we use as transfer benchmarks (see our paper for citations) 
* aircraft ([Download]( https://robustnessws4285631339.blob.core.windows.net/public-datasets/fgvc-aircraft-2013b.tar.gz?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D
))
* birds ([Download]( https://robustnessws4285631339.blob.core.windows.net/public-datasets/birdsnap.tar?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D
))
* caltech101 ([Download]( https://robustnessws4285631339.blob.core.windows.net/public-datasets/caltech101.tar?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D
))
* caltech256 ([Download]( https://robustnessws4285631339.blob.core.windows.net/public-datasets/caltech256.tar?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D
))
* cifar10 **(Automatically downloaded when you run the code)**
* cifar100 **(Automatically downloaded when you run the code)**

* flowers ([Download]( https://robustnessws4285631339.blob.core.windows.net/public-datasets/flowers.tar?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D
))
* food ([Download]( https://robustnessws4285631339.blob.core.windows.net/public-datasets/food.tar?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D
))
* pets ([Download]( https://robustnessws4285631339.blob.core.windows.net/public-datasets/pets.tar?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D
))
* stanford_cars ([Download]( https://robustnessws4285631339.blob.core.windows.net/public-datasets/stanford_cars.tar?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D
))
* SUN397 ([Download]( https://robustnessws4285631339.blob.core.windows.net/public-datasets/SUN397.tar?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D
))

We have created an [FFCV](https://ffcv.io/) version of each of these datasets to enable super fast training. We will make these datasets available soon!

## Download our pretrained models
Coming soon!

## A detailed demo
Coming soon!
