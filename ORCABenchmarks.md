# ORCA Performance Benchmarks
In this page, we provide performance benchmarks of DeGirum ORCA1 AI accelerator 
on various models. The frames per second (FPS) numbers are obatained by running the [single model performance test.ipynb](examples/benchmarks/single_model_performace_test.ipynb) jupyter notebook on a machine equipped with ORCA1. The script can also be run on the cloud platform to estimate the performance. All FPS numbers are for __batch_size=1__. This page will be periodically updated to reflect the latest performance numbers. As our compiler and software mature, we expect to add more models and also improve the performance.

__Last updated on: May 22, 2023__

| Model Name                                    | FPS |
| -------------                                 |:------:| 
| efficientnet_es_imagenet--224x224_quant       | 187 | 
| mobiledet_coco--320x320_quant                 | 128 | 
| mobilenet_v1_imagenet--224x224_quant          | 407 | 
| mobilenet_v2_imagenet--224x224_quant          | 360 |
| resnet50_imagenet--224x224_pruned_quant       | 250 |
| yolo_v5s_face_det--512x512_quant              | 126 |
