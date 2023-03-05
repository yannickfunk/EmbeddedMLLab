## Ideas for Chapter 3 - Further Optimizations

- COCO could be a vaiable dataset to improve upon VOCDetection

## Unrelated Ideas

- Replace evaluation logic with functions from `torchmetrics`. 


## Ideas for Evaluation/Experiments

To show our results at the end in a structured manner, we could do the following comparisons: 

- Compare combinations of our methods regarding mAP, inference latency (should be directly correlated to FPS and could be tested in COLAB). 
  Methods to compare:
  - Each methods speedup on its own
  - Each methods contribution to the overall result
  - Compare different pruning techniques (Dependency aware or not). For unaware pruning there could be still some potential because of the sparseity.
  - Different pruning levels. 
  - Test if pruning iteratively up to the desired pruning ratio or doing it in one step yields different results (pruning aware training)

# Open questions

## How to handle the different experiments in code?

I thought of having a separate python file for each experiment e.g. for pruning, retraining, onnx, layer fusing, etc. 
Each file should define its own trainer (maybe inherit a base trainer). 

It seems like all the parameters passed to trainer are saved in the checkpoint. 
Therefor it should be possible to load a specific state of training and change only the parameters necessary. 

## Task description

What does this mean: "In difference to the exercise, the tinyyoloV2 implementation does not automatically adjust to pruned channels."?

## Problems

It seems that the implemented ONNX runtime runs just on the CPU -> onnxruntime-gpu