# NEXUS downstream diagnosis reproduction

The original supervisor dispatched the original downstream agent. The downstream agent retrieved the published T1 gray matter prediction guide, used its ReAct loop, wrote a classifier, and reported back. The model received a description of the MRI tensor and task. Participant images, identifiers, labels, and validation metrics remained offline.

The experiment measures the downstream modeling component on existing structural MRI inputs. Raw image preprocessing, visual quality control, and the functional connectivity experiments in the NEXUS paper were outside this run. Code generation allowed the upstream file writing tool at one destination. A separate evaluator ran the reviewed code to keep the outer validation labels and the test split out of model selection.

## Reproduce

Clone [the official source](https://github.com/LearningKeqi/Virtual-Neuroscientist-NEXUS/tree/c2e15d4983489fd89d604fdbbefbbf1726958068) at commit `c2e15d4983489fd89d604fdbbefbbf1726958068`. The orchestrator environment used Python 3.13 and the packages in `environment.lock`. Set `OPENAI_API_KEY` in the generation environment. Generating a fresh classifier uses the external model and may produce a different program. The frozen program can be evaluated entirely offline.

```sh
python generate.py upstream new_generation
python evaluate.py /path/to/adhd_round1 generation_01 /path/to/new_output --threshold 0.5
```

The frozen classifier extracts subject-local intensity summaries, multiscale block means and variances, and low frequency discrete cosine coefficients from whole-head T1 arrays. The classifier standardizes using training data and selects logistic regression regularization by stratified fivefold cross validation inside the training set. The evaluator reports probability scores and validation metrics under the original outer fold.

A real ADHD fold one smoke check used six subjects per class for training and two per class for validation. It returned four scores in 5.42 seconds. The resulting metrics serve as interface checks only. The full ADHD and ABIDE submissions have separate directories and an immutable program digest recorded in Delta's submission manifest.
