# VAEXMnist

**VAEX (Variational Autoencoder Explainability)** is a model-agnostic generative framework for visual counterfactual explanations. Counterfactual reasoning explains a black-box classifier by fabricating realistic variations of an input that elicit a different prediction: *"How would this image have to change for the classifier to predict another class?"*

Unlike methods that require white-box gradient access, per-sample latent optimization, or heuristic input perturbations, VAEX conditions a generative model directly on the continuous predictions of the classifier under audit. By constraining generation to the learned data manifold, VAEX produces smooth, semantically plausible transformations. This repository applies VAEX to MNIST and a shortcut-biased MNIST variant to investigate whether counterfactual explanations edit true semantic features (digit shape) or spurious shortcut features (stroke width).

The project uses the [DRYTorch](https://github.com/nverchev/drytorch) engine for ML experimentation.

---

## The VAEX Method

Visual counterfactual explanations answer class-contrastive questions: given an input evaluated by a classifier, what minimal, plausible changes would flip the prediction?

VAEX achieves this through a four-step generative process:

1. **Classifier Evaluation:** The black-box classifier evaluates an input $x$ to produce a continuous prediction $y$ (class probabilities).
2. **Conditional Encoding:** The VAE encoder maps the input together with its predicted condition into a latent representation $z \sim q_\phi(z \mid x, y)$.
3. **Intervention:** The conditioning vector is replaced by the target class condition $y^*$.
4. **Counterfactual Generation:** The decoder generates the counterfactual sample $x^* \sim p_\theta(x \mid z, y^*)$. 

Because generation is constrained to the data manifold modeled by the VAE, counterfactuals remain realistic and semantically sound without needing gradient access to the classifier or per-sample optimization loops.

---

## Experimental Setup

This repository uses VAEX to audit whether counterfactual explanations edit true semantic features or spurious shortcuts. 

### Experiment Design
- **Classifier** — A convolutional neural network trained on the digits, acting as the audited model.
- **Generative Backbone** — A convolutional variational autoencoder with residual blocks that models the image distribution.
- **Conditioning Modes** (`autoencoding.model.name`):
  - `vaex` (default): The condition is the softmax of the classifier's predictions scaled by temperature, capturing the classifier's continuous decision boundary.
  - `cvae`: The condition is the discrete one-hot ground-truth label.
- **Shortcut Audit:** Evaluated on a stroke-thickness biased MNIST and on a random half of MNIST of the same size (`half_mnist`) (detailed in [docs/biased_mnist.md](docs/biased_mnist.md)) to test which feature the generator alters when multiple predictive routes exist.

---

## Biased MNIST

`biased_mnist` is a variant in which **stroke thickness leaks the digit class**: thick
samples are removed from digits 0–4 and thin ones from digits 5–9, allowing a model to reach
the label through stroke width alone. This gives the counterfactual generator a shortcut
feature it is expected to edit.

The unbiased comparison, `half_mnist`, removes the same share of each class at random,
matching the biased dataset in size and class balance without the shortcut.

Each phase chooses the data it trains on (`training_data/biased.yaml` or
`training_data/half.yaml`), and four entry configurations combine them:

| Entry configuration | Classifier trained on | VAE trained on |
|---|---|---|
| `half` (default) | `half_mnist` | `half_mnist` |
| `biased` | `biased_mnist` | `biased_mnist` |
| `biased_classifier` | `biased_mnist` | `half_mnist` |
| `biased_vae` | `half_mnist` | `biased_mnist` |

`biased_classifier` and `biased_vae` separate the two sources of a shortcut in the
counterfactuals: the classifier, which they should explain, and the training data of the VAE,
which they should not reflect. Each entry configuration keeps its experiments in its own
folder: select one with `--config-name`.

> For theoretical motivation, derivation of the morphological estimator, and empirical validation, see [**docs/biased_mnist.md**](docs/biased_mnist.md).

```bash
uv run explore_dataset.py --config-name biased  # samples of the biased dataset
```

### Options (`configs/experiment/data/dataset/mnist.yaml`)

| Option | Meaning |
|--------|---------|
| `thin_classes` | Classes kept thin; every other class is kept thick. |
| `quantile` | Fraction of each class kept, as a per-class quantile of the stroke width. `0.5` removes the half that disagrees with the label; `1.0` keeps the standard dataset. |
| `ink_threshold` | Intensity above which a pixel counts as ink when the stroke width is measured. |

Thresholds are measured on the training partition (`train`, or `train_val` with
`final=true`), and training and validation are cut at the same absolute widths. The test
partition stays whole, to measure how much a model relies on the shortcut.

---

## Project structure

```
VAEXMnist/
├── configs/
│   ├── experiment/               # Phases (classification, autoencoding), data and user settings
│   └── tuning/                   # Optuna search spaces
├── docs/                         # Construction and statistics of the biased MNIST
├── src/
│   ├── config/                   # Pydantic specs, options, and DRYTorch experiment wiring
│   ├── data/                     # MNIST and biased MNIST loading, structures
│   ├── module/                   # Classifier, CVAE and VAEX, and their layers
│   ├── trackers/                 # Trackers of the images the scripts publish, and of the learning curves
│   ├── lib/                      # DRYTorch components: losses, metrics, learning schemas, models
│   └── utils/                    # Distributed training, profiling, tuning
├── tests/                        # Unit tests
├── train_classifier.py           # Classifier training
├── evaluate_classifier.py        # Classifier evaluation across test partitions
├── train_vae.py                  # CVAE / VAEX training
├── tune_vae.py                   # VAE hyperparameter tuning
├── plot_tuning_results.py        # Hyperparameter search visualisations
├── evaluate_counterfactuals.py   # Classifier on the shortcut, VAE and counterfactuals
├── explore_dataset.py            # Dataset inspection and statistics
├── profile_vae.py                # VAE parameter count and throughput profiling
├── reconstruct_samples.py        # Sample reconstructions
├── generate_counterfactuals.py   # Counterfactual grid
├── generate_misclassified_counterfactuals.py  # Misclassified samples turned towards their labels
├── generate_random_samples.py    # Random samples
├── plot_confidence.py            # Distribution of predicted class probabilities
└── pyproject.toml
```

---

## Installation

The project requires **Python 3.13** and uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install core dependencies
uv sync

# Optional extras
uv sync --extra track_all   # TensorBoard, W&B, SQLAlchemy
uv sync --extra finetune    # Optuna hyperparameter tuning
uv sync --all-extras --dev  # everything, with the development tools
```

### Configure the paths

Set the paths in a `.env` file at the project root. MNIST is downloaded into `DATA_DIR`
on the first run.

```dotenv
DATA_DIR=/path/to/your/datasets
ROOT_EXP_DIR=./experiments
```

---

## Usage

### Training

- **Train the classifier and test its accuracy:**
  ```bash
  uv run train_classifier.py
  uv run evaluate_classifier.py
  ```

- **Train VAEX (default):**
  Conditioned on the trained classifier predictions:
  ```bash
  uv run train_vae.py
  ```

- **Train CVAE:**
  CVAE is conditioned on the ground-truth labels instead of the classifier predictions (`autoencoding.model.name=cvae`). Because overrides define a distinct experiment name, if you want to use CVAE you need to clone the trained classifier into it (for downstream evaluation) or retrain the classifier separately:

  - *Clone the classifier (recommended):*
    ```bash
    uv run clone_experiment.py autoencoding.model.name=cvae
    uv run train_vae.py autoencoding.model.name=cvae
    ```
  - *Or train the classifier separately:*
    ```bash
    uv run train_classifier.py autoencoding.model.name=cvae
    uv run train_vae.py autoencoding.model.name=cvae
    ```

Any custom override that changes the experiment name (such as `autoencoding.train.n_epochs=100`) follows the same rule: clone the base experiment with `clone_experiment.py <overrides>` to reuse earlier phases, or train them with the matching overrides. Use `--from NAME` to clone from another experiment variation (e.g. `--from my_variation`).

### Tuning

```bash
uv run tune_vae.py                  # learning rate, weight decay and warmup
uv run tune_vae.py tune=architecture
uv run plot_tuning_results.py tune=architecture
```

The trials of VAEX read the classifier trained with the fixed overrides of the study
(`overrides` in `configs/tuning/defaults.yaml`): train it first with the same overrides.

### Visualisation and Evaluation

- **Evaluate the classifier on all, aligned and conflicting images, the reconstructions and the
  classifier on them, and the success and the changes of the counterfactuals:**
  ```bash
  uv run evaluate_counterfactuals.py
  ```

- **Generate counterfactuals of the selected samples:**
  ```bash
  uv run generate_counterfactuals.py
  uv run generate_counterfactuals.py user.generate.target_class=7 user.generate.target_value=0.5
  ```
  `target_value` is the probability of the target class in the condition, the other classes
  sharing the rest: `1.0` asks for the target class alone, `0.1` for none in particular.

- **Generate counterfactuals of misclassified samples, towards their labels:**
  ```bash
  uv run generate_misclassified_counterfactuals.py
  ```

- **Reconstruct dataset samples:**
  ```bash
  uv run reconstruct_samples.py
  ```

- **Generate random samples:**
  ```bash
  uv run generate_random_samples.py
  ```

- **Plot the distribution of the probability of the predicted class**, the condition VAEX
  receives, after `autoencoding.model.temperature`; `user.plot.use_train=true` uses the
  training images:
  ```bash
  uv run plot_confidence.py final=true
  ```

- **Inspect the dataset and profile the VAE:**
  ```bash
  uv run explore_dataset.py
  uv run profile_vae.py
  ```

The scripts that draw images publish them as events: the `matplotlib` tracker saves them
under the `images` folder of the run, and the `tensorboard` tracker shows them in
TensorBoard. Each sample has its own folder, which every script adds to:

```
test/sample_11/original.png
test/sample_11/reconstruction.png
test/sample_11/difference.png         # between the original and the reconstruction
test/sample_11/target_1.png
test/sample_11/target_1_value_0.7.png  # with user.generate.target_value=0.7
```

Set `user.trackers.matplotlib=false` to skip the files, or `user.plot.show=true` to also
display them. Running a script under a debugger skips every tracker, so it writes no files.

### Final pipeline

To run the complete benchmark sequentially with final test evaluation, for each entry
configuration:

```bash
for config in half biased biased_classifier biased_vae; do
  uv run train_classifier.py         --config-name $config final=true
  uv run train_vae.py                --config-name $config final=true
  uv run evaluate_classifier.py      --config-name $config final=true
  uv run evaluate_counterfactuals.py --config-name $config final=true
  uv run generate_counterfactuals.py --config-name $config final=true
done
```

`evaluate_counterfactuals.py` tests the classifier on the aligned and conflicting test
images, the VAE, the classifier on its reconstructions (`Classification_reconstructions`), and
its counterfactuals, including those that turn each misclassified image towards its label
(`Counterfactuals_misclassified`); the trackers record each test, for example in
`csv_metrics` with `user.trackers.csv=true`.

---

## Results

Evaluated on the full test partition, with the logits of the classifier divided by a temperature of 5 before the softmax. The temperature does not apply to the CVAE, which is conditioned on the labels.

### Reconstruction and Generative Quality

Reconstructions decoded from the latent representation are evaluated directly and under the classifier trained on the same configuration.

| Entry configuration | NLL | $D_{KL}$ | Reconstruction MSE | Reconstructions classified |
|---|---|---|---|---|
| `half` | −212.6 | 26.5 | 4.95e-3 | 98.2 % |
| `biased` | −210.3 | 27.0 | 5.39e-3 | 90.4 % |
| `biased_classifier` | −212.2 | 26.6 | 5.02e-3 | 92.4 % |
| `biased_vae` | −211.2 | 27.1 | 5.22e-3 | 98.3 % |
| `half`, CVAE | −212.7 | 27.8 | 4.92e-3 | 99.4 % |

### Counterfactual Explanations and Shortcut Attribution

Counterfactuals turn each input toward unpredicted classes (and misclassified inputs toward their true label). Changes are measured against the reconstruction from the same latent sample: **Change** is the mean squared pixel difference, while **Width change** is the stroke width delta between digit groups (0–4 vs. 5–9).

| Entry configuration | Successful counterfactuals | Change | Misclassified, towards label | Change, misclassified | Width change to 5–9 | Width change to 0–4 |
|---|---|---|---|---|---|---|
| `half` | 98.6 % | 0.064 | 96.6 % | 0.027 | −0.08 | +0.07 |
| `biased` | 97.7 % | 0.063 | 99.5 % | 0.030 | +0.54 | −0.34 |
| `biased_classifier` | 97.4 % | 0.060 | 98.5 % | 0.024 | +0.17 | −0.20 |
| `biased_vae` | 98.1 % | 0.065 | 100.0 % | 0.029 | +0.54 | −0.26 |
| `half`, CVAE | 86.6 % | 0.053 | 63.9 % | 0.000 | −0.06 | +0.06 |

### Key Findings

- **Consistent generative quality:** Across all configurations, the VAE models achieve comparable likelihood, reconstruction error, and perturbation magnitudes. Observed differences in counterfactuals stem from feature attribution rather than generative capacity.
- **Generative bias inflates attribution:** When the VAE is trained on shortcut-biased data (`biased`, `biased_vae`), counterfactuals alter stroke thickness far more (+0.54) than a biased classifier alone causes (+0.17), even when the audited classifier does not use that shortcut. The decoder learns the shortcut from its training distribution and bakes it into generation, falsely attributing stroke-width dependence even to an unbiased model (`biased_vae`).
- **Unbiased backbones isolate classifier shortcuts:** A classifier's genuine reliance on the shortcut only reveals itself when the generative backbone is unbiased (`biased_classifier`). There, the counterfactuals of a shortcut-dependent classifier edit stroke width in the direction of the shortcut (+0.17 and −0.20), while those of an unbiased classifier do not (−0.08 and +0.07 in `half`). The success rate does not tell the two apart.
- **VAEX vs. CVAE on misclassifications:** Conditioning on classifier predictions is critical for explaining errors. When presented with a misclassified image, a label-conditioned CVAE receives the true label at the encoder, causing counterfactual generation toward the label to collapse into identity reconstruction (zero change). VAEX instead conditions on the classifier's actual prediction, producing targeted semantic transformations that flip misclassified samples toward their true label.
- **Effect of the temperature:** Softening the condition makes counterfactuals more effective. Compared to a temperature of 1, the success rate on `half` rises from 89.1 % to 98.6 %, at the cost of a larger change (0.054 to 0.064).

> For verification of classifier shortcut reliance, see §5.1 in [docs/biased_mnist.md](docs/biased_mnist.md).

---

## Citation

```bibtex
@inproceedings{vercheval2021hierarchical,
  title={Hierarchical Variational Autoencoders for Visual Counterfactuals},
  author={Vercheval, Nicolas and Pizurica, Aleksandra},
  booktitle={2021 IEEE International Conference on Image Processing (ICIP)},
  pages={2513--2517},
  year={2021},
  organization={IEEE},
  doi={10.1109/ICIP42928.2021.9506253}
}
```
