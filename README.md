**CSLR — CorrNet+ (Continuous Sign Language Recognition)
Overview**

This repository contains end-to-end code and a lightweight Flask demo for Continuous Sign
Language Recognition (CSLR) using a CorrNet+-style architecture (ResNet-based visual
backbone, temporal convolutions, BiLSTM, and CTC decoding). The demo accepts short videos
and outputs sentence glosses. The research tracks two settings:

* German Sign Language (PHOENIX2014-T): strong baseline performance (WER
~18–20%, depending on exact hyperparameters).
* Indian Sign Language (ISL-CSLRT) fine-tuning: initial results (WER ~71%), highlighting
low-resource and domain-shift challenges.

**Responsible Use**

This project is for research and education. It is not a substitute for certified interpreters,
especially in legal, medical, or emergency contexts.

**Key Features**

* CorrNet+-style spatio-temporal modeling with CTC decoding.
* Reproducible configs and a simple CLI for train/eval.
* Video preprocessing and light augmentation utilities.
* Flask web UI for quick demos that output sentence glosses.
* Hooks for cross-lingual fine-tuning (e.g., from PHOENIX2014-T to ISL).

**Datasets**

* **PHOENIX2014-T (German Sign Language):** primary benchmark for training and
evaluation.
* **ISL (e.g., ISL-CSLRT):** used for fine-tuning from a PHOENIX2014-T checkpoint.

**Results (indicative)**

* PHOENIX2014-T: WER ≈ 18–20% (varies with preprocessing, seeds, and
hyperparameters).
* ISL fine-tuning (ISL-CSLRT): WER ≈ 71% (underscores data scarcity and domain shift).

**Reflections**

* Strong cross-lingual transfer requires adequate, task-aligned ISL data; architecture alone
is not the bottleneck.
* Gradual unfreezing and moderate augmentation improved stability but could not fully
overcome data limitations.
* Clear communication of model limits and responsible release practices is essential in
low-resource settings.

**Roadmap**

* Multi-temporal attention and better temporal modeling.
* Domain adaptation for ISL (feature alignment, gloss mapping, pseudo-labeling).
* Inference efficiency for web/mobile (pruning, quantization, distillation).
* Expanded tests and CI; improved user feedback loop in the demo.

**Acknowledgements**

Thanks to the CSLR research community and prior CorrNet+/I3D/BiLSTM-CTC works that
informed this implementation and structure.
