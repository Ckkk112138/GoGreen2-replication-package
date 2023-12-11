# The Impact of Knowledge Distillation on the Performance and Energy Consumption of NLP Models

This repository serves as a companion page for the paper titled "The Impact of Knowledge Distillation on the Performance and Energy Consumption of NLP Models," submitted to the CAIN 2024 conference.

In this work, we aim to determine how Knowledge Distillation affects the energy consumption and performance of NLP models.

<!-- For any inquiries, please contact the authors via the provided email addresses. -->

<!-- ## Authors (Affiliation: Vrije Universiteit Amsterdam)
**Eloise Zhang** (j.zhang6@student.vu.nl), **Ye Yuan** (y.yuan3@student.vu.nl), **Zongyao Zhang** (z.zhang14@student.vu.nl)
, **Kaiwei Chen** (k.chen2@student.vu.nl), **Jiacheng Shi** (j.shi2@student.vu.nl), **Vincenzo Stoico** (v.stoico@vu.nl), **Ivano Malavolta** (i.malavolta@vu.nl) -->


## Repository Contents
This repository contains all the materials required for replicating the study, including:

- Source code required to reproduce our experiment. Detailed information about dependencies and setup instructions are provided.

- Dataset generated from the experiment that led to the results presented in our paper.

- Code necessary to reproduce the results presented in our paper using the provided dataset.

<!-- ## How to cite us
The scientific article describing design, execution, and main results of this study is available [here](https://www.google.com).<br> 
If this study is helping your research, consider to cite it is as follows, thanks!

```
@article{,
  title={},
  author={},
  journal={},
  volume={},
  pages={},
  year={},
  publisher={}
}
``` -->

## Quick started

- To get started:

  ```
  git clone https://github.com/Ckkk112138/GoGreen2-replication-package.git
  cd GoGreen2-replication-package/
  pip install -r requirements.txt
  ```

- To execute the experiment and generate the dataset in `data/experiment_data`:

  `python src/experiment-runner src/script/linux-powerjoular-profiling/RunnerConfig.py`

- To reproduce the results from the dataset:

  `python data/script/<example_test>`
  Replace `<example_test>` with the actual name of your test file under `data/script` folder.

## Repository Structure
This is the root directory of the repository. The directory is structured as follows:

    GoGreen2-replication-package
     .
     |
     |--- src/                             Source code and dependencies used in the paper
            |
            |--- script/                   Code for executing the experiment
     |
     |--- documentation/                   Our paper detailing the experiment settings and results
     |
     |--- data/                            Data used in the paper 
            |
            |--- experiment_data/          Dataset generated from the experiment
            |--- script/                   Script for generating and visualizing the graphs from the dataset
            |--- plot/                     Graphical representation of the results
  

<!-- Usually, replication packages should include:
* a [src](src/) folder, containing the entirety of the source code used in the study,
* a [data](data/) folder, containing the raw, intermediate, and final data of the study
* if needed, a [documentation](documentation/) folder, where additional information w.r.t. this README is provided. 

In addition, the replication package can include additional data/results (in form of raw data, tables, and/or diagrams) which were not included in the study manuscript. -->

<!-- ## Replication package naming convention
The final name of this repository, as appearing in the published article, should be formatted according to the following naming convention:
`<short conference/journal name>-<yyyy>-<semantic word>-<semantic word>-rep-pkg`

For example, the repository of a research published at the International conference on ICT for Sustainability (ICT4S) in 2022, which investigates cloud tactics would be named `ICT4S-2022-cloud-tactics-rep-pkg` -->

## License
<!-- As general indication, we suggest to use:
* [MIT license](https://opensource.org/licenses/MIT) for code-based repositories, and 
* [Creative Commons Attribution 4.0	(CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) for text-based repository (papers, docts, etc.). -->

GoGreen2-replication-package is licensed under the [MIT license](https://opensource.org/licenses/MIT), allowing users to freely use, modify, and distribute the code for various purposes.

<!-- For more information on how to add a license to your replication package, refer to the [official GitHUb documentation](https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions/adding-a-license-to-a-repository). -->
