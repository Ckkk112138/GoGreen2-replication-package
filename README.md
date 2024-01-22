# cain-2024-replication-package
This repository contains the replication package of the paper **The Impact of Knowledge Distillation on the Performance and Energy Consumption of NLP Models**, published at the International Conference on AI Engineering - Software Engineering for AI (CAIN 2024).

This study has been designed, developed and reported by the folllowing investigators:
- [Ye Yuan](mailto:y.yuan3@student.vu.nl) (Vrije Universiteit Amsterdam)
- [Eloise Zhang](mailto:j.zhang6@student.vu.nl) (Vrije Universiteit Amsterdam)
- [Zongyao Zhang](mailto:z.zhang14@student.vu.nl) (Vrije Universiteit Amsterdam)
- [Kaiwei Chen](mailto:k.chen2@student.vu.nl) (Vrije Universiteit Amsterdam)
- [Jiacheng Shi](mailto:j.shi2@student.vu.nl) (Vrije Universiteit Amsterdam)

For any information, interested researchers can contact us by sending an email to any of the investigators listed above.

## Structure of the replication package
This replication package is organized according to the following structure:
```
├── README.md: The file you are reading right now.
├── LICENSE: File describing under which license the repository's content is being made available.
├── data                        Data used in the paper 
│   ├── experiment_data         Dataset generated from the experiment
│   ├── script                  Script for generating and visualizing the graphs from the dataset
│   └── plot                    Graphical representation of the results
├── documentation               Our paper detailing the experiment settings and results
└── src                         Source code and dependencies used in the paper
    └──script                   Code for executing the experiment
```

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
  
## How to cite this work
If the data or software contained in this replication package is helping your research, consider citing it is as follows, thanks!

```
@inproceedings{
}
```

## License
<!-- As general indication, we suggest to use:
* [MIT license](https://opensource.org/licenses/MIT) for code-based repositories, and 
* [Creative Commons Attribution 4.0	(CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) for text-based repository (papers, docts, etc.). -->

GoGreen2-replication-package is licensed under the [MIT license](https://opensource.org/licenses/MIT), allowing users to freely use, modify, and distribute the code for various purposes.
