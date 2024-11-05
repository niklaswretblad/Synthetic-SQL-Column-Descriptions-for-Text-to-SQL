# Synthetic SQL Column Descriptions and Their Impact on Text-to-SQL Performance

## Updates

- **2024-11-15:** The paper got accepted and will be presented at the 3rd Table Representation Learning Workshop @ NeurIps 2024! See you at NeurIps! 🎉 

## Introduction

This is the official repository for the paper [Synthetic SQL Column Descriptions and Their Impact on Text-to-SQL Performance](https://arxiv.org/abs/2408.04691)

Relational databases often have ambiguous columns and hard-to-interpret values, hindering both human users and text-to-SQL models. This work leverages large language models (LLMs) to automatically generate natural language descriptions for SQL database columns, enhancing text-to-SQL performance and automating metadata creation. We refined the column descriptions from the development set of the BIRD-Bench benchmark, developed a column difficulty taxonomy, and evaluated various LLMs. While models struggled with generating descriptions for inherently ambiguous columns, the generated descriptions consistently improved text-to-SQL performance, especially for larger models like GPT-4o and Qwen2 72B. Surprisingly, descriptions containing by annotators rated superflous information outperformed curated ones, indicating benefits from more comprehensive metadata. Future work will explore optimizing these descriptions and expanding metadata types.

## Dataset

As part of the study, we curate a dataset of column descriptions and their difficulty ratings based on the development set of [BIRD-Bench](https://bird-bench.github.io/). The dataset can be found in the `column_descriptions.csv` file in the top level directory. 

Columns were categorized into four difficulty levels: "Self-Evident," "Context-Aided," "Ambiguity-Prone," and "Domain-Dependent," based on the amount of information available in the database to generate accurate descriptions. We also included an alternative naming convention to make it more intuitive to interpret the difficulties of columns in the file: "Easy", "Medium", "Hard" and "Very Hard".

## Annotations

All annotations made for the paper can be found in the /annotations folder. 

## Running the code

To do. 

## Citation

Bibtex:
```
@misc{wretblad2024syntheticsqlcolumndescriptions,
      title={Synthetic SQL Column Descriptions and Their Impact on Text-to-SQL Performance}, 
      author={Niklas Wretblad and Oskar Holmström and Erik Larsson and Axel Wiksäter and Oscar Söderlund and Hjalmar Öhman and Ture Pontén and Martin Forsberg and Martin Sörme and Fredrik Heintz},
      year={2024},
      eprint={2408.04691},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.04691}, 
}
```
