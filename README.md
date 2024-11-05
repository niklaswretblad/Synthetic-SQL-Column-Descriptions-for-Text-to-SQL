# Synthetic SQL Column Descriptions and Their Impact on Text-to-SQL Performance

## Updates

- **2024-11-15:** The paper got accepted and will be presented at the 3rd Table Representation Learning Workshop @ NeurIps 2024! See you at NeurIps! 🎉 

## Introduction

This is the official repository for the paper [Synthetic SQL Column Descriptions and Their Impact on Text-to-SQL Performance](https://arxiv.org/abs/2408.04691)

Relational databases often suffer from uninformative descriptors of table contents, such as ambiguous columns and hard-to-interpret values, impacting both human users and text-to-SQL models. In this paper, we explore the use of large language models (LLMs) to automatically generate detailed natural language descriptions for SQL database columns, aiming to improve text-to-SQL performance and automate metadata creation. We create a dataset of gold column descriptions based on the BIRD-Bench benchmark, manually refining its column descriptions and creating a taxonomy for categorizing column difficulty. We then evaluate several different LLMs in generating column descriptions across the columns and different difficulties in the dataset, finding that models unsurprisingly struggle with columns that exhibit inherent ambiguity, highlighting the need for manual expert input. We also find that incorporating such generated column descriptions consistently enhances text-to-SQL model performance, particularly for larger models like GPT-4o, Qwen2 72B and Mixtral 22Bx8. Notably, Qwen2-generated descriptions, containing by annotators deemed superfluous information, outperform manually curated gold descriptions, suggesting that models benefit from more detailed metadata than humans expect. Future work will investigate the specific features of these high performing descriptions and explore other types of metadata, such as numerical reasoning and synonyms, to further improve text-to-SQL systems.

## Datasets

As part of the study, we curate a dataset of column descriptions and their difficulty ratings based on the development set of [BIRD-Bench](https://bird-bench.github.io/). The dataset can be found in the /data folder. 


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
