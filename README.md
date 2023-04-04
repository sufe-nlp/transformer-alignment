## Code for EMNLP 2020 paper **Accurate Word Alignment Induction from Neural Machine Translation**.

First install fairseq-v0.9.0 with guidance [here](https://github.com/pytorch/fairseq/tree/v0.9.0). Then process data with the [guidance](https://github.com/lilt/alignment-scripts/tree/master/preprocess) and get binarized processed data with `fairseq-preprocess` command.

More details are in scripts/run.sh file.


> If you find this repo useful, please cite our paper:


```bibtex
@inproceedings{chen2020accurate,
    title = "Accurate Word Alignment Induction from Neural Machine Translation",
    author = "Chen, Yun  and
      Liu, Yang  and
      Chen, Guanhua  and
      Jiang, Xin  and
      Liu, Qun",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    pages = "566--576",
}

```