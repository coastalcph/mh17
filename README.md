# Mapping (Dis-)Information Flow about the MH17 Plane Crash

This repository contains the code for the tweet classification experiments described in Hartmann et al. (2019): Mapping (Dis-)Information Flow about the MH17 Plane Crash

The classifiers are located in encoders/ folder. The classifiers rely on pre-trained embeddings, in the experiments reported in the paper we used fasttext embeddings trained on Wikipedia and aligned between languages available here: https://fasttext.cc/docs/en/aligned-vectors.html 
Location of data and embeddings have to be specified in the config.cfg. Hyperparameters can be specified in the .csv files in hyperparams/

Unfortunately, we are not allowed to share the data used in the experiments directly. We can only provide the annotations along with matching tweet ids. If you are interested in obtaining those or have questions, please contact Mareike at hartmann@di.ku.dk

If you use code from this repo, please cite:

@inproceedings{hartmann2019mapping,\
  title={Mapping (Dis-) Information Flow about the MH17 Plane Crash},\
  author={Hartmann, Mareike and Golovchenko, Yevgeniy and Augenstein, Isabelle},\
  booktitle={Proceedings of the Second Workshop on Natural Language Processing for Internet Freedom: Censorship, Disinformation, and Propaganda},\
  pages={45--55},\
  year={2019}
}


If you use data or annotations used in this project, please cite:

@article{golovchenko2018state,\
  title={State, media and civil society in the information warfare over Ukraine: citizen curators of digital disinformation},\
  author={Golovchenko, Yevgeniy and Hartmann, Mareike and Adler-Nissen, Rebecca},\
  journal={International Affairs},\
  volume={95},\
  number={5},\
  pages={975--994},\
  year={2018}
}


