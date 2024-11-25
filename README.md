# libris_asr

This repository is the basic implementation of Automatic Speech Recognition using Pytorch-lightning.

### Training examples

```
python ./train.py
```

### Results
model (train-100)|test-clean
:---:|:---:
DS2|23.04%
Conformer|26.3%

<img width="1028" alt="스크린샷 2024-07-28 오전 11 26 11" src="https://github.com/user-attachments/assets/a5925cef-3d77-4c45-b05b-1fac761517ff">
<img width="1017" alt="스크린샷 2024-07-26 오전 6 25 26" src="https://github.com/user-attachments/assets/8d3364f0-95dd-44d6-8709-9a5e0855c386">


### Citation

[1] _Deep speech 2: End-to-end speech recognition in english and mandarin_
```
@inproceedings{amodei2016deep,
  title={Deep speech 2: End-to-end speech recognition in english and mandarin},
  author={Amodei, Dario and Ananthanarayanan, Sundaram and Anubhai, Rishita and Bai, Jingliang and Battenberg, Eric and Case, Carl and Casper, Jared and Catanzaro, Bryan and Cheng, Qiang and Chen, Guoliang and others},
  booktitle={International conference on machine learning},
  pages={173--182},
  year={2016},
  organization={PMLR}
}
```

[2] _Conformer: Convolution-augmented transformer for speech recognition_
```
@article{gulati2020conformer,
  title={Conformer: Convolution-augmented transformer for speech recognition},
  author={Gulati, Anmol and Qin, James and Chiu, Chung-Cheng and Parmar, Niki and Zhang, Yu and Yu, Jiahui and Han, Wei and Wang, Shibo and Zhang, Zhengdong and Wu, Yonghui and others},
  journal={arXiv preprint arXiv:2005.08100},
  year={2020}
}
```

[3] _E-branchformer: Branchformer with enhanced merging for speech recognition_
```
@inproceedings{kim2023branchformer,
  title={E-branchformer: Branchformer with enhanced merging for speech recognition},
  author={Kim, Kwangyoun and Wu, Felix and Peng, Yifan and Pan, Jing and Sridhar, Prashant and Han, Kyu J and Watanabe, Shinji},
  booktitle={2022 IEEE Spoken Language Technology Workshop (SLT)},
  pages={84--91},
  year={2023},
  organization={IEEE}
}
```
### Code Reference

[1] https://www.assemblyai.com/blog/end-to-end-speech-recognition-pytorch/

[2] https://github.com/espnet/espnet
