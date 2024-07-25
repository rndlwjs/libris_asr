# libris_asr

This repository is the basic implementation of Automatic Speech Recognition.

### Training examples

```
python ./train.py
```

 ### Results
 model|test-100 wer
	:---:|:---:
	DS2|-
	Conformer|26.3%

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

### Code Reference

[1] https://www.assemblyai.com/blog/end-to-end-speech-recognition-pytorch/
