<img alt="SIGNXAI-Example" src="img/sign_mnist.png" width=500>

[![PyPI Version](https://img.shields.io/pypi/v/signxai2)](https://pypi.org/project/signxai2/)
[![License](https://img.shields.io/pypi/l/signxai2)](https://github.com/TimeXAIgroup/signxai2/blob/master/COPYING.LESSER)


SIGN (**S**ign-based **I**mprovement of **G**radient-based expla**N**ations) is a novel XAI method intended to reduce bias in explanations that are intrinsically induced by several state-of-the-art XAI methods. The SIGN-XAI-2 package enables simple application of this method in your projects using the established Zennit (**Z**ennit **e**xplains **n**eural **n**etworks **i**n **t**orch) package [pypi.org/project/zennit/](https://pypi.org/project/zennit/).

If you use this package or parts of it in your own work, please consider citing our [paper](https://doi.org/10.1016/j.inffus.2023.101883):
```bibtex
 @article{Gumpfer2023SIGN,
    title = {SIGNed explanations: Unveiling relevant features by reducing bias},
    author = {Nils Gumpfer and Joshua Prim and Till Keller and Bernhard Seeger and Michael Guckert and Jennifer Hannig},
    journal = {Information Fusion},
    pages = {101883},
    year = {2023},
    issn = {1566-2535},
    doi = {https://doi.org/10.1016/j.inffus.2023.101883}
}
```

## Documentation
The latest documentation is available from [timexaigroup.github.io/signxai2](https://timexaigroup.github.io/signxai2/).

## Install
To install the package directly from PyPI using pip, use:
```shell
$ pip install signxai2
```

## Usage
SIGN-XAI-2 is based on Zennit and works with PyTorch. If you want to know more about Zennit and its usage, visit [github.com/chr5tphr/zennit](https://github.com/chr5tphr/zennit). There is also a version of SIGN available for usage with TensorFlow environments [pypi.org/project/signxai/](https://pypi.org/project/signxai/).

Example Code:

```python
import signxai2
```

For more details and examples, have a look at our
[**documentation**](https://timexaigroup.github.io/signxai2/).

## License
SIGN-XAI-2 and Zennit are licensed under the GNU LESSER GENERAL PUBLIC LICENSE VERSION 3 OR LATER -- see the [COPYING](COPYING) and [COPYING.LESSER](COPYING.LESSER) files for details.
