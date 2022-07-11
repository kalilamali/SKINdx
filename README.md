# SKINdx

A deep learning system that can classify an image in 2 ways:

1. Binary classification of skin vs. non-skin.
2. 304 disease categories.

Categories excluded from predictions: [1, 50, 69, 73, 139, 222, 235, 253, 257, 296] (See categories.json)

## Getting Started
Before starting download the weights as best1.tar and place them in:

networks_binary: [best1.tar](https://figshare.com/articles/software/best1_tar/12906956)

networks_categories: [best1.tar](https://figshare.com/articles/software/Categories_weights/12906986)

```
python3 app.py
```

### Prerequisites


See requirements.txt

#### Questions

For questions, contact s181423@student.dtu.dk

## LICENSE
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
