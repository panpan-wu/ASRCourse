- train.txt 来自 thchs30 train
- dev.txt 来自 thchs30 dev
- test.txt 来自 thchs30 test
- lm.sh 是实验相关脚本

执行

```
ngram-count -read train-3gram.count -order 3 -lm train-3gram.arpa -interpolate -kndiscount
```

会报下列错误，网上有种说法是训练集过小，有待验证。

> one of required modified KneserNey count-of-counts is zero
> error in discount estimator for order 3

所以这次作业是用 wittenbil 算法进行实验的。
