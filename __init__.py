# V2主要是用来生成full data中每个examples token-level的gpt2的loss
# V3用来初步探索P(y|x1,y1,x) - p(y|x)
# V4是用来写progressive地找重要样本以及搜索最终的拼接结果的。
# V4之前写了迭代搜最优的importance+diversity score，但是涉及到搜train上的acc的时候得改下mydataset，所以就新开了一个V5
#V6是多了对NLI数据集的支持
#V7是由于之前实验结果太乱了 所以重新开一个

