# NLP Assignment: Aspect-Term Polarity Classification in Sentiment Analysis

## Students

- Alexandre Maranhão
- Artur Gallois

## Description

We used DistilBERT [1] followed by a classification network as our model. The choice of BERT derived models is motivated by them being pre-trained in a large corpus, open-source and still having a reasonable size. In particular, DistilBERT is a smaller, lighter version of BERT trained by distillation of the original model, that is, trained to predict the same probabilities and similar hidden states. As we don't have access to a lot of training data, a smaller model is less prone to overfitting.

DistilBERT is capable of doing sentence pair classification. We use as first sentence the original sentence field from the dataset, while as for the second one we map each category to a string, and then concatenate the target to it. For example, consider the following row from the dev set:

- Sentence: Great wine selection, Gigondas is worth the price, and the house champagne is a great value.
- Category: DRINKS#PRICES
- Target: Gigondas

DRINKS#PRICES is mapped to "prices of drink". Then, the input to the model will be the tokenized version of

```
[CLS] Great wine selection, Gigondas is worth the price, and the house champagne is a great value. [SEP] prices of drink Gigondas [SEP]
```

where [CLS] is the classification token that marks the beggining of the input and [SEP] is the separator between two sentences. The motivation for this approach is that having "prices of drink" on the second phrase should orient the model to look to the segment "worth the price".

The DistilBERT model outputs one representation per token, providing a word-level understanding. For sentence classification problems, we can take the representation that it assigns to the token [CLS], which is a sentence-level embedding. So from its raw output we take only raw_output[:, 0, :], where the first dimension is for each sample in the batch, the second dimension is for each token, and the third is the dimension of its feature space. This selected output is then passed through a MLP with two hidden layers and an output layer, with sizes 256, 32, 3 and ReLU activations. 

The whole model, including the pre-trained weight, is optimized with AdamW. Weight decay is a term in optimizer algorithms such as SGD and Adam that exponentially decreases the weights in order to regularize them - in the case of SGD, it is equivalent to L2 regularization. AdamW [2] is a version of Adam that decouples weight decay from the optimization steps with respect to the loss function, and ultimately improves Adam's ability to generalize. The authors suggest using as weight decay a hyperparameter called normalized weight decay multiplied by the square root of the ratio between the batch size b and the product between number of training samples B and number of epochs T ([2], Appendix B.1), that is,
$$ \lambda = \lambda_{norm} \sqrt{\frac{b}{BT}}$$

We ran training for 5 epochs with AdamW as a form of early stopping to prevent overfitting. The optimization process uses learning rate 1.2e-5, batch size 64, and normalized weight decay 0.048.

## Results

On the dev set, we got an average accuracy of 83.67% over 5 runs.

## References 

- [1] V. Sanh et al., DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter (2019), arXiv:1910.01108
- [2] Loshchilov,I., Hutter, F., Decoupled Weight Decay Regularization (2019), arXiv:1711.05101