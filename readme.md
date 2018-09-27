## 比赛信息

### 赛题任务

自然语言处理一直是人工智能领域的重要话题，而人类语言的复杂性也给 NLP 布下了重重困难等待解决。长文本的智能解析就是颇具挑战性的任务，如何从纷繁多变、信息量庞杂的冗长文本中获取关键信息，一直是文本领域难题。随着深度学习的热潮来临，有许多新方法来到了 NLP 领域，给相关任务带来了更多优秀成果，也给大家带来了更多应用和想象的空间。

此次比赛提供了一批长文本数据和分类信息，我们需要构建文本分类模型，实现精准分类。

### 赛题数据

这是一个文本分类的任务，并且针对的是长文本，数据给定了长文本经过分词后的结果，并且数据进行了脱敏处理，无法知道原文本是什么内容，数据特征列如下：

- word_seg：str，文本分词后的每一个词，用空格间隔，一个词用一个数字代表
- article：str，文本中的每一个字，同样用空格间隔，一个数字代表一个字。从这里可以看出文本应该是中文，存在字和词的信息。
- class：int，文本所属的类别（共19个类）



## 解决方案

### 特征提取

文本信息不能作为分类模型的输入，我们必须要将文本的词信息转成数字信息，常用的方法有onehot，tf-idf，LDA，word2vec，在本次比赛中，我采用了tf-idf和word2vec这两种特征提取方法

sklearn中有TfidfVectorizer的工具可以直接将文本信息转成t-fidf矩阵，比较重要的参数如下

- ngram-range：tuple，ngram是指将多个词看成一个词的一种做法，这样做可以将词序考虑进去，而不是将每个词都看成是独立的。
- min_df, max_df：int/double，这两个参数都是限定哪些词需要考虑的，出现太多次或者太少次都需要被过滤。
- use_idf：默认为true
- smooth_idf：默认为true防止分母为0。

```python
vec = TfidfVectorizer(ngram_range=(1,3),min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
x = vec.fit_transform(x)
```




### 单模型

#### 线性模型

长文本提取出来的tf-idf矩阵维度达到了百万级，远远大于训练集的样本数，这时候选取线性模型的分类效果和训练时间都会比较好，因此我选择了逻辑回归分类器和线性svm作为基本的分类器，并且通过交叉验证和网格搜索找到较好的参数设置。

#### 深度学习模型

- fasttext是Facebook提出的一个快速文本分类的模型，本质上是一个线性模型，因为它的网络没有用到激活函数，而fasttext重要的一点是用到了character-ngram的trick，但是在这个任务中， 由于数据脱敏了，所以这个trick就没有太大的作用

  ```python
  clf = fasttext.supervised(input_file = 'fasttext_trainset.txt', output='fasttext_model', label_prefix = '__label__', bucket = 2000000, word_ngrams = 1, ws = 20, epoch=40, silent = 0)
  ```

- CNN模型用于文本分类，主要是将文本拼成一个矩阵，然后进行卷积操作，网络结构如下

  ```python
  input_layer = Input(shape = (MAX_WORD_NUM,), dtype = 'int32', name = 'input')
  embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length= MAX_WORD_NUM, trainable = True)(input_layer)
  
  conv_output = []
  for filter_ in [1,2,3,4,5,6]:
      conv = Conv1D(256, filter_, padding='same')(embedding_layer)
      conv = Activation('relu')(conv)
      conv = GlobalMaxPool1D()(conv)
      conv_output.append(conv)
  
  conv_output = concatenate(conv_output)
  full_connected_layer = Dense(256)(conv_output)
  full_connected_layer = Dropout(0.3)(full_connected_layer)
  full_connected_layer = Activation('relu')(full_connected_layer)
  full_connected_layer = Dense(72)(full_connected_layer)
  full_connected_layer = Activation('relu')(full_connected_layer)
  full_connected_layer = Dense(19)(full_connected_layer)
  ```

- bi-lstm用于文本分类也很常见，lstm能够解决RNN梯度消失的问题，并且能够捕获文本长时间的依赖关系，网络结构如下

  ```python
  input_layer = Input(shape = (MAX_WORD_NUM, ), dtype='int32')
  embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=MAX_WORD_NUM, trainable = True)(input_layer)
  x = Dropout(0.3)(embedding_layer)
  x = LSTM(units=128, return_sequences= True)(embedding_layer)
  x = Dropout(0.3)(x)
  x = Flatten()(x)
  x = Dense(units=19)(x)
  ```

  ### 模型融合

  单模型效果最好的是几个线性模型，深度模型的效果都一般般，但通过最后的stacking，使得整体模型的效果有了一定的提升，stacking过程分5折
  - 第一层
    - 逻辑回归，搭配不同的solver
    - linear svc
    - fasttext
    - lstm
    - cnn
  - 第二层：lgb





<br>