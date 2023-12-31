# EmoteML

This is an implementation of an attention-based model that analyzes speech-to-text data from Twitch TV live chat to perform sentiment analysis and emote prediction.
[Paper explaining this project](Sentiment_Analysis_with_Emotes_using_Attention_Based_Neural_Networks.pdf)
#### Data Acquiring and Processing
Due to various difficulties dealing with speech-to-text, subtitles from Youtube were used as the input corpus. Subtitles of five YouTube videos of past streams of the famous streamer Félix Lengyel (xQc) and Twitch live chat text data were obtained for training data generation.

gen_data.py processes the Twitch chat data to determine which emote was frequently sent by the viewers during that period 
of time to determine the target emote. Then we traced back the subtitles before that period of time in the srt file to determine the input text for the training data.
emotes.txt contains the 14 Twitch emotes used for the sentiment analysis. 

cv_split.py merges all the training sets into a single training set and performs a cross-validation split with 4 folds.

Here is an example of the training data:

"Today I received an insane NFT sponsor offer and the amounts are astronomical one", "BatChest"

#### Model and Training
lib/transformer.py contains the model architecture which is inspired by https://www.tensorflow.org/text/tutorials/transformer.

Pretrained Word2vec embeddings were downloaded from https://wikipedia2vec.github.io/wikipedia2vec/pretrained/.

emote_class.py performs the training and predicts the test results from the soft voting of the 4 cross-validation splits.

#### Results
test_result.csv shows the 30 test samples and the target emote predicted by the model. For further details please find in the discussion section in the [Paper](Sentiment_Analysis_with_Emotes_using_Attention_Based_Neural_Networks.pdf).

#### Future Ideas
- [ ] implement other models such as FastFormer
