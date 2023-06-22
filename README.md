# EmoteML
A project analyzing Twitch TV live chat data

This is the code to perform sentiment analysis and emote prediction on the speech-to-text data from Twitch TV live chat.

#### Data Acquiring and Processing
The subtitles of five YouTube videos of past streams and Twitch chat text data were obtained to generate training data. 
gen_data.py processes the Twitch chat data to determine which emote was frequently sent by the viewers during that period 
of time to determine the target emote. 
Then we traced back the subtitles before that period of time in the srt file to determine the input text for the training data.
emotes.txt contains the 14 Twitch emotes used for the sentiment analysis. 

cv_split.py merges all the training sets into a single training set, and performs a cross-validation split with 4 folds.

Here is an example of the training data:

"today i received an insane nft sponsor offer and the amounts are astronomical one", "BatChest"

#### Model and Training
lib/transformer.py contains the model architecture which is inspired by https://www.tensorflow.org/text/tutorials/transformer.

Pretrained Word2vec embeddings were downloaded from https://wikipedia2vec.github.io/wikipedia2vec/pretrained/.
Gensim package (https://radimrehurek.com/gensim/) was used to load the pre-trained Word2vec embeddings.

emote_class.py performs the training and predicts the test results from the soft voting of the 4 cross-validation splits.

#### Results
test_result.csv shows the 30 test samples and the target emote predicted by the model.
