# UNIL_IWC
#### Gaetan Stein and Laurent Sierro
## Foreword

When the project was initially presented to us, we felt disoriented, uncertain about where to begin. The project seemed both simple and complex: simple in its concise and direct instruction, but complex due to the lack of foundational elements to build upon for a start.

In this project, we sourced our information from various avenues, such as articles, discussion forums, and similar projects. For its implementation, we utilized Visual Studio Code with the Copilot extension, GitHub, ChatGPT, and Google Colab.

Visual Studio Code was our primary tool for developing our models, aided by ChatGPT and Copilot. We shared our progress through a GitHub repository. Although the installation process was lengthy and fraught with errors, once operational, our productivity and efficiency significantly improved. The mere act of pushing the code allowed the other team member to easily pull it.

However, training our model on Visual Studio Code proved too slow, leading us to switch to Google Colab and opt for the T4 runtime type, which was considerably faster. We even subscribed to the paid version to further accelerate the process.

Link to the presentation video : https://youtu.be/ihEi_XEGRV4

## Goals

The project asked us to predict the CEFR level of french sentences by training a model on given and labelled sentences. 

## Methodology
Here is the different steps we went through in order to reach the goal.

### Research on CEFR-level

First, we conducted some research on the internet to get a precise idea of what is "CEFR level". Maybe, knowing the criteria of classification would have helped us. In fact, we quickly found out that the criteria were blury and stopped our research here.

### CamemBERT

Because we waited about one and half week before starting to try out solutions, we had already heard at the time the name "CamemBERT". After a bit of a chitchat with our classmates, we found out the best lead to follow was this one. This is the reason why we didn't even try the logistic regression, kNN, Decision Tree and Random forests solutions as we knew this wouldn't lead us to the top of the leaderboard.
Our research started on CamemBERT to, first, understand how it works, second, how to use it. After several hours of "Die-and-Retry", we got a first result with an accuracy of .527 which told us that this was the right way to go.

### Hyper-parameters

We quickly found out that Camembert was quite "capricious". Its parameters were tricky to handle properly. 

#### Learning-Rate

If the lr is too big, the function that tries to minimize the train-loss won't be able to converge, and if it is too small, the training will be slow and it can converge to a local minimum. We, therefore, stayed in the range of 2e-05 to 6e-05 for our tests.

#### Epochs

To get the best accuracy without having an overfitted model was tricky. We started our journey with CamemBERT with 20 epochs which was way too much. At the end of our work, we stay in the range of 3 to 8 epochs.

#### Batch Size

The batch size influence the speed of converge if small but the amount of noise in the model updates too. We found out, by trying, that the optimal batch size were between 16 and 64. 

### Data augmentation

First, we tried implementing the length of the sentence as a new parameter. We thought that the longer a sentence is, the harder it would be. We found that adding the length didn't change the accuracy significatly. 
Second, we increased the number of sentence by translating the whole training dataset in Portugese and back to French using the deep-translator libraries. By doing so, we hoped that the translation process would add differences into the sentences without changing the overall level. After runing the model with the new dataset, we found ourselves in front an big overfittig problem. Indeed, sentences of low levels (A1, A1, B1) were usually translated perfectly back to french without any differences added. To resolve this problem, we put the .csv file on chatGPT and made a prompt to remove all sentences with a similarity level above 70%. 
After several tries, we found out that the model overfitted way less but we couldn't manage to beat our previous score of .584.

## Results

Before looking at the results, you can download our trained model with this link. You can use the code of "camemBERT_loader.py" to make prediction with it. Our trained model is avaible to download on the GitHub "Tag" tab. If it is not working, use the link bellow as a backup.

Model : https://www.swisstransfer.com/d/b67892b3-3a75-4120-ae4a-bd00debcc705 
<span style="color:red">expires on 17-01-2024</span>

As said above, we started which an accuracy of .527 which is not bad. After 22 submissions, we reached our overall best score of .584 accuracy on Kaggle. Sadly we couldn't manage to go further. 
Let's break down the metrics of our best run. 

| | Best Run | 
|:---------:|:---------:|
| Precision | 0.54 |  
| Recall    | 0.539  | 
| F1-score | 0.538 | 
| Accuracy* | 0.56 | 

*Accuracy of the model, not on Kaggle.

**Precision (0.54)** This indicates that 54% of the instances that the model predicted as positive are actually positive. 

**Recall (0.539)** Recall measures the proportion of actual positive instances that the model correctly identified. A recall of approximately 0.54 means that the model identifies about 54% of all actual positive instances.

**Accuracy (0.56)** This represents the proportion of total predictions (both positive and negative) that were correct. An accuracy of 0.56 indicates that the model correctly predicts 56% of the instances.

**F1-score** The F1-score is the harmonic mean of precision and recall, providing a single score that balances both metrics.

The confusion matrix is as followed :

![Image 1](https://github.com/Aztol/UNIL---Kaggle-DS-ML-competition/blob/main/images/confusion_matrix.png)

Knowing the different metrics presented above, we could think that the model's lack of precision makes it useless. But the confusion matrix shows the tendency of the model to predict adjacent CEFR level. The model fails to determine precisely the CEFR level. The CEFR level requirements being blury, we found that our model performed quite well.

## Conclusion

In conclusion, the journey through this project provided precious insights into the challenges and intricacies of natural language processing and machine learning . The decision of focusing on camemBERT proved fruitful despite an early steep learning curve. Overall, this project underlined the complexity of language modeling and laid solid fondations for the next semestre.