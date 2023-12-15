# UNIL_IWC

## Foreword

When the project was initially presented to us, we felt disoriented, uncertain about where to begin. The project seemed both simple and complex: simple in its concise and direct instruction, but complex due to the lack of foundational elements to build upon for a start.

In this project, we sourced our information from various avenues, such as articles, discussion forums, and similar projects. For its implementation, we utilized Visual Studio Code with the Copilot extension, GitHub, ChatGPT, and Google Colab.

Visual Studio Code was our primary tool for developing our models, aided by ChatGPT and Copilot. We shared our progress through a GitHub repository. Although the installation process was lengthy and fraught with errors, once operational, our productivity and efficiency significantly improved. The mere act of pushing the code allowed the other team member to easily pull it.

However, training our model on Visual Studio Code proved too slow, leading us to switch to Google Colab and opt for the T4 runtime type, which was considerably faster. We even subscribed to the paid version to further accelerate the process.

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




