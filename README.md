The first time I ran the project, I used alot of the code from handwriting.py in the lecture notes in my get_model function. My accuracy was .05 at the start and it hardly improved throughout the 10 epochs, and my loss was initially 8.3 and finished at 3.5. 

I initially experimented by duplicating the existing hidden layer, however this had little impact. Additionally, I tried increasing the pool size dimensions from (2, 2) to (3, 3) but that did not help. 

After more experimenting to improve accuracy, I found that reducing the dropout rate from .5 to .2 and increasing the filter count in my convolutional layer from 32 to 60 improved the outcome significantly. After these changes my accuracy was .92 and my loss was .5 by the end of the ten epochs. 

By the end, I found that adding an additional convolutional layer and setting the filter size to 80, dropout rate to .4, and hidden layer units to 200, gave me the best outcome of accuracy .97 and loss .10 by the end of the ten epochs. 
![alt text](https://file%2B.vscode-resource.vscode-cdn.net/Users/halseytyler/Downloads/CSCI%20E-80/traffic/training_curves.png?version%3D1766972025438)
![alt text](confusion_matrix.png)

https://www.kaggle.com/datasets/bhaveshmittal/melanoma-cancer-dataset/data