# Member 2 Report Notes

## Support Vector Machine

A Support Vector Machine is a supervised machine-learning algorithm that
finds a decision boundary that maximizes the margin between classes. Since
Human Activity Recognition contains six activity classes, the SVM performs
multiclass classification internally using multiple binary classification
problems.

The model was trained using the 561 engineered smartphone sensor features.
The features had already been standardized by Member 1 using a
`StandardScaler` fitted only on the training set. Therefore, no additional
normalization was applied by Member 2.

## Baseline model

The baseline SVM used:

- Kernel: RBF
- C: 1.0
- Gamma: scale

## Hyperparameter tuning

GridSearchCV was used with five-fold cross-validation. Model selection was
based on Macro F1 because every activity class should receive equal
importance.

The following values were tested:

- Kernel: linear and RBF
- C: 0.1, 1, and 10
- RBF gamma: scale and auto

## Evaluation

The final model was evaluated on the untouched validation set using:

- Accuracy
- Macro precision
- Macro recall
- Macro F1
- Per-class classification report
- Confusion matrix

## Result paragraph template

The default RBF SVM achieved a validation Macro F1-score of **[INSERT]**.
After hyperparameter tuning, the best configuration was **[INSERT BEST
PARAMETERS]**, which achieved a validation Macro F1-score of **[INSERT]**.
The tuned SVM was compared with Member 1's tuned Decision Tree, which
achieved a Macro F1-score of 0.9399. The SVM performed **[BETTER/WORSE]** by
**[INSERT DIFFERENCE]**. The confusion matrix showed that the model was
particularly effective at recognizing **[INSERT CLASSES]**, while most
remaining confusion occurred between **[INSERT CONFUSED CLASSES]**.

## Why Macro F1 was used

Accuracy can hide poor performance in individual classes. Macro F1
calculates the F1-score separately for every class and then takes the
unweighted average. Therefore, all six activities contribute equally to
the final evaluation score.
