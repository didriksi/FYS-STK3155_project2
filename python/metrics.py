import numpy as np

def MSE(y, y_tilde):
    """MSE"""
    return np.mean(np.square(y - y_tilde), axis=0)

def R_2(y, y_tilde):
    """R2"""
    return 1 - np.sum(np.square(y - y_tilde)) / np.sum(np.square(y - np.mean(y)))

def R_2_loss(y, y_tilde):
    """1 - R2"""
    return 1 - np.sum(np.square(y - y_tilde)) / np.sum(np.square(y - np.mean(y)))

def cross_entropy(y, y_tilde):
    """Cross entropy"""
    return - np.mean(y * np.log(y_tilde + 1e-12))

def accuracy(y, y_tilde):
    """Accuracy"""
    targ = np.where(y == 1)[1]
    pred = np.argmax(y_tilde, axis=1)
    return np.mean(targ == pred)

def accuracy_loss(y, y_tilde):
    """1 - accuracy"""
    targ = np.where(y == 1)[1]
    pred = np.argmax(y_tilde, axis=1)
    return 1 - np.mean(targ == pred)


def classification_accuracy(subplots, data):
    """Makes a confusion table, and prints out accuracy and other relevant metrics for classification problems.

    subplots:  (models, sgd kwargs)
                Each tuples first element is a list of models to be tested together.
    data:       dict
                Big dictionary with training, testing and validation data.
    """
    best_models = []

    all_models = [models for models, _ in subplots]
    true_train = np.where(data['y_train'] == 1)[1]
    true_val = np.where(data['y_validate'] == 1)[1]
    true_test = np.where(data['y_test'] == 1)[1]
    for j, models in enumerate(all_models):
        model_scores = []

        for k, model in enumerate(models):
            predict_val = np.argmax(model.predict(data['x_validate']), axis=1)

            accuracy = np.mean(predict_val == true_val)
            model_scores.append(accuracy)
        max_score = max(model_scores)

        best_models.append([j, model_scores.index(max_score)])

    for index in best_models:
        model = all_models[index[0]][index[1]]
        predict_train = model.class_predict(data['x_train'])
        predict_val = model.class_predict(data['x_validate'])
        predict_test = model.class_predict(data['x_test'])

        print(f"Model assessment of {model.name}.")
        print(f"---{textify_dict(model.property_dict)}---")
        print("Test data", model.name)
        print(classification_report(true_test, predict_test))
        print("Validation data")
        print(classification_report(true_val, predict_val))
        cf = confusion_matrix(true_val, predict_val)
        plt.clf()
        sns.heatmap(cf, annot=True, cbar=False)
        plt.title(f"Confusion matrix - {model.name} \n {len(data['x_validate'])} samples of validation data")
        plt.savefig(f"../plots/classification_cf_{model.name}")
        plt.clf()