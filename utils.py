def calc_and_print_metrics(cm, isPrinting=True):
    """
    Given a confusion matrix (`cm`), calculates, prints and returns metrics
    such as precision, recall, f1, accuracy and specificity.
    """
    if isPrinting: print('Confusion matrix\n', cm)

    TP = cm[0, 0]
    TN = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]
    if isPrinting: print(f"True Positives(TP) = {TP}")
    if isPrinting: print(f"True Negatives(TN) = {TN}")
    if isPrinting: print(f"False Positives(FP) = {FP}")
    if isPrinting: print(f"False Negatives(FN) = {FN}\n")

    # Precision
    precision = (TP)/(TP + FP)
    if isPrinting: print(f"Precision: {precision:.4f}")

    # Recall
    recall = (TP)/(TP + FN)
    if isPrinting: print(f"Recall: {recall:.4f}")

    # F1
    f1 = (2 * precision *recall)/(precision + recall)
    if isPrinting: print(f"F1: {f1:.4f}")

    # Accuracy
    accuracy = (TP + TN)/float(TP + TN + FP + FN)
    if isPrinting: print(f"Accuracy: {accuracy:.4f}")

    # Specificity
    specificity = (TN)/(TN + FP)
    if isPrinting: print(f"Specificity: {specificity:.4f}")

    return {
        "cm": cm,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "specificity": specificity
    }

def graph(dictionary, isKNN=True):
    """
    Plots 3 graphs
    """
    colours = {3:"red", 4:"orange", 5:"green", 6:"blue", 7:"purple", 0: "red", 0.1: "green", 0.2: "blue"}
    kfold_set = [x for x in range(5, 12)]
    mi_set = [0, 0.1, 0.2]
    if isKNN:
        knn_set = [x for x in range(3, 8)]
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 6), dpi=300)
        for i in range(len(mi_set)): #0
            mi = mi_set[i]
            for knn in knn_set: #3
                knn_accuracy = []
                knn_precision = []
                for kfold in kfold_set: #5
                    knn_accuracy.append(dictionary[kfold][mi][knn]['mean_accuracy'])
                    knn_precision.append(dictionary[kfold][mi][knn]['mean_precision'])

                # Graph and Annotate Accuracy
                axes[i].plot(kfold_set, knn_accuracy, marker='o',label=f"{knn}_knn_accuracy", color=colours[knn])
                axes[i].text(kfold_set[-1], knn_accuracy[-1], f"{knn}_knn_accuracy", {"color":colours[knn]})

                # Graph and Annotate Precision
                axes[i].plot(kfold_set, knn_precision, marker='o', label=f"{knn}_knn_precision", color=colours[knn])
                axes[i].text(kfold_set[-1], knn_precision[-1], f"{knn}_knn_precision", {"color":colours[knn]})

                axes[i].set_xlabel("K-fold")
                axes[i].set_ylabel("Precision/Accuracy")
                # plt.ylim((None,1))
                axes[i].set_title(f"Precision and Accuracy of KNN\nwith MI Threshold of {mi}")
                # axes[i].legend.set_text(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        fig.tight_layout()
        plt.savefig('knn_precision_and_accuracy.png', dpi=300)
    else:
        fig, axes = plt.subplots(dpi=300)
        for mi in mi_set:
            dt_accuracy = []
            dt_precision = []
            for k_fold in kfold_set:
                dt_accuracy.append(dictionary[k_fold][mi]['accuracy'])
                dt_precision.append(dictionary[k_fold][mi]['precision'])
            # Graph and Annotate Accuracy
            axes.plot(kfold_set, dt_accuracy, marker='o', label=f"{mi}_MI_accuracy", color=colours[mi])
            axes.text(kfold_set[-1], dt_accuracy[-1], f"{mi}_MI_accuracy", {"color":colours[mi]})

            # Graph and Annotate Precision
            axes.plot(kfold_set, dt_precision, marker='o', label=f"{mi}_MI_precision", color=colours[mi])
            axes.text(kfold_set[-1], dt_precision[-1], f"{mi}_MI_precision", {"color":colours[mi]})
        axes.set_xlabel("K-fold")
        axes.set_ylabel("Precision/Accuracy")
        axes.set_title(f"Precision and Accuracy \nof Decision Trees")
        plt.savefig('dt_precision_and_accuracy.png', dpi=300)
