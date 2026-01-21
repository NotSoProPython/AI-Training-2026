import tensorflow as tf

def CustomLoss(y_true, y_pred):

    for i in range(128):

        print("- - - Answer - - -")
        print(y_true[i])

        for j in range(10):

            print("- - - Predicted - - -")
            print(y_pred[i][j])

    return 1