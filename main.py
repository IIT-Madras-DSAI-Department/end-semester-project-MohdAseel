import pandas as pd
import numpy as np
import time
from algorithms import MyStandardScaler, MyPCA, MySoftmaxRegression, KNearestNeighbors


TRAIN_DATA_PATH = "MNIST_train.csv"
VAL_DATA_PATH = "MNIST_validation.csv"
# TEST_DATA_PATH = 'MNIST_test.csv'

train_data = pd.read_csv(TRAIN_DATA_PATH)
val_data = pd.read_csv(VAL_DATA_PATH)
# test_data = pd.read_csv(TEST_DATA_PATH)

train_data_x = train_data.drop(columns=["label"])
train_data_y = train_data["label"]
val_data_x = val_data.drop(columns=["label"])
val_data_y = val_data["label"]

scaler = MyStandardScaler()
scaled_train_data_x = scaler.fit_transform(train_data_x)
scaled_val_data_x = scaler.transform(val_data_x)

pca98 = MyPCA(0.98)
pca98.fit(train_data_x)
pca95 = MyPCA(0.95)
pca95.fit(train_data_x)
pca90 = MyPCA(0.90)
pca90.fit(train_data_x)
pca98_train_img = pd.DataFrame(np.real(pca98.transform(train_data_x)))
pca98_test_img = pd.DataFrame(np.real(pca98.transform(val_data_x)))
pca95_train_img = pd.DataFrame(np.real(pca95.transform(train_data_x)))
pca95_test_img = pd.DataFrame(np.real(pca95.transform(val_data_x)))
pca90_train_img = pd.DataFrame(np.real(pca90.transform(train_data_x)))
pca90_test_img = pd.DataFrame(np.real(pca90.transform(val_data_x)))
pca_dict = [
    {
        "object": pca98,
        "train_img": pca98_train_img,
        "test_img": pca98_test_img,
        "n_components": pca98.n_components_,
        "knn": 6,
    },
    {
        "object": pca95,
        "train_img": pca95_train_img,
        "test_img": pca95_test_img,
        "n_components": pca95.n_components_,
        "knn": 4,
    },
    {
        "object": pca90,
        "train_img": pca90_train_img,
        "test_img": pca90_test_img,
        "n_components": pca90.n_components_,
        "knn": 6,
    },
]
print([x["n_components"] for x in pca_dict])


## from analyzing my models on different PCA components, I found these to best models and params
softmax_accuracy = {}

for dict in pca_dict:
    print(f"PCA with {dict['n_components']} components:")
    start_time = time.time()
    model = MySoftmaxRegression(
        n_classes=10, n_features=dict["n_components"], learning_rate=0.01
    )
    model.fit(dict["train_img"].values, train_data_y.values, epochs=200)
    train_acc = model.evaluate(dict["train_img"].values, train_data_y.values)
    val_acc = model.evaluate(dict["test_img"].values, val_data_y.values)
    softmax_accuracy[dict["n_components"]] = (train_acc, val_acc)
    print(f"Training Accuracy: {train_acc*100:.2f}%")
    print(f"Validation Accuracy: {val_acc*100:.2f}%\n")
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds\n")

model_lr_1 = MySoftmaxRegression(
    n_classes=10, n_features=pca95.n_components_, learning_rate=0.1
)
model_lr_1.fit(pca95_train_img.values, train_data_y.values, epochs=200)

model_lr_2 = MySoftmaxRegression(
    n_classes=10, n_features=pca98.n_components_, learning_rate=0.1
)
model_lr_2.fit(pca98_train_img.values, train_data_y.values, epochs=200)

model_knn_1 = KNearestNeighbors(k=6)
model_knn_1.fit(pca90_train_img, train_data_y.values)

model_knn_2 = KNearestNeighbors(k=4)
model_knn_2.fit(pca95_train_img, train_data_y.values)


predict_lr_1 = model_lr_1.predict(pca95_test_img.values)
predict_lr_2 = model_lr_2.predict(pca98_test_img.values)


predict_knn_1 = model_knn_1.predict(pca90_test_img.values)
predict_knn_2 = model_knn_2.predict(pca95_test_img.values)

df_predictions = pd.DataFrame(
    {
        "Softmax_PCA95": predict_lr_1,
        "Softmax_PCA98": predict_lr_2,
        "KNN_PCA90": predict_knn_1,
        "KNN_PCA95": predict_knn_2,
    }
)

df_predictions["predicted"] = df_predictions.mode(axis=1)[0]
df_predictions["actual"] = val_data_y.values

accuracy = np.sum(df_predictions["predicted"] == df_predictions["actual"]) / len(
    df_predictions
)
print(f"Ensembled Model Accuracy: {accuracy*100:.2f}%")
