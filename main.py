import Data
import Model

if __name__ == '__main__':
    data = Data.CategoricalPWM(
        ratio=[0, 0.25, 0.5, 0.75, 1],
        period=20,
        cycle=5
    ).make_data(
        num=100,
        ratio=[0.6, 0.3]
    )
    model = Model.CategoricalLSTM(
        data_set=data,
        epoch=10,
        learning_rate=0.01,
        lstm_size=100
    )
    model.build_model()

    history = model.fit(
        data.x_train, data.y_train,
        batch_size=1,
        epochs=model.epoch,
        validation_data=(data.x_val, data.y_val)
    )

    loss, mse, accuracy = model.evaluate(data.x_test, data.y_test)
    print(loss, mse, accuracy)
