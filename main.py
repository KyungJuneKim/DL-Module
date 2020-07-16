import Data
import Model

if __name__ == '__main__':
    data = Data.PWM()
    model = Model.CategoricalLSTM(
        epoch=10,
        learning_rate=0.01,
        data_set=data,
        lstm_size=100
    ).build_model()
