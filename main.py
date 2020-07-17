import Data
import Model

if __name__ == '__main__':
    data = Data.PWM()
    model = Model.CategoricalLSTM(
        data_set=data,
        epoch=10,
        learning_rate=0.01,
        lstm_size=100
    )
