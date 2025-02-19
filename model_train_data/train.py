
from data_preprocessing import prepare_data
from model_building import build_model
from trainning import train_model
from evaluation import evaluate_model
from visualization import plot_results


image_size = 150
image_channel = 3
batch_size = 32


train_generator, val_generator = prepare_data(image_size, batch_size)

model = build_model(image_size, image_channel)


model, history = train_model(model, train_generator, val_generator)

evaluate_model(model, train_generator, val_generator)


plot_results(history)


model.save('cat_dog_model.h5')
