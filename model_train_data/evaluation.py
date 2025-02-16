
def evaluate_model(model, train_generator, val_generator):
    train_loss, train_acc = model.evaluate(train_generator)
    print(f'Train Accuracy: {train_acc*100:.2f}%')
    print(f'Train Loss: {train_loss:.4f}')
    
    val_loss, val_acc = model.evaluate(val_generator)
    print(f'Validation Accuracy: {val_acc*100:.2f}%')
    print(f'Validation Loss: {val_loss:.4f}')
