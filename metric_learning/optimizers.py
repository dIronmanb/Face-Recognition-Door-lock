import torch.optim as optimizer

def get_optimizer(optimizer_name, model, learning_rate):
    if optimizer_name.lower() == "adam":
        return optimizer.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == "sgd":
        return optimizer.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == "rmsprop":
        return optimizer.RMSprop(model.parameters(), lr=learning_rate)