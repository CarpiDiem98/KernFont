from tqdm import tqdm


def run_train(model, train_loader, criterizion, optimizer, experiment, epochs, device):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_accuracy = 0
        for _, data in tqdm(enumerate(train_loader)):
            sx = data[0].to(device)
            dx = data[1].to(device)
            kern_value = data[2].to(device)
            optimizer.zero_grad()
            output = model(sx, dx)
            loss = criterizion(output, kern_value)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * sx.size(0)  # Fixing size calculation
            train_accuracy += criterizion(output, kern_value).item() * sx.size(0)

        # Logging after each epoch
        train_loss /= len(train_loader.dataset)
        train_accuracy = (
            train_accuracy / len(train_loader.dataset) * 100
        )  # Convert to percentage
        experiment.log_metrics(
            {"loss": train_loss, "accuracy": train_accuracy},
            step=epoch + 1,
        )
        print(
            f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%"
        )
