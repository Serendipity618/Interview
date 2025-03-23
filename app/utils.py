def training_step(model, batch, optimizer, criterion_a, criterion_b, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        input_ids, attention_mask, labels_a, labels_b = batch
        logits_a, logits_b = model(input_ids, attention_mask)

        # Compute multi-task losses
        loss_a = criterion_a(logits_a, labels_a)
        loss_b = criterion_b(logits_b, labels_b)
        total_loss = loss_a + loss_b

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss.item():.4f}")

    return total_loss.item()
