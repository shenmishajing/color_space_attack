import torch

# epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]


def fgsm(model, inputs, data_samples, eps):
    inputs.requires_grad = True
    with torch.enable_grad():
        loss, _ = model.parse_losses(model(inputs, data_samples, mode="loss"))
    loss.backward()
    perturbed_inputs = torch.clamp(inputs + eps * inputs.grad.sign(), 0, 1)

    # Re-classify the perturbed image
    outputs = model(perturbed_inputs, data_samples, mode="predict")

    return (
        torch.cat([output.gt_label != output.pred_label for output in outputs]),
        perturbed_inputs,
    )
