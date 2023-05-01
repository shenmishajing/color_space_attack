import torch
from mmlab_lightning.models import MMLabModelAdapter

from .attack import attack_methods


class ColorSpaceAttackModel(MMLabModelAdapter):
    def __init__(
        self,
        predict_tasks=("attack",),
        attack_cfg: dict = {"func": "fgsm", "args": {"eps": 0.05}},
        *args,
        **kwargs
    ):
        super().__init__(predict_tasks=predict_tasks, *args, **kwargs)
        self.attack_cfg = attack_cfg
        self.predict_attack_outputs = []

    def predict_attack(self, *args, inputs, predict_outputs, output_path, **kwargs):
        attack_result = torch.cat(
            [output.gt_label != output.pred_label for output in predict_outputs]
        )

        indices = (~attack_result).nonzero()

        res, perturbed_inputs = attack_methods[self.attack_cfg["func"]](
            self.model,
            inputs[indices],
            [predict_outputs[i] for i in indices],
            **self.attack_cfg["args"]
        )
        attack_result[indices] = res

        self.predict_attack_outputs.append(attack_result)

    def on_predict_epoch_end(self) -> None:
        attack_results = torch.cat(self.predict_attack_outputs)
        attack_acc = attack_results.sum().item() / len(attack_results)
        self.log_dict(
            self.flatten_dict({"attack_acc": attack_acc}, "predict"), sync_dist=True
        )
        return super().on_predict_epoch_end()
