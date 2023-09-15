from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class GlobalWeight(Hook):
    def __init__(
            self,
            max_value=2,
    ):
        self.max_value = max_value

    def after_train_iter(self, runner):
        curr_step = runner.iter
        max_step = runner.max_iters
        if is_module_wrapper(runner.model):
            runner.model.module.global_weight = curr_step / max_step * self.max_value
        else:
            runner.model.global_weight = curr_step / max_step * self.max_value


@HOOKS.register_module()
class GlobalWeightStep(Hook):
    def __init__(
            self,
            value=1,
            step=15000,
    ):
        self.value = value
        self.step = step
        self.flag = 0

    def after_train_iter(self, runner):
        if self.flag:
            return
        curr_step = runner.iter
        if self.step <= curr_step:
            if is_module_wrapper(runner.model):
                runner.model.module.global_weight = self.value
            else:
                runner.model.global_weight = self.value
            self.flag = 1
