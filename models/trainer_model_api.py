
class Trainer_Model_API:
    """
    如果model是module, 那么trainer通过调用self.model.module if isinstance(self.model, DDP) else self.model 获得module

    # Trainer调用model的:
    1. optimize()
    2. get_lr_group_dicts()
    3. state_dict()
        input: None
    4. optimize_state_dict()

    5. load_state_dict()
        input: checkpoint['model'], strict=True

    6. load_optimize_state_dict()
    7. train()
    7. eval()
    8. self.model() # forward
    9. sample()

    """
    pass