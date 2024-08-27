from typing import Dict, Optional
import os
import wandb

__all__ = ["set_wandb"]


def set_wandb(opt: Dict, local_rank: int = 0, force_mode: Optional[str] = None) -> str:
    if local_rank != 0:
        return ""

    # opt = opt
    save_dir =  opt['save_dir']

    wandb_mode = opt["wandb"]["mode"].lower()
    if force_mode is not None:
        wandb_mode = force_mode.lower()
    if wandb_mode not in ("online", "offline", "disabled"):
        raise ValueError(f"WandB mode {wandb_mode} invalid.")

    assert os.path.exists(save_dir)

    wandb_project = opt["wandb"]["project"]
    wandb_name = opt["wandb"]["name"]
    wandb_id = opt["wandb"].get("id", None)
    wandb_group = opt['wandb']['group']
    wandb_notes = opt["wandb"].get("notes", None)
    wandb_tags = [wandb_group, wandb_name]
    wandb_tags.extend(opt["wandb"]['tags'])

    wandb_name = f'{wandb_group}_{wandb_name}'

    wandb.init(
        project=wandb_project,
        name=wandb_name,
        dir=save_dir,
        resume="allow",
        mode=wandb_mode,
        id=wandb_id,
        notes=wandb_notes,
        tags=wandb_tags,
        config=opt,
        group=wandb_group
    )
    wandb_path = wandb.run.dir if (wandb_mode != "disabled") else save_dir
    return wandb_path


import logging
from termcolor import colored

def _highlight(code, filename):
    try:
        import pygments
    except ImportError:
        return code

    from pygments.lexers import Python3Lexer, YamlLexer
    from pygments.formatters import Terminal256Formatter

    lexer = Python3Lexer() if filename.endswith(".py") else YamlLexer()
    code = pygments.highlight(code, lexer, Terminal256Formatter(style="monokai"))
    return code

class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)
        # CRITICAL: 'CRITICAL',
        # ERROR: 'ERROR',
        # WARNING: 'WARNING',
        # INFO: 'INFO',
        # DEBUG: 'DEBUG',
        # NOTSET: 'NOTSET',
    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        message = record.message
        # message, asctime, name, filename = record.message, record.asctime, record.name, record.filename
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if (record.levelno == logging.WARNING) or (record.levelno == logging.ERROR) or (record.levelno == logging.CRITICAL):
            colored_message = colored(message, "red", attrs=["blink", "underline"])
        elif record.levelno == logging.DEBUG:
            colored_message = colored(message, "yellow", attrs=["blink", "underline"])
        else: # INFO/NOTSET
            colored_message = colored(message, "white")  
        return log + colored_message
        # TODO: 实现多卡log, 现在只有主进程可以log
        # 如果主题背景是白色的话 需要在 555 上加上一个特数split_symbol, 然后log.split(split_symbol)得到message之前的东西
        # In Python print statements, the escape sequence \x1b[0m is used to reset the text formatting to the default settings. Specifically, it is used for resetting text attributes like color, style, and background color to their default values.
    
def set_logging_file(output_dir, file_name, mode='a'):
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join(output_dir, file_name), mode=mode)
    formatter = _ColorfulFormatter(
        colored("[%(asctime)s %(name)s %(filename)s]: ", "green"), # 555
        datefmt="%m/%d %H:%M:%S",
        root_name=os.path.join(output_dir, file_name),
        abbrev_name=str('grey'),
    )
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(logging.DEBUG)