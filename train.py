"""Pretraining, supervised fine-tuning, LoRA, and distillation."""

from trainer.launcher import launch


if __name__ == '__main__':
    launch({
        'pretrain': 'train_pretrain.py',
        'sft': 'train_full_sft.py',
        'lora': 'train_lora.py',
        'distill': 'train_distillation.py',
    }, 'MiniLLM supervised training')
