import torch, os, random
import torch.nn as nn 
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from layers import init_model
from eval_utils import Evaluator

class ExpModule(pl.LightningModule):

    def __init__(self, args):
        super().__init__()

        # compute weights (only if explicitly requested via --use_weighted_loss):
        if args.task == 'bone_class' and getattr(args, 'use_weighted_loss', False) and hasattr(args, 'metadata_csv'):
            import pandas as pd
            import numpy as np

            # Load training counts directly from the source CSV file.
            # Use the FULL set of class ids actually present anywhere in the csv
            # (not a hardcoded range(9)) so this stays correct if the label set changes.
            df = pd.read_csv(args.metadata_csv)
            train_df = df[df['split'] == 'train']
            all_classes = sorted(df['label'].unique().tolist())
            num_classes = len(all_classes)

            # Count occurrences of each class label in TRAIN split only
            counts = train_df['label'].value_counts().to_dict()
            class_counts = [counts.get(c, 1) for c in all_classes]  # default to 1 to avoid div-by-zero

            # Inverse frequency weights
            total_samples = sum(class_counts)
            class_weights = [total_samples / (num_classes * count) for count in class_counts]

            # Normalize so the minimum weight is 1.0 (avoid over-damping the majority class)
            min_weight = min(class_weights)
            class_weights = [w / min_weight for w in class_weights]

            print(f"[use_weighted_loss] class counts (train): {dict(zip(all_classes, class_counts))}")
            print(f"[use_weighted_loss] class weights: {[round(w,3) for w in class_weights]}")

            # Save it to arguments so layers.py can read it
            args.class_weights = class_weights
        else:
            args.class_weights = None

        ###

        self.save_hyperparameters(args)

        self.model = init_model(self.hparams)

        self.scorer = Evaluator(self.hparams.task)

        self.batch_size = self.hparams.batch_size
        self.learning_rate = self.hparams.lr
        self.fusion = True if self.hparams.modality == 'both' else False
        self.validation_step_outputs = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.hparams.wd)
        return optimizer

    # def forward(self, input, target, mask):

    #     if isinstance(input, tuple):
    #         x1,x2 = input
    #         m1,m2 = mask
    #         logits, loss = self.model(x1,x2, target, m1,m2)
    #     else:
    #         logits, loss = self.model(input, target, mask)

    #     return logits, loss 
    def forward(self, input, target, mask):
        # Check if input arrived packed as a list or tuple [images, notes]
        if isinstance(input, (tuple, list)) and len(input) == 2:
            x_ts, x_txt = input[0], input[1]
            
            # Unpack masks if they arrived as a pair (masks_img, masks_txt)
            if isinstance(mask, (tuple, list)) and len(mask) == 2:
                ts_attn_mask, txt_attn_mask = mask[0], mask[1]
            else:
                ts_attn_mask, txt_attn_mask = None, None
                
            # Pass them cleanly using explicit keyword arguments
            logits, loss = self.model(
                x_ts=x_ts, 
                x_txt=x_txt, 
                labels=target, 
                ts_attn_mask=ts_attn_mask, 
                txt_attn_mask=txt_attn_mask
            )
        else:
            # Fallback path for other tasks
            logits, loss = self.model(input, target, mask)

        return logits, loss

    def step(self, batch, train_stage=False):
        input, target, mask, stay = batch 

        logits, loss = self.forward(input, target, mask)

        return loss, (logits, target)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        input, target, mask, stay = batch 

        if self.fusion:
            if hasattr(self.model.fusion, 'train_stage'): self.model.fusion.train_stage=False 
        logits, loss = self.forward(input, target, mask)

        return logits, target, stay

    def training_step(self, batch, batch_idx):
        loss, (logits, y) = self.step(batch)
        self.log('train_loss', loss, on_epoch=True, batch_size=self.batch_size)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss, (logits, y) = self.step(batch)
        if self.fusion:
            if hasattr(self.model.fusion, 'train_stage'): self.model.fusion.train_stage=False 
        self.log('valid_loss', loss, batch_size=self.batch_size)
        val_output = {'loss': loss, 'logits': logits.detach(), 'y': y}
        self.validation_step_outputs.append(val_output)
        return val_output

    def test_step(self, batch, batch_idx):
        loss, (logits, y) = self.step(batch)
        if self.fusion:
            if hasattr(self.model.fusion, 'train_stage'): self.model.fusion.train_stage=False 
        self.log('test_loss', loss, on_epoch=True, batch_size=self.batch_size)
        return {'loss': loss, 'logits':logits, 'y':y}
    
    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        if len(outputs) == 0:
            return
            
        all_logits, all_y = [], []
        for d in outputs:
            all_logits.append(d['logits'])
            all_y.append(d['y'])

        logits = torch.cat(all_logits).detach().cpu()
        y = torch.cat(all_y).cpu()
        
        scores, line = self.scorer.eval_all_scores(logits, y)
        # score = scores[self.scorer.score_main]
        if self.hparams.task == 'bone_class':
            # Check your printout headers: if it prints as 'MacroF1', use 'MacroF1'. 
            # If it throws a KeyError, change this string to 'f1_macro' or 'macrof1'.
            score = scores.get('MacroF1', scores.get('f1_macro', scores.get('macrof1')))
        else:
            score = scores[self.scorer.score_main]
        self.log('valid_score', score)

        if not self.hparams.silent:
            print(line)
            
        self.validation_step_outputs.clear()

    
def init_trainer(args):

    RESULT_DIR = args.output_dir

    version = str(random.randint(0, 2000))
    log_dir = os.path.join(RESULT_DIR, 'log', args.modality)
    model_dir = os.path.join(RESULT_DIR, 'model', args.modality, args.task, version)

    metric = 'valid_score'
    mode = 'max'

    logger = TensorBoardLogger(save_dir=log_dir, version=version, name=args.task)
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir, filename='{epoch}-{%s:.3f}' % metric, 
        monitor=metric, mode=mode, save_weights_only=True, 
    )
    early_stopping = EarlyStopping(
        monitor=metric, min_delta=0., patience=args.patience,
        verbose=False, mode=mode
    )

    if args.device == -2:
        accelerator = "auto"
        devices = "auto"
    else:
        if torch.cuda.is_available():
            accelerator = "gpu"
            devices = [args.device]
        elif torch.backends.mps.is_available():
            accelerator = "mps"
            devices = 1
        else:
            accelerator = "cpu"
            devices = "auto"

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        accelerator=accelerator,
        devices=devices,
        num_sanity_val_steps=0,
        max_epochs=args.epochs if not args.debug else 2, 
        enable_progress_bar=not args.silent,
    )
    return trainer