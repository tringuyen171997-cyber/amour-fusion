
import numpy as np, logging
from collections import defaultdict

import pytorch_lightning as pl

from data_utils import ExpDataModule
from eval_utils import Evaluator, evaluate_predict_output, _print_score_dicts
from modules import ExpModule, init_trainer

from options import args 

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%m/%d %I:%M:%S %p')

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def run_diagnostic_analysis(all_hadms, all_trues, all_preds, all_probs, class_names=None):
    """
    Executes a complete error evaluation post-training.
    """
    os.makedirs("./output", exist_ok=True)
    
    # 1. Print Text-Based Classification Report
    print("\n" + "="*50)
    print("         POST-TRAINING ERROR ANALYSIS          ")
    print("="*50)
    print(classification_report(all_trues, all_preds, target_names=class_names))
    
    # 2. Compute and Save Confusion Matrix Plot
    cm = confusion_matrix(all_trues, all_preds)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_percent, 
        annot=cm, # Displays raw prediction counts inside the squares
        fmt="d", 
        cmap="Blues",
        xticklabels=class_names if class_names else True,
        yticklabels=class_names if class_names else True
    )
    plt.ylabel('Actual Label (Ground Truth)')
    plt.xlabel('Predicted Label')
    plt.title('Normalized Confusion Matrix')
    
    matrix_path = './output/confusion_matrix_diagnostic.png'
    plt.savefig(matrix_path, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved confusion matrix visualization to: {matrix_path}")
    
    # 3. Export Spreadsheet of Confidently Wrong Predictions
    records = []
    for i in range(len(all_hadms)):
        t_label = all_trues[i]
        p_label = all_preds[i]
        
        if t_label != p_label:
            records.append({
                "hadm_id": all_hadms[i],
                "true_class": t_label,
                "predicted_class": p_label,
                "confidence_score": float(all_probs[i][p_label]) # How sure it was about the wrong answer
            })
            
    if records:
        df = pd.DataFrame(records).sort_values(by="confidence_score", ascending=False)
        csv_path = "./output/confident_failures_audit.csv"
        df.to_csv(csv_path, index=False)
        print(f"[✓] Exported {len(df)} misclassifications to: {csv_path}\n")
    else:
        print("[!] Amazing! No misclassifications found on this split.")

def main_run(args, dm=None, seed=None):

    if seed is not None:
        pl.seed_everything(seed)
        logging.info(f'Set seed to {seed}')

    if dm is None:
        dm = ExpDataModule(args)
        dm.setup()
    model = ExpModule(args)

    if args.task != 'bone_class' and args.modality != 'text' and getattr(args, 'baseline_type', None) != 'early':
        model.model.grud._init_x_mean(dm.X_mean)


    trainer = init_trainer(args)
    trainer.fit(model, dm)

    output = trainer.predict(model, dataloaders=dm.test_dataloader(), ckpt_path='best')
    main_score, scores = evaluate_predict_output(output, args.task)
    # -------------------------------------------------------------
    try:
        all_hadms = []
        all_trues = []
        all_preds = []
        all_probs = []

        for batch_out in output:
            # Handle unpack structures depending on your predict_step return mapping
            if isinstance(batch_out, dict):
                logits = batch_out['logits']
                labels = batch_out['labels']
                hadms = batch_out.get('hadm_id', batch_out.get('hadms', []))
            elif isinstance(batch_out, (tuple, list)):
                logits, labels = batch_out[0], batch_out[1]
                hadms = batch_out[2] if len(batch_out) > 2 else range(len(labels))
            else:
                continue

            # Compute prediction probabilities
            probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
            preds = np.argmax(probs, axis=-1)
            trues = torch.tensor(labels).numpy()

            all_hadms.extend(hadms)
            all_trues.extend(trues)
            all_preds.extend(preds)
            all_probs.extend(probs)

        # Run diagnostic if samples were parsed successfully
        if len(all_trues) > 0:
            run_diagnostic_analysis(
                all_hadms=all_hadms,
                all_trues=np.array(all_trues),
                all_preds=np.array(all_preds),
                all_probs=np.vstack(all_probs)
                # Option: pass class_names=['Normal', 'Fracture', ...] to match your data indices
            )
    except Exception as e:
        logging.error(f"Failed to generate diagnostics: {str(e)}")
    # -------------------------------------------------------------
    return main_score, scores

def seed_runs(args):

    score_dicts = []
    seeds = [3407, 34071, 34072, 340, 1234]
    if args.debug: seeds = seeds[:2]
    for seed in seeds:
        main_score, scores = main_run(args, seed=seed)
        score_dicts.append(scores)

    _ = _print_score_dicts(score_dicts, args)


if __name__ == '__main__':
    # main_run(args, seed=3407)
    seed_runs(args)

    