import numpy as np 
import pandas as pd 
from tqdm import tqdm

import torch, os, pickle

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

class ExpDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()

        self.task = args.task
        # task/input config
        self.target = _task2target(args.task) # e.g., MORT_HOSP
        self.modality = args.modality # default to both
        self.batch_size = args.batch_size
        self.silent = args.silent
        self.trim_cohort = args.trim_cohort

        self.use_weighted_sampler = getattr(args, 'use_weighted_sampler', False)
        self.sampler_smoothing = getattr(args, 'sampler_smoothing', 0.4)
        self.img_dropout_prob = getattr(args, 'img_dropout_prob', 0.0)

        # self.note_encode_dir = args.note_encode_dir
        self.note_encode_name = args.note_encode_name

        root_dir = os.path.abspath(args.root_dir)
        self.root_dir = root_dir

        if self.task == 'bone_class':
            # Load metadata
            metadata_df = pd.read_csv(args.metadata_csv)
            # Filter rows by split
            self.train_df = metadata_df[metadata_df['split'] == 'train']
            self.val_df = metadata_df[metadata_df['split'] == 'val']
            self.test_df = metadata_df[metadata_df['split'] == 'test']
            
            with open(args.txt_encode_path, 'rb') as f:
                self.data_txt = pickle.load(f)
            with open(args.img_encode_path, 'rb') as f:
                self.data_img = pickle.load(f)
                
            self.my_collate_fn = collate_both_bone
            
            if args.debug:
                self.train_df, self.val_df, self.test_df = [df.head(5) for df in [self.train_df, self.val_df, self.test_df]]
        else:
            # cohort df 
            cohort_name = _task2cohort(args.task)
            if not args.trim_cohort:
                all_cohort_df = pd.read_pickle(os.path.join(root_dir, 'data/cohort', f'splits_{cohort_name}.p'))
            else:
                all_cohort_df = pd.read_pickle(os.path.join(root_dir, 'data/cohort', f'trim_splits_{cohort_name}.p'))

            self.train_df, self.val_df, self.test_df = all_cohort_df['train'], all_cohort_df['val'], all_cohort_df['test']
            if args.debug:
                self.train_df, self.val_df, self.test_df = [df.head(200) for df in [self.train_df, self.val_df, self.test_df]]

            if args.train_size_frac < 1:
                # use frac of all train data
                num_sample = int(len(self.train_df) * args.train_size_frac)
                self.train_df = self.train_df.head(num_sample)


            if self.modality == 'both':
                self.my_collate_fn = collate_both

                self.data_txt = pd.read_pickle(os.path.join(root_dir, 'data/notes_encoded', f'{cohort_name}_df_{self.note_encode_name}.p' ))
                self.data_ts  = pd.read_pickle(os.path.join(root_dir, 'data/measurements', f'{cohort_name}_hourly.p'))
                self.X_mean = get_X_mean(self.data_ts[0])

            elif self.modality == 'struct':
                self.my_collate_fn = collate_ts

                self.data_ts  = pd.read_pickle(os.path.join(root_dir, 'data/measurements', f'{cohort_name}_hourly.p'))
                self.X_mean = get_X_mean(self.data_ts[0])

            elif self.modality == 'text':
                self.my_collate_fn = collate_txt

                self.data_txt = pd.read_pickle(os.path.join(root_dir, 'data/notes_encoded', f'{cohort_name}_df_{self.note_encode_name}.p' ))
    

    def init_datasets(self, cohort_df_list):

        if self.task == 'bone_class':
            target_str = 'trim_missing' if self.trim_cohort else self.target
            # cohort_df_list is always [train_df] or [train_df, val_df, test_df] in that order
            is_train_flags = [True] + [False] * (len(cohort_df_list) - 1) if len(cohort_df_list) > 1 else [False]
            dataset_list = [
                BoneDataset(
                    target_str, cohort_df, self.data_txt, self.data_img, self.silent,
                    img_dropout_prob=self.img_dropout_prob, is_train=is_train,
                )
                for cohort_df, is_train in zip(cohort_df_list, is_train_flags)
            ]
        elif self.modality == 'struct':
            dataset_list = [
                StructDataset(self.target, cohort_df, data_df, self.silent)
                for cohort_df, data_df in zip(cohort_df_list, self.data_ts)
            ] 
        elif self.modality == 'text':
            dataset_list = [
                TextDataset(self.target, cohort_df, self.data_txt, silent=self.silent)
                for cohort_df in cohort_df_list
            ]
        elif self.modality == 'both':
            dataset_list = [
                BimodalDataset(self.target, cohort_df, [self.data_txt, data_df], silent=self.silent)
                for cohort_df, data_df in zip(cohort_df_list, self.data_ts)
            ]
        else:
            raise NotImplementedError

        return dataset_list


    def setup(self, stage=None):
        if stage == None:
            self.train_dataset, self.val_dataset, self.test_dataset = self.init_datasets(
                [self.train_df, self.val_df, self.test_df]
            )

        elif stage == 'test':
            if self.task == 'bone_class':
                self.test_dataset = self.init_datasets([self.test_df])[0]
            else:
                self.test_dataset = BimodalDataset(self.target, self.test_df, [self.data_txt, self.data_ts[-1]], silent=self.silent )


    # def train_dataloader(self, bsize=None):
    #     batch_size = bsize if bsize else self.batch_size
    #     return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.my_collate_fn, drop_last=True)

    def train_dataloader(self, bsize=None):
        batch_size = bsize if bsize else self.batch_size
        
        if self.task == 'bone_class' and getattr(self, 'use_weighted_sampler', False):
            from torch.utils.data import WeightedRandomSampler
            
            train_labels = [sample[1] for sample in self.train_dataset.data]
            num_classes = len(set(train_labels))
            # bincount's length is max(label)+1 already; minlength is just a floor,
            # but make it explicit so this is correct even if a class is entirely
            # absent from a particular run (e.g. debug subsampling).
            class_counts = np.bincount(train_labels, minlength=max(train_labels) + 1)
            class_counts = np.where(class_counts == 0, 1, class_counts)

            # --- SMOOTHED SAMPLING WEIGHTS ---
            # exponent of 1.0 = full inverse frequency (aggressive oversampling of rare classes)
            # exponent of 0.0 = uniform sampling (no rebalancing)
            class_weights = 1.0 / (class_counts ** self.sampler_smoothing)
            sample_weights = [class_weights[label] for label in train_labels]

            print(f"[use_weighted_sampler] class counts: {class_counts.tolist()}")
            print(f"[use_weighted_sampler] sampler smoothing={self.sampler_smoothing}, "
                  f"per-class weight: {[round(w,3) for w in class_weights]}")
            
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            
            return DataLoader(
                self.train_dataset, 
                batch_size=batch_size, 
                sampler=sampler,
                collate_fn=self.my_collate_fn, 
                drop_last=True,
                num_workers=4,          # Set to 4 or 8 to enable asynchronous data fetching
                pin_memory=True        # Speeds up tensor transfers to the GPU
            )

        return DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.my_collate_fn, drop_last=True
        )

    def val_dataloader(self, bsize=None):
        batch_size = bsize if bsize else self.batch_size
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.my_collate_fn)
    def test_dataloader(self, bsize=None):
        batch_size = bsize if bsize else self.batch_size
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.my_collate_fn)

    def test_dataloader_missing_image(self, bsize=None):
        """Eval-only dataloader that forces every image to be treated as missing,
        to measure how well the model degrades to text-only inference.
        Only valid for task == 'bone_class'."""
        assert self.task == 'bone_class', "missing-image eval only implemented for bone_class"
        batch_size = bsize if bsize else self.batch_size
        target_str = 'trim_missing' if self.trim_cohort else self.target
        forced_missing_dataset = BoneDataset(
            target_str, self.test_df, self.data_txt, self.data_img, self.silent,
            img_dropout_prob=1.0, is_train=True,  # is_train=True just to make the dropout prob active; it still only touches img
        )
        return DataLoader(forced_missing_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.my_collate_fn)


def _task2target(task):
    if task in ['ms_drg', 'apr_drg']:
        target = 'DRG'
    else:
        target = task.upper()
    return target

def _task2cohort(task):
    if 'ms' in task:
        cohort = 'drg_ms'
    elif 'apr' in task:
        cohort = 'drg_apr'
    else:
        cohort = 'mextract'
    return cohort
    


def collate_txt(batch):

    notes = []
    masks = []
    for b in batch:
        n = b[0]
        if len(n) > 0 and isinstance(n, list):
            notes.append(torch.tensor(np.array(n)))
            masks.append(torch.ones(len(n)).long())
        else:
            notes.append(torch.zeros(0, 768))
            masks.append(torch.zeros(1))

    # notes = [ torch.tensor(np.array(b[0])) for b in batch ]
    # masks = [ torch.ones(n.size(0)).long() for n in notes ]

    notes = pad_sequence(notes, batch_first=True)
    masks = pad_sequence(masks, batch_first=True)

    labels = torch.tensor(np.array([ b[1] for b in batch ]))
    stays  = np.array([ b[2] for b in batch ])

    return notes, labels, masks, stays


def collate_ts(batch):

    input_window = batch[0][-1]

    series = []
    masks = []
    for b in batch:
        x = b[0]
        if x is not None:
            series.append(x)
            masks.append(np.ones(input_window))
        else:
            series.append(np.zeros((input_window, 312)))
            masks.append(np.zeros(input_window))

    series = torch.tensor(np.array(series))
    masks = torch.tensor(np.array(masks)).long()

    labels = torch.tensor(np.array([ b[1] for b in batch ]))
    stays  = np.array([ b[2] for b in batch ])

    return series, labels, masks, stays


def collate_both(batch):

    # ts 
    input_window = batch[0][-1]

    series = []
    masks_ts = []
    for b in batch:
        x = b[0][0]
        if x is not None:
            series.append(x)
            masks_ts.append(np.ones(input_window))
        else:
            series.append(np.zeros((input_window, 312)))
            masks_ts.append(np.zeros(input_window))

    series = torch.tensor(np.array(series))
    masks_ts = torch.tensor(np.array(masks_ts)).long()

    # txt
    notes = []
    masks_txt = []
    for b in batch:
        n = b[0][1]
        if len(n) > 0 and isinstance(n, list):
            notes.append(torch.tensor(np.array(n)))
            masks_txt.append(torch.ones(len(n)).long())
        else:
            notes.append(torch.zeros(0, 768))
            masks_txt.append(torch.zeros(1))

    notes = pad_sequence(notes, batch_first=True)
    masks_txt = pad_sequence(masks_txt, batch_first=True)

    # others
    labels = torch.tensor(np.array([ b[1] for b in batch ]))
    stays  = np.array([ b[2] for b in batch ])

    return (series, notes), labels, (masks_ts, masks_txt), stays


class TemplateDataset(Dataset):
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


class TextDataset(TemplateDataset):
    def __init__(self, target, cohort_df, note_df, max_num_note=31, silent=False):
        super().__init__()

        self.data = []
        input_window = 48 if target == 'DRG' else 24

        for _, row in tqdm(cohort_df.reset_index().iterrows(), total=len(cohort_df), disable=silent):
            
            x, hadm = _query_note(row, note_df)
            x = x[::-1][:max_num_note] # keep latest note first 

            y = row[target]
            self.data.append([x, y, hadm, input_window])


class StructDataset(TemplateDataset):
    def __init__(self, target, cohort_df, ts_df, silent=False):
        super().__init__()

        self.data = []
        input_window = 48 if target == 'DRG' else 24

        for _, row in tqdm(cohort_df.reset_index().iterrows(), total=len(cohort_df), disable=silent):

            x, hadm = _query_ts(row, ts_df)

            y = row[target]
            self.data.append([x, y, hadm, input_window])



class BimodalDataset(TemplateDataset):
    def __init__(self, target, cohort_df, data_dfs, max_num_note=31, silent=False):
        super().__init__()

        assert len(data_dfs) == 2
        note_df, ts_df = data_dfs

        self.data = []
        input_window = 48 if target == 'DRG' else 24

        for _, row in tqdm(cohort_df.reset_index().iterrows(), total=len(cohort_df), disable=silent):

            x_ts, hadm = _query_ts(row, ts_df)
            x_txt, hadm2=_query_note(row, note_df)
            x_txt = x_txt[::-1][:max_num_note] # keep latest note first 

            assert hadm == hadm2 

            y = row[target]
            self.data.append([(x_ts, x_txt), y, hadm, input_window])



def _query_ts(row, ts_df):
    subj, hadm, icu = row['SUBJECT_ID'], row['HADM_ID'], row['ICUSTAY_ID']
    msk = id_msk(ts_df, subj, hadm, icu)

    if msk.any() == False:
        return None, hadm

    else:
        x = ts_df[msk].values
        return x, hadm 


def _query_note(row, note_df):
    hadm = row['HADM_ID']
    msk = note_df['HADM_ID'] == hadm

    if msk.any() == False:
        return [], hadm

    else:
        x = note_df[msk]['vector'].tolist()
        return x, hadm




def id_msk(df, subj, hadm, icu):
    msk1 = df.index.get_level_values('subject_id') == subj
    msk2 = df.index.get_level_values('hadm_id') == hadm
    msk3 = df.index.get_level_values('icustay_id') == icu
    return msk1 & msk2 & msk3

def to_3D_tensor(df):
    idx = pd.IndexSlice
    return np.dstack([df.loc[idx[:,:,:,i], :].values for i in sorted(set(df.index.get_level_values('hours_in')))])

def get_X_mean(lvl2_train):
    X_mean = np.nanmean(
            to_3D_tensor(
                lvl2_train.loc[:, pd.IndexSlice[:, 'mean']] * 
                np.where((lvl2_train.loc[:, pd.IndexSlice[:, 'mask']] == 1).values, 1, np.NaN)
            ),
            axis=0, keepdims=True
        ).transpose([0, 2, 1])
    X_mean = np.nan_to_num(X_mean,0)
    return X_mean


# def collate_both_bone(batch):
#     # Each item in batch is: [(x_img, x_txt), label, hadm_id, 1]
#     # x_img: np.array of shape [1536] or None (if missing)
#     # x_txt: list/vector of shape [1536]
    
#     images = []
#     masks_img = []
#     for b in batch:
#         img_feat = b[0][0]
#         if img_feat is not None and not (isinstance(img_feat, float) and np.isnan(img_feat).all()):
#             images.append(torch.tensor(img_feat, dtype=torch.float32).unsqueeze(0)) # [1, 1536]
#             masks_img.append(torch.ones(1).long())
#         else:
#             images.append(torch.zeros(1, 1536, dtype=torch.float32))
#             masks_img.append(torch.zeros(1).long())
            
#     images = torch.stack(images) # [batch_size, 1, 1536]
#     masks_img = torch.stack(masks_img) # [batch_size, 1]
    
#     notes = []
#     masks_txt = []
#     for b in batch:
#         txt_feat = b[0][1]
#         if txt_feat is not None:
#             if isinstance(txt_feat, list) and len(txt_feat) > 0:
#                 if isinstance(txt_feat[0], (list, np.ndarray)):
#                     notes.append(torch.tensor(txt_feat[0], dtype=torch.float32).unsqueeze(0))
#                 else:
#                     notes.append(torch.tensor(txt_feat, dtype=torch.float32).unsqueeze(0))
#                 masks_txt.append(torch.ones(1).long())
#             elif isinstance(txt_feat, np.ndarray):
#                 notes.append(torch.tensor(txt_feat, dtype=torch.float32).unsqueeze(0))
#                 masks_txt.append(torch.ones(1).long())
#             else:
#                 notes.append(torch.tensor(txt_feat, dtype=torch.float32).unsqueeze(0))
#                 masks_txt.append(torch.ones(1).long())
#         else:
#             notes.append(torch.zeros(1, 1536, dtype=torch.float32))
#             masks_txt.append(torch.zeros(1).long())
            
#     notes = torch.stack(notes) # [batch_size, 1, 1536]
#     masks_txt = torch.stack(masks_txt) # [batch_size, 1]
    
#     labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
#     hadms = [b[2] for b in batch]
    
#     return (images, notes), labels, (masks_img, masks_txt), hadms

def collate_both_bone(batch):
    images = []
    masks_img = []
    EMBED_DIM = 3584
    for b in batch:
        img_feat = b[0][0]
        if img_feat is not None and not (isinstance(img_feat, float) and np.isnan(img_feat).all()):
            images.append(torch.tensor(img_feat, dtype=torch.float32).view(1, -1)) # [1, 1536]
            masks_img.append(torch.ones(1, 1, dtype=torch.long))                  # [1, 1] -> Keep it 2D
        else:
            images.append(torch.zeros(1, EMBED_DIM, dtype=torch.float32))
            masks_img.append(torch.zeros(1, 1, dtype=torch.long))
            
    images = torch.stack(images, dim=0).squeeze(1) # [batch_size, 1, 1536]
    
    # CRITICAL FIX: Use torch.cat along dim=0 instead of stack+squeeze to guarantee [batch_size, 1]
    masks_img = torch.cat(masks_img, dim=0)        # Shape: [batch_size, 1]
    ### debug only
    if masks_img[0].item() == 0:
        print("CONFIRMED: Mask is 0, model is ignoring image.")
    else:
        print("WARNING: Image is active!")

    ###
    notes = []
    masks_txt = []
    for b in batch:
        txt_feat = b[0][1]
        if txt_feat is not None:
            if isinstance(txt_feat, list) and len(txt_feat) > 0:
                feat_arr = txt_feat[0] if isinstance(txt_feat[0], (list, np.ndarray)) else txt_feat
                notes.append(torch.tensor(feat_arr, dtype=torch.float32).view(1, -1))
                masks_txt.append(torch.ones(1, 1, dtype=torch.long))
            elif isinstance(txt_feat, np.ndarray):
                notes.append(torch.tensor(txt_feat, dtype=torch.float32).view(1, -1))
                masks_txt.append(torch.ones(1, 1, dtype=torch.long))
            else:
                notes.append(torch.tensor(txt_feat, dtype=torch.float32).view(1, -1))
                masks_txt.append(torch.ones(1, 1, dtype=torch.long))
        else:
            notes.append(torch.zeros(1, EMBED_DIM, dtype=torch.float32))
            masks_txt.append(torch.zeros(1, 1, dtype=torch.long))
            
    notes = torch.stack(notes, dim=0).squeeze(1)   # [batch_size, 1, 1536]
    
    # CRITICAL FIX: Match the 2D configuration here too
    masks_txt = torch.cat(masks_txt, dim=0)        # Shape: [batch_size, 1]
    
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    hadms = [b[2] for b in batch]
    
    return (images, notes), labels, (masks_img, masks_txt), hadms


class BoneDataset(Dataset):
    def __init__(self, target, cohort_df, data_txt, data_img, silent=False,
                 img_dropout_prob=0.0, is_train=False):
        super().__init__()
        self.data = []

        # img_dropout_prob: probability of simulating a MISSING image at __getitem__ time,
        # even when a real image embedding is available. Only meaningful when is_train=True
        # (we don't want eval metrics randomly fluctuating run to run on the standard val/test loop).
        # This is the modality-dropout strategy used to train robustness to missing modalities:
        # text is always kept (required), image is randomly dropped so the model learns to use
        # the masked-attention fallback path (mask=0 -> CLS-only / text-only inference) instead of
        # only ever seeing fully-paired inputs at train time.
        self.img_dropout_prob = img_dropout_prob if is_train else 0.0
        self.is_train = is_train

        for _, row in cohort_df.iterrows():
            hadm = row['HADM_ID']
            label = row['label'] # target
            
            # Query text embedding
            txt_row = data_txt[data_txt['HADM_ID'] == hadm]
            if len(txt_row) > 0:
                x_txt = txt_row.iloc[0]['vector']
            else:
                x_txt = None
                
            # Query image embedding
            img_row = data_img[data_img['HADM_ID'] == hadm]
            if len(img_row) > 0:
                x_img = img_row.iloc[0]['vector']
            else:
                x_img = None
                
            # Missing modalities handling config:
            # If target is 'trim_missing' and image is missing, we exclude this sample.
            if target == 'trim_missing' and x_img is None:
                continue
                
            self.data.append([(x_img, x_txt), label, hadm, 1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        (x_img, x_txt), label, hadm, window = self.data[idx]

        if self.img_dropout_prob > 0.0 and x_img is not None and np.random.rand() < self.img_dropout_prob:
            # Simulate a missing image for this sample on this access.
            # Original self.data is left untouched, so this re-randomizes every epoch.
            x_img = None

        return [(x_img, x_txt), label, hadm, window]



