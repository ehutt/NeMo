"""
This code were adapted from 
https://github.com/google-research/google-research/tree/master/schema_guided_dst
"""

import os

import collections
import numpy as np
from torch.utils.data import Dataset

import nemo
from nemo_nlp.data.datasets.sgd import data_utils


class SGDDataset(Dataset):
    """ 
    TODO: Update here
    """

    def __init__(self,
                 task_name,
                 vocab_file,
                 do_lower_case,
                 tokenizer,
                 max_seq_length,
                 data_dir,
                 dialogues_example_dir,
                 overwrite_dial_file,
                 dataset_split,
                 schema_emb_processor):  

        # Generate the dialogue examples if needed or specified.
        dial_file_name = f"{task_name}_{dataset_split}_examples.processed"
        dial_file = os.path.join(dialogues_example_dir,
                                 dial_file_name)

        if not os.path.exists(dialogues_example_dir):
            os.makedirs(dialogues_example_dir)

        if os.path.exists(dial_file) and not overwrite_dial_file:
            nemo.logging.info(f"Loading dialogue examples from {dial_file}.")
            with open(dial_file, "rb") as f:
                self.features = np.load(f, allow_pickle=True)

        else:
            nemo.logging.info("Start generating the dialogue examples.")

            processor = data_utils.Dstc8DataProcessor(
                data_dir,
                train_file_range=data_utils.FILE_RANGES[task_name]["train"],
                dev_file_range=data_utils.FILE_RANGES[task_name]["dev"],
                test_file_range=data_utils.FILE_RANGES[task_name]["test"],
                vocab_file=vocab_file,
                do_lower_case=do_lower_case,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length)

            self.features = processor.get_dialog_examples(dataset_split)
            with open(dial_file, "wb") as f:
                np.save(f, self.features)

            nemo.logging.info(f"The dialogue examples saved at {dial_file}")
            nemo.logging.info("Finish generating the dialogue examples.")

        self.schema_data_dict = schema_emb_processor._get_schema_embeddings()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        ex = self.features[idx]
        service_id = ex.service_schema.service_id
        return (#np.array(ex.example_id),
                np.array(ex.is_real_example),
                np.array(service_id),
                np.array(ex.utterance_ids),
                np.array(ex.utterance_segment),
                np.array(ex.utterance_mask, dtype=np.long),
                
                np.array(ex.num_categorical_slots),
                np.array(ex.categorical_slot_status),
                np.array(ex.num_categorical_slot_values),
                np.array(ex.categorical_slot_values),

                np.array(ex.num_noncategorical_slots),
                np.array(ex.noncategorical_slot_status),
                np.array(ex.noncategorical_slot_value_start),
                np.array(ex.noncategorical_slot_value_end),
                np.array(ex.start_char_idx),
                np.array(ex.end_char_idx),

                np.array(ex.num_slots), # num_requested_slots
                np.array(ex.requested_slot_status, dtype=np.float32),

                np.array(ex.num_intents),
                np.array(ex.intent_status),

                np.array(self.schema_data_dict['cat_slot_emb'][service_id], dtype=np.float32),
                np.array(self.schema_data_dict['cat_slot_value_emb'][service_id], dtype=np.float32),
                np.array(self.schema_data_dict['noncat_slot_emb'][service_id], dtype=np.float32),
                np.array(self.schema_data_dict['req_slot_emb'][service_id], dtype=np.float32),
                np.array(self.schema_data_dict['intent_emb'][service_id], dtype=np.float32))

        """
        [('cat_slot_emb', (6, 768)),
         ('cat_slot_value_emb', (6, 11, 768)), 
         ('noncat_slot_emb', (12, 768)), 
         ('req_slot_emb', (18, 768)), 
         ('intent_emb', (4, 768))]
        """
        


