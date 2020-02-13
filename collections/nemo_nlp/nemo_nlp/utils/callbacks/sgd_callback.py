# Copyright (c) 2019 NVIDIA Corporation
__all__ = ['eval_iter_callback', 'eval_epochs_done_callback']

import collections

import nemo_nlp
import nemo_nlp.data.datasets.sgd.data_utils as data_utils
import numpy as np
import torch
from fuzzywuzzy import fuzz
from nemo_nlp.utils.metrics.sgd_metrics import *

from nemo.utils.exp_logging import get_logger

logger = get_logger('')

REQ_SLOT_THRESHOLD = 0.5
F1Scores = collections.namedtuple("F1Scores", ["f1", "precision", "recall"])

# Evaluation and other relevant metrics for DSTC8 Schema-guided DST.
# (1) Active intent accuracy.
ACTIVE_INTENT_ACCURACY = "active_intent_accuracy"
# (2) Slot tagging F1.
SLOT_TAGGING_F1 = "slot_tagging_f1"
SLOT_TAGGING_PRECISION = "slot_tagging_precision"
SLOT_TAGGING_RECALL = "slot_tagging_recall"
# (3) Requested slots F1.
REQUESTED_SLOTS_F1 = "requested_slots_f1"
REQUESTED_SLOTS_PRECISION = "requested_slots_precision"
REQUESTED_SLOTS_RECALL = "requested_slots_recall"
# (4) Average goal accuracy.
AVERAGE_GOAL_ACCURACY = "average_goal_accuracy"
AVERAGE_CAT_ACCURACY = "average_cat_accuracy"
AVERAGE_NONCAT_ACCURACY = "average_noncat_accuracy"
# (5) Joint goal accuracy.
JOINT_GOAL_ACCURACY = "joint_goal_accuracy"
JOINT_CAT_ACCURACY = "joint_cat_accuracy"
JOINT_NONCAT_ACCURACY = "joint_noncat_accuracy"

NAN_VAL = "NA"


def tensor2list(tensor):
    return tensor.detach().cpu().tolist()


def eval_iter_callback(tensors, global_vars):
    # intents
    if 'intent_status' not in global_vars:
        global_vars['active_intent_labels'] = []
    if 'intent_status' not in global_vars:
        global_vars['active_intent_preds'] = []

    # requested slots
    if 'requested_slot_status' not in global_vars:
        global_vars['requested_slot_status'] = []
    if 'req_slot_predictions' not in global_vars:
        global_vars['req_slot_predictions'] = []

    # categorical slots
    if 'cat_slot_correctness' not in global_vars:
        global_vars['cat_slot_correctness'] = []

    # noncategorical slots
    if 'noncat_slot_correctness' not in global_vars:
        global_vars['noncat_slot_correctness'] = []

    if 'joint_noncat_accuracy' not in global_vars:
        global_vars['joint_noncat_accuracy'] = []
    if 'joint_cat_accuracy' not in global_vars:
        global_vars['joint_cat_accuracy'] = []

    for kv, v in tensors.items():
        # intents
        if kv.startswith('logit_intent_status'):
            logit_intent_status = v[0]
        elif kv.startswith('intent_status'):
            intent_status = v[0]

        # requested slots
        elif kv.startswith('logit_req_slot_status'):
            logit_req_slot_status = v[0]
        elif kv.startswith('requested_slot_status'):
            requested_slot_status = v[0]
        elif kv.startswith('req_slot_mask'):
            requested_slot_mask = v[0]

        # categorical slots
        elif kv.startswith('logit_cat_slot_status'):
            logit_cat_slot_status = v[0]
        elif kv.startswith('logit_cat_slot_value'):
            logit_cat_slot_value = v[0]
        elif kv.startswith('categorical_slot_status'):
            categorical_slot_status = v[0]
        elif kv.startswith('num_categorical_slots'):
            num_categorical_slots = v[0]
        elif kv.startswith('categorical_slot_values'):
            categorical_slot_values = v[0]
        elif kv.startswith('cat_slot_values_mask'):
            cat_slot_values_mask = v[0]

        # noncategorical slots
        elif kv.startswith('logit_noncat_slot_status'):
            logit_noncat_slot_status = v[0]
        elif kv.startswith('logit_noncat_slot_start'):
            logit_noncat_slot_start = v[0]
        elif kv.startswith('logit_noncat_slot_end'):
            logit_noncat_slot_end = v[0]
        elif kv.startswith('noncategorical_slot_status'):
            noncategorical_slot_status = v[0]
        elif kv.startswith('num_noncategorical_slots'):
            num_noncategorical_slots = v[0]
        elif kv.startswith('noncategorical_slot_value_start'):
            noncategorical_slot_value_start = v[0]
        elif kv.startswith('noncategorical_slot_value_end'):
            noncategorical_slot_value_end = v[0]

        elif kv.startswith('start_char_idx'):
            start_char_idxs = v[0]
        elif kv.startswith('end_char_idx'):
            end_char_idxs = v[0]

        elif kv.startswith('user_utterance'):
            user_utterances = v[0]

    num_active_intents = torch.sum(intent_status, axis=1).unsqueeze(1)

    # the intents represented as a one hot vectors
    # logits shape [batch, max_num_intents + 1] where 1 is for NONE intent

    active_intent_onehot_labels = intent_status[num_active_intents.view(-1) > 0.5]
    # get indices of active intents and add 1 to take into account NONE intent

    active_intent_labels = active_intent_onehot_labels.max(dim=1)[1] + 1

    active_intent_preds = torch.argmax(logit_intent_status, 1)[num_active_intents.view(-1) > 0.5]

    global_vars['active_intent_labels'].extend(tensor2list(active_intent_labels))
    global_vars['active_intent_preds'].extend(tensor2list(active_intent_preds))

    '''
    num_active_intents = torch.sum(intent_status, axis=1).unsqueeze(1)
    tensor_ones = torch.ones(num_active_intents.size(), dtype=torch.long)
 
    if num_active_intents.is_cuda:
        tensor_ones = tensor_ones.cuda()
 
    # adding label for NONE intent - 1 if no acive intent for the dialogue
    none_intent_label = tensor_ones - num_active_intents
    onehot_intent_labels = torch.cat([none_intent_label, intent_status], axis=1)
    _, intent_labels = onehot_intent_labels.max(dim=1)

    '''

    # # mask example with no noncategorical slots
    # noncat_slots_mask = torch.sum(noncategorical_slot_status, 1) > 0

    # get req slots predictions
    req_slot_predictions = torch.nn.Sigmoid()(logit_req_slot_status)
    # mask examples with padded slots
    req_slot_predictions = req_slot_predictions.view(-1)[requested_slot_mask]
    requested_slot_status = requested_slot_status.view(-1)[requested_slot_mask]

    ones = req_slot_predictions.new_ones(req_slot_predictions.size())
    zeros = req_slot_predictions.new_zeros(req_slot_predictions.size())
    req_slot_predictions = torch.where(req_slot_predictions > REQ_SLOT_THRESHOLD, ones, zeros)

    global_vars['req_slot_predictions'].extend(tensor2list(req_slot_predictions))
    global_vars['requested_slot_status'].extend(tensor2list(requested_slot_status))

    # list of corectness scores, each corresponding to one slot in the
    # service. The score is a float either 0.0 or 1.0 for categorical slot,
    # and in range [0.0, 1.0] for non-categorical slot.

    # Categorical slots
    # mask unused slots for the service
    max_num_cat_slots = categorical_slot_status.size()[-1]
    max_num_slots_matrix = torch.arange(0, max_num_cat_slots, 1).to(num_categorical_slots.device)
    cat_slot_status_mask = max_num_slots_matrix < torch.unsqueeze(num_categorical_slots, dim=-1)

    cat_slot_status_preds = torch.argmax(logit_cat_slot_status, -1)[cat_slot_status_mask]
    cat_slot_values_preds = torch.argmax(logit_cat_slot_value, -1)[cat_slot_status_mask]
    cat_slot_status_labels = categorical_slot_status[cat_slot_status_mask]
    cat_slot_values_labels = categorical_slot_values[cat_slot_status_mask]

    # active_cat_slots_mask = cat_slot_status_labels == data_utils.STATUS_ACTIVE
    # cat_slot_status_correctness = cat_slot_status_labels == cat_slot_status_preds
    # cat_slot_correctness = (cat_slot_values_labels == cat_slot_values_preds).type(torch.int)
    # # slots are predicted correctly if the both the slot value and the slot status are correct
    # cat_slot_correctness = cat_slot_correctness * cat_slot_status_correctness.type(torch.int)
    # import pdb; pdb.set_trace()
    # cat_batch_ids = get_batch_ids(cat_slot_status_mask)


    # determine if the slot status was predicted correctly
    cat_slot_status_correctness = cat_slot_status_labels == cat_slot_status_preds

    # evaluate cat slot prediction only if slot is active
    # if predicted slot status = 0,2 (off/doncare) but the true status is 1 (active) => categorical correctness
    # for such example is 0
    active_cat_slot_status_correctness = cat_slot_status_correctness * (cat_slot_status_labels == data_utils.STATUS_ACTIVE)
    cat_slot_values_correctness = (cat_slot_values_labels == cat_slot_values_preds).type(torch.int)

    cat_slot_correctness = torch.where(
        active_cat_slot_status_correctness, cat_slot_values_correctness, cat_slot_status_correctness.type(torch.int)
    )
    # cat_slot_correctness = tensor2list(cat_slot_correctness)
    # global_vars['cat_slot_correctness'].extend(tensor2list(cat_slot_correctness))
    # global_vars['active_cat_slots'].extend(tensor2list(active_cat_slots_mask))

    # global_vars['joint_cat_accuracy'].extend(get_cat_joint_accuracy(cat_slot_status_mask, cat_slot_correctness))


    # Noncategorical slots
    max_num_noncat_slots = noncategorical_slot_status.size()[-1]
    max_num_noncat_slots_matrix = torch.arange(0, max_num_noncat_slots, 1).to(num_noncategorical_slots.device)
    noncat_slot_status_mask = max_num_noncat_slots_matrix < torch.unsqueeze(num_noncategorical_slots, dim=-1)

    noncat_slot_status_preds = torch.argmax(logit_noncat_slot_status, -1)[noncat_slot_status_mask]
    noncat_slot_value_start_preds = torch.argmax(logit_noncat_slot_start, -1)[noncat_slot_status_mask]
    noncat_slot_value_end_preds = torch.argmax(logit_noncat_slot_end, -1)[noncat_slot_status_mask]

    noncat_slot_status_labels = noncategorical_slot_status[noncat_slot_status_mask]
    noncat_slot_value_start_labels = noncategorical_slot_value_start[noncat_slot_status_mask]
    noncat_slot_value_end_labels = noncategorical_slot_value_end[noncat_slot_status_mask]

    # slots with correctly predicted status
    noncat_slot_status_correctness = noncat_slot_status_labels == noncat_slot_status_preds
    # slots with correctly predicted ACTIVE status
    active_noncat_slot_status_correctness = noncat_slot_status_correctness * (
        noncat_slot_status_labels == data_utils.STATUS_ACTIVE
    )


    ########################################
    # remove:
    active_noncat_slot_status_correctness = noncat_slot_status_labels == data_utils.STATUS_ACTIVE
    system_utterances = None
    ########################################


    noncat_batch_ids = get_batch_ids(noncat_slot_status_mask)
    noncat_slot_correctness = []
    for i in range(len(noncat_slot_status_labels)):
        if noncat_slot_status_correctness[i]:
            if noncat_slot_status_labels[i] == data_utils.STATUS_ACTIVE:
                score = get_noncat_slot_value_match_score(
                                    i,
                                    noncat_slot_value_start_labels,
                                    noncat_slot_value_end_labels,
                                    noncat_slot_value_start_preds,
                                    noncat_slot_value_end_preds,
                                    start_char_idxs,
                                    end_char_idxs,
                                    noncat_batch_ids,
                                    user_utterances,
                                    system_utterances)
            else: # correctly predicted not ACTIVE status
                score = 1
        else:
            score = 0
        noncat_slot_correctness.append(score)

    print (noncat_slot_correctness)
    import pdb; pdb.set_trace()

    


    # calculate number of correct predictions
    nonactive_noncat_slot_status_correctness = noncat_slot_status_correctness * (noncat_slot_status_labels != data_utils.STATUS_ACTIVE)


    # find indices of noncat slots for which predicted status was correctly predicted and is ACTIVE
    inds_with_correct_active_noncat_slot_status = active_noncat_slot_status_correctness.type(torch.int).nonzero()

    # use active_noncat_slot_status_correctness to select only slots with correctly predicted status ACTIVE
    noncat_slot_value_start_labels = noncat_slot_value_start_labels[active_noncat_slot_status_correctness]
    noncat_slot_value_end_labels = noncat_slot_value_end_labels[active_noncat_slot_status_correctness]
    noncat_slot_value_start_preds = noncat_slot_value_start_preds[active_noncat_slot_status_correctness]
    noncat_slot_value_end_preds = noncat_slot_value_end_preds[active_noncat_slot_status_correctness]

    # get joint accuracies
    

    

    # slot_values_preds = get_slot_values(
    #     noncat_slot_value_start_preds,
    #     noncat_slot_value_end_preds,
    #     start_char_idxs,
    #     end_char_idxs,
    #     noncat_batch_ids[active_noncat_slot_status_correctness],
    #     user_utterances,
    #     system_utterances,
    # )

    # fuzzy_scores = get_fuzzy_scores(slot_values_true, slot_values_preds)
    # fuzzy_scores = get_fuzzy_scores(slot_values_true, slot_values_true)

    # import pdb; pdb.set_trace()

    # print(fuzzy_scores)
    # print()

    # get joint goal accuracies


# def get_slot_values(
#     noncat_slot_value_start,
#     noncat_slot_value_end,
#     start_char_idxs,
#     end_char_idxs,
#     batch_ids,
#     user_utterances,
#     system_utterances,
# ):

#     slot_values = []
#     for i in range(len(noncat_slot_value_start)):
#         tok_start_idx = noncat_slot_value_start[i]
#         tok_end_idx = noncat_slot_value_end[i]
#         ch_start_idx = start_char_idxs[batch_ids[i]][tok_start_idx]
#         ch_end_idx = end_char_idxs[batch_ids[i]][tok_end_idx]

#         if ch_start_idx < 0 and ch_end_idx < 0:
#             # Add span from the system utterance
#             print('system utterance required')
#             slot_values.append(None)
#             # slot_values[slot] = (
#             #     system_utterance[-ch_start_idx - 1:-ch_end_idx])
#         elif ch_start_idx > 0 and ch_end_idx > 0:
#             # Add span from the user utterance
#             slot_values.append(user_utterances[batch_ids[i]][ch_start_idx - 1 : ch_end_idx])
#         else:
#             slot_values.append(None)
#     return slot_values



def get_noncat_slot_value_match_score(
    slot_idx,
    noncat_slot_value_start_labels,
    noncat_slot_value_end_labels,
    noncat_slot_value_start_preds,
    noncat_slot_value_end_preds,
    start_char_idxs,
    end_char_idxs,
    batch_ids,
    user_utterances,
    system_utterances):
    
    str_true = _get_noncat_slot_value(
        slot_idx,
        noncat_slot_value_start_labels,
        noncat_slot_value_end_labels,
        start_char_idxs,
        end_char_idxs,
        batch_ids,
        user_utterances,
        system_utterances)

    str_value_pred = _get_noncat_slot_value(
        slot_idx,
        noncat_slot_value_start_preds,
        noncat_slot_value_end_preds,
        start_char_idxs,
        end_char_idxs,
        batch_ids,
        user_utterances,
        system_utterances)

    if str_true is None:
        if str_preds is None:
            # if the slot value was mentioned in the previous utterances of the dialogue
            # that are not part of the current turn
            score = 1  # true and prediction don't modify previously set slot value
        else:
            score = 0  # preds incorrectly modifyes previously set slot value
    else:
        score = fuzz.token_sort_ratio(str_true, str_preds) / 100.0

    return score



def _get_noncat_slot_value(
    slot_idx,
    noncat_slot_value_start,
    noncat_slot_value_end,
    start_char_idxs,
    end_char_idxs,
    batch_ids,
    user_utterances,
    system_utterances):
    tok_start_idx = noncat_slot_value_start[slot_idx]
    tok_end_idx = noncat_slot_value_end[slot_idx]
    ch_start_idx = start_char_idxs[batch_ids[slot_idx]][tok_start_idx]
    ch_end_idx = end_char_idxs[batch_ids[slot_idx]][tok_end_idx]

    if ch_start_idx < 0 and ch_end_idx < 0:
        # Add span from the system utterance
        print('system utterance required')
        slot_value = None
        # slot_values[slot] = (
        #     system_utterance[-ch_start_idx - 1:-ch_end_idx])
    elif ch_start_idx > 0 and ch_end_idx > 0:
        # Add span from the user utterance
        slot_value = user_utterances[batch_ids[slot_idx]][ch_start_idx - 1 : ch_end_idx]
    else:
        slot_value = None
    print (slot_value)
    return slot_value

    # noncat_slot_correctness = get_noncat_slot_value_match(user_utterances,
    #                                                       inds_with_correct_active_noncat_slot_status,
    #                                                       noncat_slot_value_start_labels,
    #                                                       noncat_slot_value_end_labels,
    #                                                       noncat_slot_value_start_preds,
    #                                                       noncat_slot_value_end_preds,
    #                                                       num_noncategorical_slots[0])

    # global_vars['noncat_slot_correctness'].extend(noncat_slot_correctness)
    # import pdb; pdb.set_trace()
    # print()
    # joint_noncat_accuracy = torch.prod(noncat_slot_correctness.view(-1,num_noncategorical_slots[0]), -1)
    # global_vars['joint_noncat_accuracy'].extend(tensor2list(joint_noncat_accuracy))

def get_batch_ids(slot_status_mask):
    # determine batch_id slot active slot is associated with
    # it's needed to get the corresponing user utterance correctly
    splitted_mask = torch.split(slot_status_mask, 1)
    splitted_mask = [i * x for i, x in enumerate(splitted_mask)]
    utterance_batch_ids = [i * x.type(torch.int) for i, x in enumerate(splitted_mask)]
    utterance_batch_ids = torch.cat(utterance_batch_ids)[slot_status_mask]
    return utterance_batch_ids


def get_joint_accuracy(slot_status_mask, slot_correctness_list):
    batch_ids = tensor2list(get_batch_ids(slot_status_mask))

    joint_accuracy = {}
    start_idx = 0
    for k, v in sorted(collections.Counter(batch_ids).items()):
        joint_accuracy[k] = np.prod(slot_correctness_list[start_idx : start_idx + v])
        start_idx += v
    return joint_accuracy


# def get_fuzzy_scores(slot_values_true_list, slot_values_preds_list):
#     """Returns fuzzy string similarity score in range [0.0, 1.0]."""

#     # The higher the score, the higher the similarity between the two strings
#     fuzzy_scores = []

#     for str_true, str_preds in zip(slot_values_true_list, slot_values_preds_list):
#         if str_true is None:
#             if str_preds is None:
#                 # if the slot value was mentioned in the previous utterances of the dialogue
#                 # that are not part of the current turn
#                 score = 1  # true and prediction don't modify previously set slot value
#             else:
#                 score = 0  # preds incorrectly modifyes previously set slot value
#         else:
#             score = fuzz.token_sort_ratio(str_true, str_preds) / 100.0
#         fuzzy_scores.append(score)
#     return fuzzy_scores

# def get_noncat_slot_value_match_score(true_slot_values_truet, slot_values_preds_list):
#     """Returns fuzzy string similarity score in range [0.0, 1.0]."""

#     # The higher the score, the higher the similarity between the two strings


    
#     if str_true is None:
#         if str_preds is None:
#             # if the slot value was mentioned in the previous utterances of the dialogue
#             # that are not part of the current turn
#             score = 1  # true and prediction don't modify previously set slot value
#         else:
#             score = 0  # preds incorrectly modifyes previously set slot value
#     else:
#         score = fuzz.token_sort_ratio(str_true, str_preds) / 100.0

#     return score



# def get_noncat_slot_value_match(
#     user_utterances,
#     indices,
#     noncat_slot_value_start_labels,
#     noncat_slot_value_end_labels,
#     noncat_slot_value_start_preds,
#     noncat_slot_value_end_preds,
#     num_noncategorical_slots,
# ):
#     """Calculate non-categorical slots correctness.

#     Args:
#       str_ref_list: a list of reference strings.
#       str_hyp: the hypothesis string.

#     Returns:
#     score: The highest fuzzy string match score of the references and hypotheis.
#     """
#     noncat_slot_correctness = []
#     # user_utterance_ind = indices /
#     for i, ind in enumerate(indices):
#         user_utterance = user_utterances[indices / num_noncategorical_slots]
#         str_label = user_utterance[noncat_slot_value_start_labels[ind] : noncat_slot_value_end_labels[ind]]
#         str_preds = user_utterance[noncat_slot_value_start_preds[ind] : noncat_slot_value_end_preds[ind]]
#         noncat_slot_correctness.append(max(0, fuzzy_string_match(str_label, str_preds)))

#     return noncat_slot_correctness


def eval_epochs_done_callback(global_vars):
    active_intent_labels = np.asarray(global_vars['active_intent_labels'])
    active_intent_preds = np.asarray(global_vars['active_intent_preds'])

    active_intent_accuracy = sum(active_intent_labels == active_intent_preds) / len(active_intent_labels)

    req_slot_predictions = np.asarray(global_vars['req_slot_predictions'], dtype=int)
    requested_slot_status = np.asarray(global_vars['requested_slot_status'], dtype=int)
    req_slot_metrics = compute_f1(req_slot_predictions, requested_slot_status)

    correctness_cat_slots = np.asarray(global_vars['cat_slot_correctness'], dtype=int)
    # joint_acc, turn_acc = \
    #     evaluate_metrics(global_vars['comp_res'],
    #                      global_vars['gating_labels'],
    #                      global_vars['gating_preds'],
    #                      data_desc.gating_dict["ptr"])

    # gating_comp_flatten = (np.asarray(global_vars['gating_labels']) == np.asarray(global_vars['gating_preds'])).ravel()
    # gating_acc = np.sum(gating_comp_flatten) / len(gating_comp_flatten)

    cat_slot_correctness = np.asarray(global_vars['cat_slot_correctness'])
    # noncat_slot_correctness = np.asarray(global_vars['noncat_slot_correctness'])

    average_cat_accuracy = np.mean(cat_slot_correctness)
    joint_cat_accuracy = np.mean(np.asarray(global_vars['joint_cat_accuracy'], dtype=int))

    # average_noncat_accuracy = np.mean(noncat_slot_correctness)
    # average_goal_accuracy = np.mean(np.concatenate((cat_slot_correctness, noncat_slot_correctness)))

    metrics = {
        'all_services': {
            # Active intent accuracy
            "active_intent_accuracy": active_intent_accuracy,
            "average_cat_accuracy": average_cat_accuracy,
            # "average_goal_accuracy": average_goal_accuracy,
            # "average_noncat_accuracy": average_noncat_accuracy,
            "joint_cat_accuracy": joint_cat_accuracy,
            # "joint_goal_accuracy": 0.4904726693494299,
            # "joint_noncat_accuracy": 0.6226867035546613,
            # Slot tagging F1
            "requested_slots_f1": req_slot_metrics.f1,
            "requested_slots_precision": req_slot_metrics.precision,
            "requested_slots_recall": req_slot_metrics.recall,
            # Average goal accuracy
        }
    }

    print('\n' + '#' * 50)
    for k, v in metrics['all_services'].items():
        print(f'{k}: {v}')
    print('#' * 50 + '\n')

    # active_intent_acc = metrics.get_active_intent_accuracy(
    #         frame_ref, frame_hyp)
    # slot_tagging_f1_scores = metrics.get_slot_tagging_f1(
    #     frame_ref, frame_hyp, turn_ref["utterance"], service)
    # requested_slots_f1_scores = metrics.get_requested_slots_f1(
    #     frame_ref, frame_hyp)
    # goal_accuracy_dict = metrics.get_average_and_joint_goal_accuracy(
    #     frame_ref, frame_hyp, service)

    return metrics


def get_average_and_joint_goal_accuracy(frame_ref, frame_hyp, service):
    """Get average and joint goal accuracies of a frame.

  Args:
    frame_ref: single semantic frame from reference (ground truth) file.
    frame_hyp: single semantic frame from hypothesis (prediction) file.
    service: a service data structure in the schema. We use it to obtain the
      list of slots in the service and infer whether a slot is categorical.

  Returns:
    goal_acc: a dict whose values are average / joint
        all-goal / categorical-goal / non-categorical-goal accuracies.
  """
    goal_acc = {}

    list_acc, slot_active, slot_cat = compare_slot_values(
        frame_ref["state"]["slot_values"], frame_hyp["state"]["slot_values"], service
    )

    # (4) Average goal accuracy.
    active_acc = [acc for acc, active in zip(list_acc, slot_active) if active]
    goal_acc[AVERAGE_GOAL_ACCURACY] = np.mean(active_acc) if active_acc else NAN_VAL
    # (4-a) categorical.
    active_cat_acc = [acc for acc, active, cat in zip(list_acc, slot_active, slot_cat) if active and cat]
    goal_acc[AVERAGE_CAT_ACCURACY] = np.mean(active_cat_acc) if active_cat_acc else NAN_VAL
    # (4-b) non-categorical.
    active_noncat_acc = [acc for acc, active, cat in zip(list_acc, slot_active, slot_cat) if active and not cat]
    goal_acc[AVERAGE_NONCAT_ACCURACY] = np.mean(active_noncat_acc) if active_noncat_acc else NAN_VAL

    # (5) Joint goal accuracy.
    goal_acc[JOINT_GOAL_ACCURACY] = np.prod(list_acc) if list_acc else NAN_VAL
    # (5-a) categorical.
    cat_acc = [acc for acc, cat in zip(list_acc, slot_cat) if cat]
    goal_acc[JOINT_CAT_ACCURACY] = np.prod(cat_acc) if cat_acc else NAN_VAL
    # (5-b) non-categorical.
    noncat_acc = [acc for acc, cat in zip(list_acc, slot_cat) if not cat]
    goal_acc[JOINT_NONCAT_ACCURACY] = np.prod(noncat_acc) if noncat_acc else NAN_VAL

    return goal_acc


F1Scores = collections.namedtuple("F1Scores", ["f1", "precision", "recall"])


def compute_f1(predictions, labels):
    """Compute F1 score from labels (grouth truth) and predictions.

  Args:
    predictions: numpy array of predictions
    labels: numpy array of labels

  Returns:
    A F1Scores object containing F1, precision, and recall scores.
  """
    true = sum(labels)
    positive = sum(predictions)
    true_positive = sum(predictions & labels)

    precision = float(true_positive) / positive if positive else 1.0
    recall = float(true_positive) / true if true else 1.0
    if precision + recall > 0.0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:  # The F1-score is defined to be 0 if both precision and recall are 0.
        f1 = 0.0

    return F1Scores(f1=f1, precision=precision, recall=recall)
