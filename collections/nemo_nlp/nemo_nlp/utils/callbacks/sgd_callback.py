# Copyright (c) 2019 NVIDIA Corporation
__all__ = ['eval_iter_callback', 'eval_epochs_done_callback']

import numpy as np
import torch

from nemo.utils.exp_logging import get_logger
import nemo_nlp
from nemo_nlp.utils.metrics.sgd_metrics import *

logger = get_logger('')


def eval_iter_callback(tensors,
                       global_vars):
    import pdb; pdb.set_trace()

    if 'logit_intent_status' not in global_vars:
        global_vars['logit_intent_status'] = []
    if 'logit_req_slot_status' not in global_vars:
        global_vars['logit_req_slot_status'] = []
    if 'logit_cat_slot_status' not in global_vars:
        global_vars['logit_cat_slot_status'] = []
    if 'logit_cat_slot_value' not in global_vars:
        global_vars['logit_cat_slot_value'] = []
    if 'logit_noncat_slot_status' not in global_vars:
        global_vars['logit_noncat_slot_status'] = []
    if 'logit_noncat_slot_start' not in global_vars:
        global_vars['logit_noncat_slot_start'] = []
    if 'logit_noncat_slot_end' not in global_vars:
        global_vars['logit_noncat_slot_end'] = []
    if 'intent_status' not in global_vars:
        global_vars['intent_status'] = []
    if 'requested_slot_status' not in global_vars:
        global_vars['requested_slot_status'] = []
    if 'intent_status' not in global_vars:
        global_vars['intent_status'] = []
    if 'num_slots' not in global_vars:
        global_vars['num_slots'] = []
    if 'categorical_slot_status' not in global_vars:
        global_vars['categorical_slot_status'] = []
    if 'num_categorical_slots' not in global_vars:
        global_vars['num_categorical_slots'] = []
    if 'categorical_slot_values' not in global_vars:
        global_vars['categorical_slot_values'] = []
    if 'noncategorical_slot_status' not in global_vars:
        global_vars['noncategorical_slot_status'] = []
    if 'categorical_slot_values' not in global_vars:
        global_vars['categorical_slot_values'] = []
    if 'num_noncategorical_slots' not in global_vars:
        global_vars['num_noncategorical_slots'] = []
    if 'noncategorical_slot_value_start' not in global_vars:
        global_vars['noncategorical_slot_value_start'] = []
    if 'noncategorical_slot_value_end' not in global_vars:
        global_vars['noncategorical_slot_value_end'] = []


    for kv, v in tensors.items():
        if kv.startswith('logit_intent_status'):
            logit_intent_status = v[0]
        elif kv.startswith('logit_req_slot_status'):
            global_vars['logit_req_slot_status'].append(v[0].cpu().numpy())
        elif kv.startswith('logit_cat_slot_status'):
            global_vars['logit_cat_slot_status'].append(v[0].cpu().numpy())
        elif kv.startswith('logit_cat_slot_value'):
            global_vars['logit_cat_slot_value'].append(v[0].cpu().numpy())
        elif kv.startswith('logit_noncat_slot_status'):
            global_vars['logit_noncat_slot_status'].append(v[0].cpu().numpy())
        elif kv.startswith('logit_noncat_slot_start'):
            global_vars['logit_noncat_slot_start'].append(v[0].cpu().numpy())
        elif kv.startswith('logit_noncat_slot_end'):
            global_vars['logit_noncat_slot_end'].append(v[0].cpu().numpy())
        elif kv.startswith('requested_slot_status'):
            global_vars['requested_slot_status'].append(v[0].cpu().numpy())
        elif kv.startswith('intent_status'):
            intent_status = v[0]
        elif kv.startswith('num_slots'):
            global_vars['num_slots'].append(v[0].cpu().numpy())
        elif kv.startswith('categorical_slot_status'):
            global_vars['categorical_slot_status'].append(v[0].cpu().numpy())
        elif kv.startswith('num_categorical_slots'):
            global_vars['num_categorical_slots'].append(v[0].cpu().numpy())
        elif kv.startswith('categorical_slot_values'):
            global_vars['categorical_slot_values'].append(v[0].cpu().numpy())
        elif kv.startswith('noncategorical_slot_status'):
            global_vars['noncategorical_slot_status'].append(v[0].cpu().numpy())
        elif kv.startswith('num_noncategorical_slots'):
            global_vars['num_noncategorical_slots'].append(v[0].cpu().numpy())
        elif kv.startswith('noncategorical_slot_value_start'):
            global_vars['noncategorical_slot_value_start'].append(v[0].cpu().numpy())
        elif kv.startswith('noncategorical_slot_value_end'):
            global_vars['noncategorical_slot_value_end'].append(v[0].cpu().numpy())

    import pdb; pdb.set_trace()

    num_active_intents = torch.sum(intent_status, axis=1).unsqueeze(1)
    tensor_ones = torch.ones(num_active_intents.size(), dtype=torch.long)

    if num_active_intents.is_cuda:
        tensor_ones = tensor_ones.cuda()

    # adding label for NONE intent - 1 if no acive intent for the dialogue
    none_intent_label = tensor_ones - num_active_intents
    onehot_intent_labels = torch.cat([none_intent_label, intent_status], axis=1)
    _, intent_labels = onehot_intent_labels.max(dim=1)

    global_vars['logit_intent_status'].extend(torch.argmax(logit_intent_status, 1).cpu().numpy())
    global_vars['intent_status'].extend(intent_labels.cpu().numpy())


    # point_outputs_max = torch.argmax(point_outputs, dim=-1)
    # mask_paddings = (tgt_ids == data_desc.vocab.pad_id)
    # comp_res = ((point_outputs_max == tgt_ids) | mask_paddings)
    # comp_res = torch.all(comp_res, axis=-1, keepdims=False)

    # global_vars['comp_res'].extend(comp_res.cpu().numpy())
    # global_vars['gating_preds'].extend(torch.argmax(gate_outputs, axis=-1).cpu().numpy())


def eval_epochs_done_callback(global_vars, data_desc):
    joint_acc, turn_acc = \
        evaluate_metrics(global_vars['comp_res'],
                         global_vars['gating_labels'],
                         global_vars['gating_preds'],
                         data_desc.gating_dict["ptr"])

    gating_comp_flatten = (np.asarray(global_vars['gating_labels']) == np.asarray(global_vars['gating_preds'])).ravel()
    gating_acc = np.sum(gating_comp_flatten) / len(gating_comp_flatten)

    evaluation_metrics = {"Joint_Goal_Acc": joint_acc,
                          "Turn_Acc": turn_acc,
                          "Gate_Acc": gating_acc}
    print(evaluation_metrics)



    active_intent_acc = metrics.get_active_intent_accuracy(
            frame_ref, frame_hyp)
    slot_tagging_f1_scores = metrics.get_slot_tagging_f1(
        frame_ref, frame_hyp, turn_ref["utterance"], service)
    requested_slots_f1_scores = metrics.get_requested_slots_f1(
        frame_ref, frame_hyp)
    goal_accuracy_dict = metrics.get_average_and_joint_goal_accuracy(
        frame_ref, frame_hyp, service)

    return evaluation_metrics


def evaluate_metrics(comp_res, gating_labels, gating_preds, ptr_code):
    # TODO: Calculate precision, recall, and F1
    total_slots = 0
    correct_slots = 0
    total_turns = 0
    correct_turns = 0
    for result_idx, result in enumerate(comp_res):
        turn_wrong = False
        total_turns += 1
        for slot_idx, slot_eq in enumerate(result):
            total_slots += 1
            if gating_labels[result_idx][slot_idx] == ptr_code:
                if slot_eq:
                    correct_slots += 1
                else:
                    turn_wrong = True
            elif gating_labels[result_idx][slot_idx] == gating_preds[result_idx][slot_idx] \
                    or (slot_eq and gating_preds[result_idx][slot_idx] == ptr_code):
                correct_slots += 1
            else:
                turn_wrong = True
        if not turn_wrong:
            correct_turns += 1

    turn_acc = correct_slots / float(total_slots) if total_slots != 0 else 0
    joint_acc = correct_turns / float(total_turns) if total_turns != 0 else 0
    return joint_acc, turn_acc

def get_metrics(dataset_ref, dataset_hyp, service_schemas, in_domain_services):
  """Calculate the DSTC8 metrics.
  Args:
    dataset_ref: The ground truth dataset represented as a dict mapping dialogue
      id to the corresponding dialogue.
    dataset_hyp: The predictions in the same format as `dataset_ref`.
    service_schemas: A dict mapping service name to the schema for the service.
    in_domain_services: The set of services which are present in the training
      set.
  Returns:
    A dict mapping a metric collection name to a dict containing the values
    for various metrics. Each metric collection aggregates the metrics across
    a specific set of frames in the dialogues.
  """
  # Metrics can be aggregated in various ways, eg over all dialogues, only for
  # dialogues containing unseen services or for dialogues corresponding to a
  # single service. This aggregation is done through metric_collections, which
  # is a dict mapping a collection name to a dict, which maps a metric to a list
  # of values for that metric. Each value in this list is the value taken by
  # the metric on a frame.
  metric_collections = collections.defaultdict(
      lambda: collections.defaultdict(list))

  # Ensure the dialogs in dataset_hyp also occur in dataset_ref.
  assert set(dataset_hyp.keys()).issubset(set(dataset_ref.keys()))
  logger.logging.info("len(dataset_hyp)=%d, len(dataset_ref)=%d", len(dataset_hyp),
                  len(dataset_ref))

  # Store metrics for every frame for debugging.
  per_frame_metric = {}
  for dial_id, dial_hyp in dataset_hyp.items():
    dial_ref = dataset_ref[dial_id]

    if set(dial_ref["services"]) != set(dial_hyp["services"]):
      raise ValueError(
          "Set of services present in ground truth and predictions don't match "
          "for dialogue with id {}".format(dial_id))

    for turn_id, (turn_ref, turn_hyp) in enumerate(
        zip(dial_ref["turns"], dial_hyp["turns"])):
      if turn_ref["speaker"] != turn_hyp["speaker"]:
        raise ValueError(
            "Speakers don't match in dialogue with id {}".format(dial_id))

      # Skip system turns because metrics are only computed for user turns.
      if turn_ref["speaker"] != "USER":
        continue

      if turn_ref["utterance"] != turn_hyp["utterance"]:
        tf.logging.info("Ref utt: %s", turn_ref["utterance"])
        tf.logging.info("Hyp utt: %s", turn_hyp["utterance"])
        raise ValueError(
            "Utterances don't match for dialogue with id {}".format(dial_id))

      hyp_frames_by_service = {
          frame["service"]: frame for frame in turn_hyp["frames"]
      }

      # Calculate metrics for each frame in each user turn.
      for frame_ref in turn_ref["frames"]:
        service_name = frame_ref["service"]
        if service_name not in hyp_frames_by_service:
          raise ValueError(
              "Frame for service {} not found in dialogue with id {}".format(
                  service_name, dial_id))
        service = service_schemas[service_name]
        frame_hyp = hyp_frames_by_service[service_name]

        active_intent_acc = metrics.get_active_intent_accuracy(
            frame_ref, frame_hyp)
        slot_tagging_f1_scores = metrics.get_slot_tagging_f1(
            frame_ref, frame_hyp, turn_ref["utterance"], service)
        requested_slots_f1_scores = metrics.get_requested_slots_f1(
            frame_ref, frame_hyp)
        goal_accuracy_dict = metrics.get_average_and_joint_goal_accuracy(
            frame_ref, frame_hyp, service)

        frame_metric = {
            metrics.ACTIVE_INTENT_ACCURACY:
                active_intent_acc,
            metrics.REQUESTED_SLOTS_F1:
                requested_slots_f1_scores.f1,
            metrics.REQUESTED_SLOTS_PRECISION:
                requested_slots_f1_scores.precision,
            metrics.REQUESTED_SLOTS_RECALL:
                requested_slots_f1_scores.recall
        }
        if slot_tagging_f1_scores is not None:
          frame_metric[metrics.SLOT_TAGGING_F1] = slot_tagging_f1_scores.f1
          frame_metric[metrics.SLOT_TAGGING_PRECISION] = (
              slot_tagging_f1_scores.precision)
          frame_metric[
              metrics.SLOT_TAGGING_RECALL] = slot_tagging_f1_scores.recall
        frame_metric.update(goal_accuracy_dict)

        frame_id = "{:s}-{:03d}-{:s}".format(dial_id, turn_id,
                                             frame_hyp["service"])
        per_frame_metric[frame_id] = frame_metric
        # Add the frame-level metric result back to dialogues.
        frame_hyp["metrics"] = frame_metric

        # Get the domain name of the service.
        domain_name = frame_hyp["service"].split("_")[0]
        domain_keys = [ALL_SERVICES, frame_hyp["service"], domain_name]
        if frame_hyp["service"] in in_domain_services:
          domain_keys.append(SEEN_SERVICES)
        else:
          domain_keys.append(UNSEEN_SERVICES)
        for domain_key in domain_keys:
          for metric_key, metric_value in frame_metric.items():
            if metric_value != metrics.NAN_VAL:
              metric_collections[domain_key][metric_key].append(metric_value)

  all_metric_aggregate = {}
  for domain_key, domain_metric_vals in metric_collections.items():
    domain_metric_aggregate = {}
    for metric_key, value_list in domain_metric_vals.items():
      if value_list:
        # Metrics are macro-averaged across all frames.
        domain_metric_aggregate[metric_key] = float(np.mean(value_list))
      else:
        domain_metric_aggregate[metric_key] = metrics.NAN_VAL
    all_metric_aggregate[domain_key] = domain_metric_aggregate
  return all_metric_aggregate, per_frame_metric