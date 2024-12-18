import random

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import DataCollatorForLanguageModeling
import json
from torch.utils.data import DataLoader, random_split, TensorDataset

torch.manual_seed(8888)
np.random.seed(8888)
random.seed(8888)

def create_pku_dataloader_from_dataset(tokenizer, dataset, fraction=1.0, batch_size=64):
    """
    Given the PKU dataset, create the dataloader on the unlearned harmful Q&A pairs.

    Args:
        tokenizer: Tokenizer.
        dataset: Loaded PKU dataset.
        batch_size: Batch size.

    Returns:
        Data loader of PKU harmful Q&A pairs.
    """

    # Preproccess function.
    def preproccess(examples):
        """
        Input: Dict[List]
        Output: Dict[List]
        """
        results = {"input_ids": [], "attention_mask": [], "start_locs": []}

        for i in range(len(examples["prompt"])):
            # Subsample if needed.
            if random.random() > fraction:
                continue

            prompt = examples["prompt"][i]
            response_list = []

            # Add only bad samples.
            if not examples["is_response_0_safe"][i]:
                response_list.append(examples["response_0"][i])
            if not examples["is_response_1_safe"][i]:
                response_list.append(examples["response_1"][i])

            # Add all responses to results or skip if none.
            for response in response_list:
                text = f"### Question: {prompt}\n ### Answer: {response}"
                tokenized = tokenizer(text, truncation=True, padding="max_length")
                results["input_ids"].append(tokenized["input_ids"])
                results["attention_mask"].append(tokenized["attention_mask"])
                # Calculate start idx for answer
                test_text = f"### Question: {prompt}\n ### Answer: "
                test_tokenized = tokenizer(
                    test_text, truncation=True, padding="max_length"
                )
                results["start_locs"].append(len(test_tokenized["input_ids"]) - 1)

        return results

    # Need to drop all original columns to emit more than one row for each original row https://huggingface.co/docs/datasets/about_map_batch#input-size-output-size.
    dataset = dataset.map(
        preproccess,
        batched=True,
        remove_columns=[
            "prompt",
            "response_0",
            "response_1",
            "is_response_0_safe",
            "is_response_1_safe",
            "better_response_id",
            "safer_response_id",
        ],
    )
    dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "start_locs"]
    )

    # Add labels and make it data loader.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=data_collator
    )

    return dataloader

def create_truthfulqa_dataloader_augmented(tokenizer, batch_size=64):
    """
    Create the TruthfulQA dataloader for the normal data and include both "Best Answer" and "Correct Answer".

    Args:
        tokenizer: Tokenizer.
        batch_size: Batch size.

    Returns:
        Data loader of TruthfulQA Q&A pairs.
    """
    df = pd.read_csv("../dataloader/TruthfulQA.csv")
    questions, good_answers, correct_answers = df["Question"].values, df["Best Answer"].values, df["Correct Answers"].values

    data = {"input_ids": [], "attention_mask": []}

    for question, answer in zip(questions, good_answers):
        text = f"### Question: {question}\n ### Answer: {answer}"
        tokenized = tokenizer(text, truncation=True, padding="max_length")
        data["input_ids"].append(tokenized["input_ids"])
        data["attention_mask"].append(tokenized["attention_mask"])

    for question, answer in zip(questions, correct_answers):
        text = f"### Question: {question}\n ### Answer: {answer}"
        tokenized = tokenizer(text, truncation=True, padding="max_length")
        data["input_ids"].append(tokenized["input_ids"])
        data["attention_mask"].append(tokenized["attention_mask"])


    dataset = Dataset.from_dict(data)

    train_len = int(0.7* len(dataset))
    val_len = int(0.1 * len(dataset))
    test_len = len(dataset) - train_len - val_len

    train_data, val_data, test_data = torch.utils.data.random_split(
        dataset, [train_len, val_len, test_len]
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, collate_fn=data_collator, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, collate_fn=data_collator, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, collate_fn=data_collator, shuffle=True
    )

    return train_dataloader, val_dataloader, test_dataloader


def flag_harmful_responses(batch):
    """
    Flag harmful responses in a batch of data.

    Args:
        batch: A batch of data from the dataset.

    Returns:
        Dictionary with flags indicating harmful responses.
    """
    is_harmful_0 = [not safe for safe in batch["is_response_0_safe"]]
    is_harmful_1 = [not safe for safe in batch["is_response_1_safe"]]
    return {"is_harmful_0": is_harmful_0, "is_harmful_1": is_harmful_1}

def get_harmful_responses(dataset_split):
    """
    Extract harmful responses from a dataset split.

    Args:
        dataset_split: A split of the dataset (train or test).

    Returns:
        List of all harmful responses in the dataset split.
    """
    # Apply the function to flag harmful responses
    flagged_dataset = dataset_split.map(
        flag_harmful_responses, batched=True
    )

    # Filter and extract harmful responses
    harmful_responses_0 = flagged_dataset.filter(lambda x: x["is_harmful_0"])["response_0"]
    harmful_responses_1 = flagged_dataset.filter(lambda x: x["is_harmful_1"])["response_1"]

    return harmful_responses_0 + harmful_responses_1

def compute_reverse_kl(pretrained_model, current_model, batch, device):
    """
    Compute *backward* KL as the normal utility loss.

    Args:
        pretrained_model: reference model which is the pretrained (original) model.
        current_model: The current unlearning model.
        batch: A batch of normal data.
        device: GPU device.

    Returns:
       The KL loss.
    """

    normal_outputs = current_model(
        batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device)
    )

    with torch.no_grad():
        pretrained_outputs = pretrained_model(
            batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device)
        )

    # Q: current model; P: pretrained model.
    prob_q = torch.nn.functional.softmax(normal_outputs.logits, dim=-1)
    prob_p = torch.nn.functional.softmax(pretrained_outputs.logits, dim=-1)

    # Negative KL divergence: sum(Q * log(Q/P))
    # loss = (prob_q * torch.log(prob_q / (prob_p + 1e-12))).sum(-1).mean()
    loss = - (prob_p * torch.log((prob_p + 1e-12) / prob_q)).sum(-1).mean()

    return loss

def get_answer_loss(operation, batch, model, device="cuda"):
    """
    Compute the loss on the answer (i.e. y) part.

    Args:
        operation: either "ga" (gradient ascent) or "gd" (gradient descent).
        batch: A batch of data.
        model: The unlearned model.
        device: GPU device.

    Returns:
       The loss.
    """
    assert operation in ["ga", "gd"], "Operation must be either GA or GD."
    input_ids, attention_mask, start_locs, labels = (
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["start_locs"],
        batch["labels"].to(device),
    )
    outputs = model(input_ids, attention_mask=attention_mask)

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    # Shift one to predict next token.
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    losses = []
    for bid in range(input_ids.shape[0]):
        one_inp, one_st = input_ids[bid], start_locs[bid]

        # GA or GD.
        position_loss = loss_fct(shift_logits[bid], shift_labels[bid])

        if operation == "ga":  # Negative the direction for GA.
            position_loss = -position_loss

        # Simply put equal weights on all answers.
        position_weight = torch.zeros_like(one_inp)
        assert len(position_weight) == len(position_loss) + 1
        position_weight[one_st:] = 1  # only focus on answer part

        # Ignore the padding part.
        position_weight[one_inp == 1] = 0
        if position_weight.sum() > 0:
            position_weight = position_weight / position_weight.sum()

        one_loss = (position_weight[:-1] * position_loss).sum()
        losses.append(one_loss)

    final_loss = torch.stack(losses).mean()

    return final_loss


def get_rand_ans_loss(bad_batch, tokenizer, normal_ans, model, K=5, device="cuda:0"):
    """
    Compute the loss of the random mismatch.

    Args:
        bad_batch: A batch of forgetting data.
        tokenizer: The tokenizer.
        normal_ans: A list of random answers.
        model: unlearned model.
        K: How many random answers sampled for each forgetting sample.
        device: GPU device.

    Returns:
       The random mismatch loss.
    """
    bad_input_ids = bad_batch["input_ids"].to(device)
    rand_ans_list = random.sample(normal_ans, k=K)
    batch_random_features = []
    for batch_idx in range(bad_input_ids.shape[0]):
        single_input_id = bad_input_ids[batch_idx, :]
        ori_text = tokenizer.decode(single_input_id)
        # Get question.
        question = ori_text.split("###")[1].split("Question:")[-1].strip()
        question_prefix = f"### Question: {question}\n ### Answer: "
        tokenized_question_prefix = tokenizer(
            question_prefix, truncation=True, padding="max_length"
        )
        # Doesn't need to minus 1 because there's a starting token in the beginning.
        start_loc = len(tokenized_question_prefix)

        # Get random answer.
        for rand_ans in rand_ans_list:
            random_sample = f"{question_prefix}{rand_ans}"

            # Tokenize.
            tokenized_rs = tokenizer(
                random_sample, truncation=True, padding="max_length"
            )
            batch_random_features.append(
                {
                    "input_ids": tokenized_rs["input_ids"],
                    "attention_mask": tokenized_rs["attention_mask"],
                    "start_locs": start_loc,
                }
            )

    # Batchify.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    batch_random = data_collator(batch_random_features)

    # GD on answer.
    random_loss = get_answer_loss("gd", batch_random, model, device=device)

    return random_loss
