
"""
A script to show an example of how to unlearn harmfulness.

The dataset used in is `PKU-SafeRLHF`. Model support OPT-2.7B, and Llama 2 (7B, 13B).
"""
import argparse
import sys

sys.path.append(('../'))
sys.path.append(('../../'))
import logging
import random
import time
import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import AdaLoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, LoraConfig
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, BitsAndBytesConfig
from task_vector import TaskVector

from utils import (
    create_pku_dataloader_from_dataset,
    get_answer_loss,
    get_rand_ans_loss,
    compute_reverse_kl, get_harmful_responses, create_truthfulqa_dataloader_augmented
)


torch.manual_seed(8888)
np.random.seed(8888)
random.seed(8888)


def main(args) -> None:
    # 16-bit quantization inference
    LORA = False
    access_token = "ENTER YOUR HUGGING FACE ACCESS TOKEN HERE"
    bnb_config = BitsAndBytesConfig(
        load_in_16bit=True,
        bnb_16bit_quant_type="nf16",
        bnb_16bit_compute_dtype=torch.float16,
        bnb_16bit_use_double_quant=True,
    )
    accelerator = Accelerator()

    if "Llama-2-7b" in args.model_name:
        LORA = True
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", quantization_config=bnb_config, device_map = "auto", token=access_token)
        tokenizer.pad_token = "[PAD]"

        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf",
                                                     quantization_config=bnb_config,
                                                     device_map='auto',
                                                     token=access_token)

        pretrained_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf",
                                                                quantization_config=bnb_config,
                                                                device_map='auto',
                                                                token=access_token
                                                                )

        model = prepare_model_for_kbit_training(model)

        config = LoraConfig(
            lora_alpha=16,
            inference_mode=False,
            r=32,
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    elif "Llama-2-13b" in args.model_name:
        LORA = True
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf",
                                                  token=access_token)

        tokenizer.pad_token = "[PAD]"

        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf",
                                                     quantization_config=bnb_config,
                                                     device_map='auto',
                                                     token=access_token)

        pretrained_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf",
                                                                quantization_config=bnb_config,
                                                                device_map='auto',
                                                                token=access_token
                                                                )

        model = prepare_model_for_kbit_training(model)

        config = LoraConfig(
            lora_alpha=8,
            inference_mode=False,
            r=16,
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    elif "opt-2.7b" in args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                     quantization_config=bnb_config,
                                                     device_map="auto")


        pretrained_model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                                quantization_config=bnb_config,
                                                                device_map="auto"
                                                                )

        ## If you need to use LORA for OPT model, you can uncomment the following lines.

        # LORA = True
        # model = prepare_model_for_kbit_training(model)
        #
        # config = LoraConfig(
        #     lora_alpha=16,
        #     inference_mode=False,
        #     r=32,
        #     bias="none",
        #     target_modules=["q_proj", "v_proj"],
        #     task_type="CAUSAL_LM",
        # )
        #
        # model = get_peft_model(model, config)
        # model.print_trainable_parameters()



    # Load harmful data.
    train_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")
    train_bad_loader = create_pku_dataloader_from_dataset(
        tokenizer, train_dataset, batch_size=args.batch_size
    )

    train_normal_loader, _, _ = create_truthfulqa_dataloader_augmented(
        tokenizer, batch_size=args.batch_size
    )

    torch.save(train_normal_loader, "../dataloader/train_normal_loader.pt")
    torch.save(train_bad_loader, "../dataloader/train_bad_loader.pt")
    print("data loader saved!")

    # Load normal answer used for random mismatch.
    bad_ans = get_harmful_responses(train_dataset)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Prepare.
    num_training_steps = args.max_unlearn_steps
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    (
        model,
        optimizer,
        train_bad_loader,
        train_normal_loader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_bad_loader, train_normal_loader, lr_scheduler
    )
    model.train()

    idx = 0
    while idx < args.max_unlearn_steps:
        for bad_batch, normal_batch in zip(train_bad_loader, train_normal_loader):

            ############ Guided Distortion Module ############
            bad_loss = get_answer_loss("gd", bad_batch, model, device="cuda")

            ############ Random Disassociation Module. ############
            random_loss = get_rand_ans_loss(
                bad_batch,
                tokenizer,
                bad_ans,
                model,
                K=5,
                device="cuda",
            )

            ############ Preservation Divergence Module ############
            normal_loss = compute_reverse_kl(pretrained_model, model, normal_batch, "cuda")

            loss = (
                args.bad_weight * bad_loss
                + args.random_weight * random_loss
                + args.normal_weight * normal_loss
            )

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Print.
            stats = (
                f"batch: {idx}, "
                f"GD_loss: {bad_loss:.2f}, "
                f"RD_loss: {random_loss:.2f}, "
                f"reversed_kl_loss: {normal_loss:.2f}, "
                f"combined_loss: {loss:.2f}, "
            )
            logging.info(stats)
            print(stats)
            idx += 1

    if LORA:
        model = model.merge_and_unload()

    print("saving model")
    model.save_pretrained(args.model_save_dir, from_pt=True)
    logging.info("Unlearning finished")

    # Save task vector.
    logging.info("Loading task vector")
    task_vector = TaskVector(pretrained_model, model)

    neg_task_vector = -task_vector

    # Apply the task vector
    new_benign_model = neg_task_vector.apply_to(pretrained_model)

    new_benign_model.save_pretrained(args.task_vector_saving_path, from_pt=True)

    print("Done saving task vector files!")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--max_unlearn_steps",
        type=int,
        default=1000,
        help="Max number of unlearning steps.",
    )
    parser.add_argument(
        "--bad_weight", type=float, default=0.5, help="Weight to adjust guided distortion module. e1"
    )
    parser.add_argument(
        "--random_weight",
        type=float,
        default=1,
        help="Weight to adjust random disassociation module. e2",
    )
    parser.add_argument(
        "--normal_weight",
        type=float,
        default=1,
        help="Weight to adjust preservation divergence. e3",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size of unlearning."
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="Unlearning LR.")

    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/opt-1.3b",
        help="Name of the pretrained model.",
    )
    parser.add_argument(
        "--model_save_dir",
        type=str,
        default="models/opt1.3b_unlearned",
        help="Directory to save model.",
    )
    parser.add_argument(
        "--saved_model", type=str, default=None, help="Saved pretrained model."
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="logs/meta-llama2-7b-unlearn.log",
        help="Log file name",
    )
    parser.add_argument(
        "--task_vector_saving_path",
        type=str,
        default="",
        help="Path to save task vector.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        filename=args.log_file,
        filemode="w+",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d-%H-%M",
        level=logging.INFO,
    )
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")
    main(args)
