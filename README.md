## Selective Knowledge negation Unlearning (SKU)

### Instructions
We conduct our experiments on OPT2.7b, llama2-7b and llama2-13b model. 

### GPU requirements
Since task vector would require state_dict deduction and addition between two models, we conduct
our OPT model experiments on 3 A100 GPUs (80GiB), and llama model experiments on 4 A100 GPUs (80GiB).


### Dataset
We use the TruthfulQA dataset (avilable on github, one can simply use ```wget``` command to pull the data) as the normal data. 
And we use PKU-SafeRLHF (avilable on Huggaingface) as  the harmful data.

### Unlearn
We can perform the unlearning by run the following command. Here we show an example of how to unlearn harmfulness learned from the pretrained OPT2.7B. 

#### Before Unlearn
After downloading the data, create a directory called ```dataloader``` to store the data. Then, create a directory named ```logs``` in
harmful_unlearn directory to store the log files.

#### Unlearn process
```bash
python unlearn_harm_new.py --bad_weight 2.5 --random_weight 2.5 --normal_weight 1 
--lr 2e-4 --max_unlearn_steps 1000 --model_name=facebook/opt-2.7b 
--model_save_dir=YOUR_MODEL_SAVING_PATH  
--batch_size 2 --log_file=logs/opt-2.7b-unlearn_new.log 
--task_vector_saving_path=YOUR_TASK_VECTOR_SAVING_PATH
```

In this command, ```bad_weight```, ```random_weight``` and ```normal_weight``` are the weights to adjust each module.
```lr``` is the learning rate. 
```max_unlearn_steps``` is the maximum number of unlearning steps. 
```model_name``` is the targeted unlearned model name. 
```model_save_dir``` is the path to save the unlearned model. 
```batch_size``` is the batch size. 
```log_file``` is the path to save the log file. 
```task_vector_saving_path``` is the path to save the task vector.


### Simple test
To test the efficient of the unlearned model (or some other model), you can simply run the following command:

```bash
python main.py
```
In main_test.py, you can change the model name and saved model path to test different models.

