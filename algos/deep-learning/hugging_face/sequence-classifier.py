from datasets import DatasetDict, Dataset, load_from_disk

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)

from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
import evaluate
import torch
import numpy as np


accuracy = evaluate.load("accuracy")


def tokenize_function(examples):
    # Extract text
    text = examples["text"]

    # Tokenize and truncate text
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=512
    )

    return tokenized_inputs


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)

    return {
        "accuracy": accuracy.compute(
            predictions=predictions,
            references=labels
        )
    }



if __name__ == "__main__":
    # Import base model
    model_checkpoint = "distilbert-base-uncased"

    # Define label maps
    id2label = {0: "Negative", 1: "Positive"}
    label2id = {"Negative":0, "Positive":1}

    # Generate classification model from model_checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=2, id2label=id2label, label2id=label2id
    )
    print(model)

    # Load dataset
    dataset = load_from_disk("data/recipe-classification-dataset")
    print(dataset)

    count = 0
    for example in dataset["validation"]:
        print(example)

        if count >= 28:
            break

        print()
        count += 1

    # Create tokenizer

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

    # Add pad token if none exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    # Tokenize training and validation datasets
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    print(tokenized_dataset)

    count = 0
    for example in tokenized_dataset["train"]:
        print(example)

        if count >= 1:
            break

        print()
        count += 1

    # Create data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Apply untrained model to text
    # Define list of examples
    text_list = [
        "Day,Recipe, \nmon,pepper pasta\nsat,panini paffuti\nsun,lentil salad \n,\nIngredient,Quantity\npenne,1.00 lb\nsauce,30.00 oz\nprosciutto,8.00 oz\nspinach,8.00 oz\nlentils,8.00 oz\nrocket,5.00 oz\npepper,3 items\nonion,1 items\nbun,4 items\ntomato,2 items\nprovolone,4 items\naubergine,1 items\n",
        "Day,Recipe, \ntue,pizza\nwed,classic pasta\nthu,fish and rice\nfri,panini paffuti\n,\nIngredient,Quantity\npizza,1.00 lb\npotato_wedges,8.00 oz\npenne,1.00 lb\nsauce,30.00 oz\ntuna_fresh,12.00 oz\nrice,6.00 oz\npea,4.00 oz\nolive_oil,4.00 tbs\nprosciutto,8.00 oz\nspinach,8.00 oz\naubergine,1 items\ncourgette,2 items\nonion,1 items\ncarrot,2 items\nlemon,1 items\nbun,4 items\ntomato,2 items\nprovolone,4 items\n",
        "Day,Recipe, \nmon,pasta con fagioli \nsun,penne amore \n,\nIngredient,Quantity\nrigatoni,1.00 lb\nsauce,30.00 oz\nkidney_bean,14.00 oz\npenne,1.00 lb\nham,8.00 oz\ncream,8.00 oz\npea,8.00 oz\nonion,1 items\npepper,1 items\nchives,3 items\n",
        "Day,Recipe, \nmon,risotto ai funghi\ntue,pepper pasta\nwed,cous cous\nthu,tuna pasta \nfri,panini paffuti\n,\nIngredient,Quantity\nrice,8.00 oz\nmushroom,8.00 oz\nbutter,8.00 oz\npenne,1.00 lb\nsauce,30.00 oz\ncouscous,1.00 lb\ncumin,1.00 tbs\nturmeric,1.00 tbs\nolive_oil,4.00 tbs\nolive,28.00 oz\ntuna_tinned,10.00 oz\nbasil,3.00 oz\nlinguine,1.00 lb\nprosciutto,8.00 oz\nspinach,8.00 oz\nonion,3 items\nsausage,4 items\npepper,4 items\ntomato,11 items\ncourgette,2 items\naubergine,1 items\nbun,4 items\nprovolone,4 items\n'",
    ]

    print("Untrained model predictions:")
    print("----------------------------")
    for text in text_list:
        # Tokenize text
        inputs = tokenizer.encode(text, return_tensors="pt")
        # Compute logits
        logits = model(inputs).logits
        # Convert logits to label
        predictions = torch.argmax(logits)

        print(text[0:50].replace("\n", "") + " - " + id2label[predictions.tolist()])

    # Fine tuning model with LoRa
    peft_config = LoraConfig(
        task_type="SEQ_CLS",  # Sequence classification
        r=4,  # Intrinsic rank of trainable weight matrix
        lora_alpha=32,  # This is like a learning rate
        lora_dropout=0.01,  # Probablity of dropout
        target_modules=['q_lin']  # We apply lora to query layer only
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Hyperparameters
    lr = 1e-3  # Size of optimization step
    batch_size = 5  # Number of examples processed per optimziation step
    num_epochs = 20  # Number of times model runs through training data

    training_args = TrainingArguments(
        output_dir=model_checkpoint + "-recipe-lora-text-classification",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,  # Our peft model
        args=training_args,  # Hyperparameters
        train_dataset=tokenized_dataset["train"],  # Training data
        eval_dataset=tokenized_dataset["validation"],  # Validation data
        tokenizer=tokenizer,  # Define tokenizer
        data_collator=data_collator,  # This will dynamically pad examples in each batch to be equal length
        compute_metrics=compute_metrics,  # Evaluates model using compute_metrics() function from before
    )

    # Train model
    trainer.train()

    # Predict using trained model
    model.to("cpu")  # Moving to cpu
    print("Trained model predictions:")
    print("--------------------------")
    for text in text_list:
        inputs = tokenizer.encode(text, return_tensors="pt").to("cpu")  # Moving to cpu

        logits = model(inputs).logits
        predictions = torch.max(logits, 1).indices

        print(text[:50].replace("\n", "") + " - " + id2label[predictions.tolist()[0]])
