from unsloth import FastLanguageModel  # isort: skip - must be imported before trl otherwise the code breaks

import json
import uuid
from enum import StrEnum
from pathlib import Path

import yaml
from datasets import Dataset, DatasetDict, concatenate_datasets
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from trl import SFTConfig, SFTTrainer, apply_chat_template

from clinical_assistant.summarization.allergies import ALLERGIES_EXTRACTION_PROMPT_TEMPLATE
from clinical_assistant.summarization.clinical_problems import CLINICAL_PROBLEM_HISTORY_PROMPT_TEMPLATE
from clinical_assistant.summarization.family_history import FAMILY_HISTORY_EXTRACTION_PROMPT_TEMPLATE
from clinical_assistant.summarization.measurements import MEASUREMENT_EXTRACTION_PROMPT_TEMPLATE
from clinical_assistant.summarization.medications import MEDICATION_EVENT_PROMPT_TEMPLATE
from clinical_assistant.summarization.procedures import PROCEDURE_EXTRACTION_PROMPT_TEMPLATE
from clinical_assistant.summarization.substances import SUBSTANCE_USE_EXTRACTION_PROMPT_TEMPLATE


class Config(BaseModel):
    model: str
    dataset_path: Path
    max_seq_length: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    lr_scheduler_type: str
    rank: int
    seed: int
    eval_strategy: str
    save_strategy: str
    bf16: bool
    packing: bool
    logging_strategy: str
    logging_steps: int
    rslora: bool
    lora_dropout: float
    warmup_ratio: float
    num_train_epochs: int


class Topic(StrEnum):
    ALLERGIES = "allergies"
    SUBSTANCES = "substances"
    FAMILY_HISTORY = "family_history"
    CLINICAL_PROBLEMS = "clinical_problems"
    PROCEDURES = "procedures"
    MEDICATIONS = "medications"
    MEASUREMENTS = "measurements"


PROMPT_TEMPLATE = {
    Topic.ALLERGIES: ALLERGIES_EXTRACTION_PROMPT_TEMPLATE,
    Topic.SUBSTANCES: SUBSTANCE_USE_EXTRACTION_PROMPT_TEMPLATE,
    Topic.FAMILY_HISTORY: FAMILY_HISTORY_EXTRACTION_PROMPT_TEMPLATE,
    Topic.CLINICAL_PROBLEMS: CLINICAL_PROBLEM_HISTORY_PROMPT_TEMPLATE,
    Topic.PROCEDURES: PROCEDURE_EXTRACTION_PROMPT_TEMPLATE,
    Topic.MEDICATIONS: MEDICATION_EVENT_PROMPT_TEMPLATE,
    Topic.MEASUREMENTS: MEASUREMENT_EXTRACTION_PROMPT_TEMPLATE,
}


class SummarizationDatasetLoader:
    def __init__(self, dataset_path: Path, tokenizer, max_seq_length: int, seed: int):
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.seed = seed

    def preprocess(self, example):
        prompt_template = PROMPT_TEMPLATE[example["topic"]]
        prompt = prompt_template.format(**example["input"])

        sft_example = {
            "prompt": [
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "completion": [
                {
                    "role": "assistant",
                    "content": json.dumps(example["output"]),
                }
            ],
        }
        sft_example = apply_chat_template(sft_example, self.tokenizer)

        prompt_ids = self.tokenizer(sft_example["prompt"])["input_ids"]
        prompt_completion_ids = self.tokenizer(sft_example["prompt"] + sft_example["completion"])["input_ids"]

        return {
            "input_ids": prompt_completion_ids,
            "completion_mask": [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids)),
        }

    def load_topic_dataset(self, topic: Topic) -> DatasetDict:
        dataset = []
        with self.dataset_path.open() as f:
            for line in f:
                example = json.loads(line)
                topic_example = {
                    "topic": topic,
                    "input": {topic: example["summary"][topic], "document": example["document"]},
                    "output": example["summary_delta"][topic],
                }
                if topic in [Topic.PROCEDURES]:
                    _ = topic_example["input"].pop(topic)
                dataset.append(topic_example)

        train_dataset, val_dataset = train_test_split(dataset, test_size=3, random_state=self.seed)
        train_dataset = Dataset.from_list(train_dataset)
        val_dataset = Dataset.from_list(val_dataset)

        dataset = DatasetDict({"train": train_dataset, "val": val_dataset})
        return dataset

    def load_dataset(self) -> DatasetDict:
        train_datasets = []
        val_datasets = []

        for topic in Topic:
            ds = self.load_topic_dataset(topic)
            train_datasets.append(ds["train"])
            val_datasets.append(ds["val"])

        dataset = DatasetDict(
            {
                "train": concatenate_datasets(train_datasets),
                "val": concatenate_datasets(val_datasets),
            }
        )

        dataset["train"] = dataset["train"].map(self.preprocess)
        dataset["val"] = dataset["val"].map(self.preprocess)

        dataset["train"] = dataset["train"].filter(lambda x: len(x["input_ids"]) <= self.max_seq_length)
        dataset["val"] = dataset["val"].filter(lambda x: len(x["input_ids"]) <= self.max_seq_length)

        return dataset


def main():
    with open("./config/summarization/finetune.yaml") as f:
        cfg_dict = yaml.safe_load(f)
        cfg = Config(**cfg_dict)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.rank,
        lora_alpha=2 * cfg.rank,
        lora_dropout=cfg.lora_dropout,
        random_state=cfg.seed,
        use_rslora=cfg.rslora,
    )

    run_id = uuid.uuid4().hex

    dataset_loader = SummarizationDatasetLoader(cfg.dataset_path, tokenizer, cfg.max_seq_length, cfg.seed)
    dataset = dataset_loader.load_dataset()

    output_dir = Path(f"./outputs/summarization/finetuning/{run_id}")
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        args=SFTConfig(
            output_dir=output_dir,
            num_train_epochs=cfg.num_train_epochs,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            per_device_eval_batch_size=cfg.per_device_eval_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            lr_scheduler_type=cfg.lr_scheduler_type,
            warmup_ratio=cfg.warmup_ratio,
            logging_strategy=cfg.logging_strategy,
            logging_steps=cfg.logging_steps,
            eval_strategy=cfg.eval_strategy,
            save_strategy=cfg.save_strategy,
            bf16=cfg.bf16,
            max_seq_length=cfg.max_seq_length,
            packing=cfg.packing,
        ),
    )

    trainer.train()
    model.save_pretrained_merged(output_dir / "finetuned_merged", tokenizer, save_method="merged_16bit")


if __name__ == "__main__":
    main()
