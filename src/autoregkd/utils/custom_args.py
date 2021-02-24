# This class is for being able to pass additional arguments to the trainer
import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from transformers import TrainingArguments

@dataclass
class CustomArguments(TrainingArguments):
    test_name: str = field(
        default="No World",
        metadata={
            "help": (
                "A simple test"
            )
        },
    )