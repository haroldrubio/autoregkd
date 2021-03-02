import json
from pathlib import Path
from transformers.models.bart import BartTokenizerFast

def load_squad(path, debug=False):
    tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')
    # Read in raw data
    contexts, questions, answers = hf_read_squad(path, debug)
    # Add in the end index
    hf_add_end_idx(answers, contexts)
    # Tokenize the questions and contexts
    encodings = tokenizer(contexts, questions, truncation=True, padding=True)
    # Convert answer indicies to token indicies
    hf_add_token_positions(encodings, answers, tokenizer)
    # Return the adjusted encodings
    return encodings

# Functions prefixed with 'hf' are source code from the huggingface website

def hf_read_squad(path, debug=False):
    path = Path(path)
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []

    example_counter = 0
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            if debug and example_counter >= 10:
                break
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
            example_counter += 1

    return contexts, questions, answers

def hf_add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        # sometimes squad answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        elif context[start_idx-1:end_idx-1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
        elif context[start_idx-2:end_idx-2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters

def hf_add_token_positions(encodings, answers, tokenizer):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length

        # if end position is None, the 'char_to_token' function points to the space before the correct token - > add + 1
        if end_positions[-1] is None:
            end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] + 1)
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})


