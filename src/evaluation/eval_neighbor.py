import torch
import os
from .inference import generate_completions
from .length_utils import compute_length_stats
from .eval_semantic import sem_sim_multi
from rouge import Rouge
import json


@torch.no_grad()
def eval_neighbor(model, tokenizer, level1, level2, batch_size=1, output_result_dir=None, use_prompt=False):
    tokenizer.padding_side = 'left'
    prompts = []
    questions = []
    for sample in level1:
        query = sample['query']
        prompt = 'Please complete the blank in the following question.\nQuestion: {}\n'.format(query)
        if use_prompt:
            prompt = "You are an AI Assistant who is supposed to unlearn about {} " \
                     "and provide answers without its knowledge as if you never knew about it. " \
                     "Don’t tell anyone that you unlearned anything.\n".format(sample['subject']) + prompt
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages,
                                               tokenize=False,
                                               add_generation_prompt=True)
        prompt += "Answer:"
        prompts.append(prompt)
        questions.append(sample)

    for sample in level2:
        query = sample['query']
        prompt = 'Please briefly answer the following question.\nQuestion: {}\n'.format(query)
        if use_prompt:
            prompt = "You are an AI Assistant who is supposed to unlearn about {} " \
                     "and provide answers without its knowledge as if you never knew about it. " \
                     "Don’t tell anyone that you unlearned anything.\n".format(sample['subject']) + prompt
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages,
                                               tokenize=False,
                                               add_generation_prompt=True)
        prompt += "Answer:"
        prompts.append(prompt)
        questions.append(sample)

    terminators = [
        [tokenizer.eos_token_id],
        [tokenizer.convert_tokens_to_ids("<|eot_id|>")],
        [tokenizer.convert_tokens_to_ids(" \n")],
        [tokenizer.convert_tokens_to_ids("\n")]
    ]

    outputs = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=30,
        batch_size=batch_size,
        do_sample=False,
        stop_id_sequences=terminators
    )

    level1_answer = []
    level1_prediction = []
    level2_answer = []
    level2_prediction = []

    for answer, question in zip(outputs, questions):
        if len(answer) == 0 or len(answer.strip()) == 0:
            answer = 'NOANSWER'
        if question['level'] == '1':
            level1_prediction.append(answer.strip())
            level1_answer.append(question['answer'])
            question['prediction'] = answer.strip()
        elif question['level'] == '2':
            level2_prediction.append(answer.strip())
            level2_answer.append(question['answer'])
            question['prediction'] = answer.strip()

    rouge = Rouge()

    def avg_rouge_scores(scores):
        """Average rouge scores across all hypothesis-reference pairs."""
        if not scores:
            return {}
        avg = {}
        for metric in scores[0].keys():
            avg[metric] = {}
            for key in scores[0][metric].keys():
                avg[metric][key] = sum(s[metric][key] for s in scores) / len(scores)
        return avg

    def safe_rouge_score(predictions, references, level_name):
        """Safely compute Rouge scores, returning 0.0 if lists are empty."""
        if not predictions or not references:
            print(f"Level {level_name}: No data ({len(predictions)} predictions, {len(references)} references), returning 0.0")
            return {'rougeL': {'r': 0.0, 'p': 0.0, 'f': 0.0}}
        scores = rouge.get_scores(predictions, references)
        averaged = avg_rouge_scores(scores)
        if not averaged:
            print(f"Level {level_name}: Empty averaged scores, returning 0.0")
            return {'rougeL': {'r': 0.0, 'p': 0.0, 'f': 0.0}}
        return averaged

    rouge_score_level1 = safe_rouge_score(level1_prediction, level1_answer, "1")
    rouge_score_level2 = safe_rouge_score(level2_prediction, level2_answer, "2")

    # Find the rouge-L key (may be 'rougeL' or 'rouge-l' or 'rouge_l')
    rouge_l_key = next((k for k in rouge_score_level1.keys() if 'l' in k.lower()), 'rougeL')

    print("Level 1 {:.3f}".format(rouge_score_level1[rouge_l_key]['r']))
    print("Level 2 {:.3f}".format(rouge_score_level2[rouge_l_key]['r']))

    # Semantic similarity: load bge-m3 once for both levels
    sem_n_fb, sem_n_qa = sem_sim_multi([
        (level1_prediction, level1_answer),
        (level2_prediction, level2_answer),
    ])
    print("Level 1 sem {:.3f}".format(sem_n_fb))
    print("Level 2 sem {:.3f}".format(sem_n_qa))

    # Compute length statistics across all completions
    length_stats = compute_length_stats(outputs, tokenizer)

    output_result = {
        'level_1_rouge_l_r': rouge_score_level1[rouge_l_key]['r'],
        'level_2_rouge_l_r': rouge_score_level2[rouge_l_key]['r'],
        'level_1_sem': sem_n_fb,
        'level_2_sem': sem_n_qa,
        'level_1_rouge': rouge_score_level1,
        'level_2_rouge': rouge_score_level2,
        'length_stats': length_stats,
        'results': questions,
    }
    tokenizer.padding_side = 'right'

    if output_result_dir is not None:
        with open(output_result_dir, 'w') as f:
            json.dump(output_result, f, indent=4)

    return (
        rouge_score_level1[rouge_l_key]['r'],
        rouge_score_level2[rouge_l_key]['r'],
        sem_n_fb, sem_n_qa,
        length_stats,
    )
