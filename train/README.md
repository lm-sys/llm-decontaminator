# Train

This document can help reproduce the llama-rephraser; you can also create new rephrase samples to study other benchmarks.

## Contents

- [Tokenize](#tokenize)
- [Fine-tune](#fine-tune)
- [Evaluation](#evaluation)


## Tokenize
The `rephrase_tokenize.py` can convert a jsonl file into tok, idx, and msk files, which can be used to finetune the model.

~~~bash
python3 rephrase_tokenize.py --model /path/to/model/weight --in /path/to/rephrase.jsonl --max-seq-len 1536
~~~

Tokenize HumanEval:

~~~bash
python3 rephrase_tokenize.py --model /path/to/llama/weight --in data/rephrase/humaneval_python.jsonl --max-seq-len 1536
~~~


## Fine-tune

Once you have the `.tok` files, you can use them to fine-tune the model. Please make modifications in `finetune.sh`.

~~~bash
bash finetune.sh
~~~

Here are some key points: 
- Ensure you set the correct `model_name_or_path`, `data_path`, and `output_dir`. Set the `data_path` to the tok file you wish to fine-tune. 
- Fine-tuning requires GPU resources; we recommend fine-tuning on 2 or more A100 or H100 GPUs. The `nproc_per_node` should be the number of GPUs you have.
- Note that `per_device_train_batch_size` * `max_steps` * `nproc_per_node` = `sample_num` * `epoch`. The `per_device_train_batch_size` depends on your GPU memory, thus the `epoch` determines `max_steps`. In the paper, achieving a full score on the test set requires more than 50 epochs. On the rephrased MMLU, 16 epochs can achieve very good results. On GSM-8k and HumanEval, we opt for 32-64 epochs.

## Evaluation

We use [instruct-eval](https://github.com/declare-lab/instruct-eval) framework to evaluate MMLU and GSM-8k. This repo works well on MMLU, but there are some [issues](https://github.com/declare-lab/instruct-eval/blob/720e66f627369266ed1cfd74426666ec37e524bc/lm_eval/base.py#L329) with GSM-8k. You may refer to [my solution](https://github.com/andy-yang-1/instruct-eval/pull/1) for a faster evaluation. For HumanEval, we use the [repo](https://github.com/openai/human-eval) provided by OpenAI.

If you use declare-lab's implementation, change [this function](https://github.com/declare-lab/instruct-eval/blob/720e66f627369266ed1cfd74426666ec37e524bc/lm_eval/base.py#L329) with the following code and unset the [load_8bit](https://github.com/declare-lab/instruct-eval/blob/720e66f627369266ed1cfd74426666ec37e524bc/lm_eval/models/llama.py#L16).

~~~py
    def greedy_until(self, requests):
        # TODO: implement fully general `until` that handles until that are
        #       multiple tokens or that span multiple tokens correctly

        # TODO: extract to TokenizedLM?
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return len(toks), x[0]

        re_ord = utils.Reorderer(requests, _collate)

        for context, until in tqdm(re_ord.get_reordered()):
            if isinstance(until, str):
                until = [until]

            primary_until = self.tok_encode(until[0])

            context_enc = torch.tensor([self.tok_encode(context)])

            cont = self._model_generate(
                context_enc, context_enc.shape[1] + self.max_gen_toks, 2
            )

            s = self.tok_decode(cont[0].tolist()[context_enc.shape[1] :-1])

            res.append(s)

        return re_ord.get_original(res)
~~~