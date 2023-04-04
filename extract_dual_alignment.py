#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import torch,os

from fairseq import bleu, checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter


def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'

    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()
        model.eval()

    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    if args.print_vanilla_alignment:
        import string
        punc = string.punctuation
        src_punc_tokens = [w for w in range(len(src_dict)) if src_dict[w] in punc]
    else:
        src_punc_tokens = None

    import time
    print('start time is :',time.strftime("%Y-%m-%d %X"))
    # import pdb;pdb.set_trace()
    with progress_bar.build_progress_bar(args, itr) as t:
        if args.decoding_path is not None:
            align_sents = [[] for _ in range(4000000)]
            f_align_sents = [[] for _ in range(4000000)]
            b_align_sents = [[] for _ in range(4000000)]

        for sample in t:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if 'net_input' not in sample:
                continue

            if args.print_vanilla_alignment:
                net_output = models[0](sample)
                alignments = models[0].extract_merge_then_align_alignment(sample, net_output)
                f_alignments = models[0].extract_align_then_merge_alignment(sample, net_output, src_punc_tokens)
                b_alignments = models[0].extract_align_then_merge_alignment(sample, net_output, src_punc_tokens, reverse=True)
            else:
                alignments, f_alignments, b_alignments = None,None,None
            
            for i, sample_id in enumerate(sample['id'].tolist()):
                if args.print_vanilla_alignment and args.decoding_path is not None:
                    align_sents[int(sample_id)].append(alignments[str(sample_id)])
                    f_align_sents[int(sample_id)].append(f_alignments[sample_id])
                    b_align_sents[int(sample_id)].append(b_alignments[sample_id])
  
    print('end time is :',time.strftime("%Y-%m-%d %X"))       
    if args.decoding_path is not None and args.print_vanilla_alignment:
        with open(os.path.join(args.decoding_path, f'{args.gen_subset}.{args.source_lang}2{args.target_lang}.bidual.align'), 'w') as f:
            for sents in align_sents:
                if len(sents)==0:
                    continue                  
                for sent in sents:
                    f.write(str(sent)+'\n')

        with open(os.path.join(args.decoding_path, f'{args.gen_subset}.{args.source_lang}2{args.target_lang}.dualf.align'), 'w') as f:
            for sents in f_align_sents:
                if len(sents)==0:
                    continue                  
                for sent in sents:
                    f.write(str(sent)+'\n')

        with open(os.path.join(args.decoding_path, f'{args.gen_subset}.{args.source_lang}2{args.target_lang}.dualb.align'), 'w') as f:
            for sents in b_align_sents:
                if len(sents)==0:
                    continue                  
                for sent in sents:
                    f.write(str(sent)+'\n')
        print("finished ...")



def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)

if __name__ == '__main__':
    cli_main()
