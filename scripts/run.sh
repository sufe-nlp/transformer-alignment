#!/bin/bash

train_model(){   
    src=$1
    tgt=$2
    workdir=$3
    MaxUpdates=50000  
    modeldir=$workdir/models/${src}2${tgt}_vanilla && mkdir -p $modeldir
    
    echo "train vanilla transformer for ${src} to ${tgt}..."
    python train.py $workdir/processed_data \
        -a transformer_iwslt_de_en --optimizer adam --lr 0.0005 -s $src -t $tgt \
        --label-smoothing 0.1 --dropout 0.3 --max-tokens 4500  --seed 32  \
        --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 --activation-fn relu \
        --criterion label_smoothed_cross_entropy --max-update $MaxUpdates --clip-norm 0.0 \
        --warmup-updates 4000 --warmup-init-lr '1e-07' --keep-last-epochs 200 \
        --adam-betas '(0.9, 0.98)' --save-dir $modeldir --update-freq 8 --ddp-backend no_c10d \
        --share-all-embeddings --alignment-task 'vanilla' --save-interval 1  \
    2>&1 | tee $modeldir/train_log.out 
}

train_AET_transformer(){   
    src=$1
    tgt=$2
    workdir=$3
    MaxUpdates=10000  
    modeldir=$workdir/models/${src}2${tgt}_aet  && mkdir -p $modeldir

    ## use trained vanilla model to init AET model, frozen parameters of vanilla transformer and only train additional alignment module
    cp $workdir/models/${src}2${tgt}_vanilla/checkpoint_best.pt $modeldir/checkpoint_last_f.pt 
    cp $workdir/models/${tgt}2${src}_vanilla/checkpoint_best.pt $modeldir/checkpoint_last_b.pt 
    
    echo "train alignment enhanced transformer (AET) for ${src} to ${tgt}..."
    python train.py $workdir/processed_data \
        -a dual_transformer_small --optimizer adam --lr 0.0005 -s $src -t $tgt \
        --label-smoothing 0.1 --dropout 0.3 --max-tokens 4500  --seed 32  \
        --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 --activation-fn relu \
        --criterion label_smoothed_cross_entropy_dual_trans --max-update $MaxUpdates \
        --warmup-updates 4000 --warmup-init-lr '1e-07' --keep-last-epochs 100  \
        --adam-betas '(0.9, 0.98)' --save-dir $modeldir --update-freq 8 --ddp-backend no_c10d \
        --share-all-embeddings --save-interval 1 \
        --alignment-task 'supalign' --set-dual-trans  --load-dual-model --load-alignments \
    2>&1 | tee $modeldir/train_log.out
}

extract_bidirectional_alignment(){    
    src=$1
    tgt=$2
    workdir=$3
    resdir=$workdir/result/${src}2${tgt}_vanilla  && mkdir -p $resdir 

    echo "Extract alignment from vanilla transformer and merge using grow-diagonal."
    for split in 'train' 'valid'; do
        echo "start to extract ${src}2${tgt} alignment."
        Model=$workdir/models/${src}2${tgt}_vanilla
        python generate_align.py $workdir/processed_data -s $src -t $tgt --path $Model/checkpoint_best.pt \
            --max-tokens 8000 --beam 1 --remove-bpe --quiet --print-vanilla-alignment --decoding-path $resdir  \
            --alignment-task vanilla --gen-subset $split --alignment-layer 2 --set-shift 
            
        echo "start to extract ${tgt}2${src} alignment."
        Model=$workdir/models/${tgt}2${src}_vanilla
        python generate_align.py $workdir/processed_data -s $tgt -t $src --path $Model/checkpoint_best.pt \
            --max-tokens 8000 --beam 1 --remove-bpe --quiet --print-vanilla-alignment --decoding-path $resdir  \
            --alignment-task vanilla --gen-subset $split --alignment-layer 2 --set-shift 

        method="grow-diagonal"
        python scripts/aer/combine_bidirectional_alignments.py $resdir/$split.${src}2${tgt}.align $resdir/$split.${tgt}2${src}.align --method $method > $resdir/$split.${src}2${tgt}.bidir.align
    done 
    echo "successfully extract $src-$tgt bidirectional alignment. Now you need to re-processe your data with generated alignment."
}


calculate_AET_att_AER(){   
    src=$1
    tgt=$2
    workdir=$3
    bpe=$workdir/bpe  ## put the BPE files here
    resdir=$workdir/genres/${src}2${tgt}_aet  && mkdir -p $resdir 
    ref_align=$workdir/raw/test.talp  # define your reference alignment file here

    split='test'
    Model=$workdir/models/${src}2${tgt}_aet
    echo "start to extract alignment with AET method."
    python extract_dual_alignment.py $workdir/processed_data -s $src -t $tgt --path $Model/checkpoint_best.pt \
        --max-tokens 4000 --beam 1 --remove-bpe --left-pad-source True --left-pad-target False \
        --decoding-path $resdir --quiet --print-vanilla-alignment --alignment-task 'supalign' \
        --gen-subset $split --alignment-layer 2 --set-shift --set-dual-trans 
    
    python scripts/aer/sentencepiece_to_word_alignments.py --src $bpe/test.$src \
       --tgt $bpe/test.$tgt --input $resdir/$split.${src}2${tgt}.bidual.align --output $resdir/$split.${src}2${tgt}.bidual.raw.align
    python scripts/aer/sentencepiece_to_word_alignments.py --src $bpe/test.$src \
       --tgt $bpe/test.$tgt --input $resdir/$split.${src}2${tgt}.dualf.align --output $resdir/$split.${src}2${tgt}.dualf.raw.align 
    python scripts/aer/sentencepiece_to_word_alignments.py --src $bpe/test.$tgt \
       --tgt $bpe/test.$src --input $resdir/$split.${src}2${tgt}.dualb.align --output $resdir/$split.${src}2${tgt}.dualb.raw.align

    method="grow-diagonal"
    python scripts/aer/combine_bidirectional_alignments.py $resdir/$split.${src}2${tgt}.dualf.raw.align $resdir/$split.${src}2${tgt}.dualb.raw.align --method $method > $resdir/$split.${src}2${tgt}.bidir.aet.align

    echo "=====AER is start to calculate AER for align then merge. ..."   
    python scripts/aer/aer.py ${ref_align} $resdir/$split.${src}2${tgt}.dualf.raw.align --fAlpha 0.5 --oneRef 
    python scripts/aer/aer.py ${ref_align} $resdir/$split.${src}2${tgt}.dualb.raw.align --fAlpha 0.5 --oneRef --reverseHyp
    python scripts/aer/aer.py ${ref_align} $resdir/$split.${src}2${tgt}.bidir.aet.align --fAlpha 0.5 --oneRef 
}

calculate_shift_att_AER(){   
    src=$1
    tgt=$2
    workdir=$3
    bpe=$workdir/bpe  ## put the BPE files here
    resdir=$workdir/result/${src}2${tgt}_vanilla  && mkdir -p $resdir 
    ref_align=$workdir/raw/test.talp  # define your reference alignment file here

    split='test'
    echo "start to extract ${src}2${tgt} alignment."
    Model=$workdir/models/${src}2${tgt}_vanilla
    python generate_align.py $workdir/processed_data -s $src -t $tgt --path $Model/checkpoint_best.pt \
        --max-tokens 4000 --beam 1 --remove-bpe --quiet --print-vanilla-alignment --decoding-path $resdir  \
        --alignment-task vanilla --gen-subset $split --alignment-layer 2 --set-shift 
    
    python scripts/aer/sentencepiece_to_word_alignments.py --src $bpe/test.$src \
       --tgt $bpe/test.$tgt --input $resdir/$split.${src}2${tgt}.align --output $resdir/$split.${src}2${tgt}.raw.align

    echo "start to extract ${tgt}2${src} alignment." 
    Model=$workdir/models/${tgt}2${src}_vanilla
    python generate_align.py $workdir/processed_data -s $tgt -t $src --path $Model/checkpoint_best.pt \
        --max-tokens 4000 --beam 1 --remove-bpe --quiet --print-vanilla-alignment --decoding-path $resdir  \
        --alignment-task vanilla --gen-subset $split --alignment-layer 2 --set-shift 
    
    python scripts/aer/sentencepiece_to_word_alignments.py --src $bpe/test.$tgt \
       --tgt $bpe/test.$src --input $resdir/$split.${tgt}2${src}.align --output $resdir/$split.${tgt}2${src}.raw.align

    echo "start to merge alignments ..."
    method="grow-diagonal"
    python scripts/aer/combine_bidirectional_alignments.py $resdir/$split.${src}2${tgt}.raw.align  $resdir/$split.${tgt}2${src}.raw.align --method $method > $resdir/$split.${src}2${tgt}.bidir.align

    echo "=====calculate AER for shift-att method=========="   
    python scripts/aer/aer.py ${ref_align} $resdir/$split.${src}2${tgt}.raw.align --fAlpha 0.5 --oneRef 
    python scripts/aer/aer.py ${ref_align} $resdir/$split.${tgt}2${src}.raw.align --fAlpha 0.5 --oneRef --reverseHyp
    python scripts/aer/aer.py ${ref_align} $resdir/$split.${src}2${tgt}.bidir.align --fAlpha 0.5 --oneRef 
}

set -e 
cd ..
export CUDA_VISIBLE_DEVICES=1

src=de 
tgt=en 
workdir=/path/to/your/work/dir
# follow https://github.com/lilt/alignment-scripts to process training data
# processed data is in $workdir/processed_data dir.

echo "after processed training-data, train $src-$tgt and $tgt-$src vanilla transformer model."
train_model $src $tgt $workdir
train_model $tgt $src $workdir

# extract alignment and calculate AER using shift-att method.
calculate_shift_att_AER $src $tgt $workdir

echo "first extract shift-att alignment, process data with alignment, then train AET model"
extract_bidirectional_alignment $src $tgt $workdir
# process new alignment data using fairseq-process here.
train_AET_transformer $src $tgt $workdir

# extract alignment and calculate AER using AET method.
calculate_AET_att_AER $src $tgt $workdir