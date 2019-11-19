#!/bin/bash
 
while getopts ":i::f::n::d::o:::" opt; do
  case $opt in
    f)
      echo "file is ! $OPTARG"
      FILE=$OPTARG
      ;;
    n)
      echo "number is ! $OPTARG" 
      NUM=$OPTARG
      ;;
    d)
      echo "div is ! $OPTARG" 
      DIV=$OPTARG
      ;;
    m)
      echo "media output file is ! $OPTARG" 
      MEDIA=$OPTARG
      ;;
    o)
      echo "output is ! $OPTARG" 
      OUT=$OPTARG
      ;;
    i)
      echo "input dir is ! $OPTARG" 
      DIR=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" 
      exit 1
      ;;
  esac
done

FILE=$DIR/$FILE
FILE_SHUF=$FILE\_SHUF
LEN=$(wc -l $FILE | awk '{print $1}')
SINGLE=$(expr $LEN / $DIV)
TRAIN=$DIR/train
TEST=$DIR/test
echo $FILE,$NUM,$DIV,$LEN,$SINGLE,$OUT,$FILE_SHUF,$TRAIN,$TEST,$MEDIA
rm -f $OUT
for I in $(seq 1 $NUM)
do
  shuf $FILE > $FILE_SHUF
  for J in $(seq 1 $DIV)
  do
    S=$(expr $(expr $J - 1) \* $SINGLE)
    E=$(expr $J \* $SINGLE)
    if [ $J -eq $DIV ]; then
      E=$LEN 
    fi
    echo $I,$J,$S,$E
    echo $FILE_SHUF
    gawk -v s=$S -v e=$E 'NR>s && NR<=e {print}' $FILE_SHUF > $TEST
    gawk -v s=$S -v e=$E 'NR<=s || NR>e {print}' $FILE_SHUF > $TRAIN
    wc -l $TEST $TRAIN
    # sh ./train.sh
    # sh ./test.sh
    cat $MEDIA >> $OUT
    rm -f $TEST $TRAIN 
  done
done 