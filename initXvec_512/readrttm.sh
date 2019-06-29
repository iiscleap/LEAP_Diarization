f1=$(cat rttm)
file='rttm'
#echo $f1
f2='../lists/dihard2019DevList'
cat $f2 | while read i; do
        echo $i
        grep $i $file > ${i}.rttm
done

