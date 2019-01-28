cat ../results-sklearn/iwpc/tree-errors.txt | while read line; do
  set $line
  echo "depth = $1"
  time python main.py iwpc tree $1 --inc $2 $3 --one-error
  time python main.py iwpc tree $1 --inc $2 $3
done
python summarize.py ../results-sklearn/iwpc/membership/unknown-test-error tree
python summarize.py ../results-sklearn/iwpc/membership/known-test-error tree

cat ../results-sklearn/eyedata/linreg-errors.txt | while read line; do
  set $line
  echo "lambda = $1"
  time python main.py eyedata linreg $1 --inc $2 $3 --one-error
  time python main.py eyedata linreg $1 --inc $2 $3
done
python summarize.py ../results-sklearn/eyedata/membership/unknown-test-error linreg
python summarize.py ../results-sklearn/eyedata/membership/known-test-error linreg