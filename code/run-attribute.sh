if [ $# != 1 ]; then
  echo "This script requires exactly one argument as the target string"
  exit 1
fi

target=$1

cat ../results-sklearn/iwpc/tree-errors.txt | while read line; do
  set $line
  echo "depth = $1"
  time python main.py iwpc tree $1 --inv $2 $3 --target $target --one-error
done
python summarize.py ../results-sklearn/iwpc/attribute/$target/unknown-test-error tree