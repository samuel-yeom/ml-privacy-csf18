for depth in {1..30}; do
  echo "depth = $depth"
  time python main.py iwpc tree $depth
done

for lambda in 5.0 10.0 20.0 50.0 100.0 200.0 500.0 1000.0 2000.0 5000.0; do
  echo "lambda = $lambda"
  time python main.py eyedata linreg $lambda
done