
# Solving crafter

```
python3 algorithms/ppo.py --track --wandb-project-name crafter --num-envs 16 --num-steps 2048  --num-minibatches 64 --update-epochs 10 --ent-coef 0
```

```
 python3 -u -m src.representations.main.curl_train curl
```

```
 python3 -u -m src.representations.main.curl_test curl
```

