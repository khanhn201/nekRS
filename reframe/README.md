# Reframe testing

## Aurora
```
module load reframe
```

Submit a reframe job that runs all CI modes in ethier with the same cache

```
reframe -C config/aurora_conf.py -c ethierBatch.py -v -r
```

## Local
```
reframe -C config/local_conf.py -c ethier.py --exec-policy=serial -v -r
```

List previous results

```
reframe --list-stored-testcases=now-1d:now/mean:/ -C config/local_conf.py 
```
