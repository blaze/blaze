## Example: Cross Validation

Afternoon sprint with Olivier Grisel

    for fold_id in range(n_folds):
        ...
        dsk[(name, 'model', model_id)] = clone(model)

        for partition_id in range(data.npartitions):
            if partition_id % n_folds == fold_id:
                dsk[(name, 'validation', validation_id)] = (score, ...)
            else:
                dsk[(name, 'model', model_id)] = (_partial_fit, ...)

            ...


## Cross Validation

<a href="../images/dask-cross-validation.pdf">
<img src="../images/dask-cross-validation.png" alt="Cross validation dask"
     width="40%">
</a>


## Cross Validation

This killed the small-memory-footprint heuristics in the dask scheduler.
Fixing with small amounts of
[static scheduling](https://github.com/ContinuumIO/dask/pull/403).

[Profile](https://rawgit.com/mrocklin/8ec0443c94da553fe00c/raw/ff7d8d0754d07f35086b08c0d21865a03b3edeac/profile.html)
