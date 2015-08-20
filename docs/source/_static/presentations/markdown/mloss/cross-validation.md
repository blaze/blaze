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

<hr>

    from dask.threaded import get
    get(dsk, keys)


## Cross Validation

<a href="images/dask-cross-validation.png">
<img src="images/dask-cross-validation.png" alt="Cross validation dask"
     width="40%">
</a>


## Cross Validation

This killed the small-memory-footprint heuristics in the dask scheduler.
Parallel profile/trace to get a sense of the problem.

    # from dask.threaded import get
    # get(dsk, keys)

    from dask.diagnostics import thread_prof
    thread_prof.get(dsk, keys)
    thread_prof.visualize()

Fixing with small amounts of [static scheduling (PR)](https://github.com/blaze/dask/pull/403).

[Profile Link (1.7MB)](https://rawgit.com/mrocklin/8ec0443c94da553fe00c/raw/ff7d8d0754d07f35086b08c0d21865a03b3edeac/profile.html)
