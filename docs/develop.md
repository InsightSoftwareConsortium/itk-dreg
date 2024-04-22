# Developer Notes

This section includes discussion and helpful tips for community members seeking
to contribute to the `itk_dreg` ecosystem either directly or via extension modules.

## Debugging

The following tools and approaches may be useful for debugging during development:

- Local testing: When developing new registration or reduction techniques it is often
useful to first develop against low-resolution data on one PC before moving to a
distributed cluster environment.

Dask can be configured to run workers in sequence for easier debugging and for
relaxed resource contraints:

```
dask.config.set(scheduler='single-threaded')
```

The `breakpoint()` function or `pdb` Python module can be useful to set breakpoints and examine state in a local
Python environment.

```
... # your code
breakpoint() # halts execution for interactive debugging
...
```

- Remote debugging: Dask [debugging documentation](https://docs.dask.org/en/latest/how-to/debug.html)
discusses approaches for parallel and remote debugging when running
on a cluster, including printing with `dask.distributed.print` and exception
handling.

`dask.distributed` supports the Python `logging` library as the preferred method for
collecting log information from workers. Logging behavior can be configured
in the Dask `Client` instance. See Dask
[logging documentation](https://distributed.dask.org/en/latest/logging.html)
for more information on event-based logging.

- We strongly encourage developers to write unit and integration tests during development
to help identify failure points and to maintain quality code practices. `itk_dreg` uses `pytest`
for its testing infrastructure.

- Use a `dask.distributed.LocalCluster` for debugging issues with graph serialization,
resource consumption, and worker behavior.
