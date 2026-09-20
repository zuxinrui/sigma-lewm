# Historical module aliases

These aliases contain no model or loss implementations. New code uses `trim.*`.
For an old serialized artifact requiring one of these module paths, set:

```bash
export PYTHONPATH="$PWD:$PWD/compat:$PYTHONPATH"
```

Do not change checkpoint contents to work around an import error. The most common
model aliases remain at repository root for existing Hydra configurations.
