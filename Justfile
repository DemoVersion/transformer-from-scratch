apply_ruff:
    uv run ruff format .
    uv run ruff check --fix .

clean_code: apply_ruff

check_code:
    uv run basedpyright . --level error
