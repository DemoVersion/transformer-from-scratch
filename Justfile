apply_isort:
    uv run isort .

apply_ruff:
    uv run ruff format .
    uv run ruff check --fix .

clean_code: apply_ruff apply_isort

check_code:
    uv run pyright . --level error
