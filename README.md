Deploy conda env:

conda env create --name <env name> --file=environment.yml (run from the directory)

Deploy server:

uvicorn main:app --reload (run from the directory)