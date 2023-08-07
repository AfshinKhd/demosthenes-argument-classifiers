import yaml


def load_conf(path) -> dict:
    try:
        with open(path, "r") as f:
            yaml_script = f.read()

        conf = yaml.safe_load(yaml_script)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
    return conf




    


    