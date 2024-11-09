import json
from src.generator import filter_excel_rows


def load_config():
    try:
        with open("config.json") as f:
            config = json.load(f)
            print("Using config.json parameters")
            return config
    except FileNotFoundError:
        try:
            # Fallback to default config
            with open("default_config.json") as f:
                config = json.load(f)
                print("Using default_config.json parameters")
                return config
        except FileNotFoundError:
            raise FileNotFoundError(
                "Neither config.json nor default_config.json found!"
            )


def main():
    config = load_config()
    result = filter_excel_rows("dataset.xlsx", config.get("tip_values"))

    if result is not None:
        print(f"Found {len(result)} rows")
        print(result)


if __name__ == "__main__":
    main()
