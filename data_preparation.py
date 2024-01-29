from re import search
from typing import List

from pandas import DataFrame, get_dummies, read_csv, to_numeric


def data_preprocessing(input_filename: str, output_filename: str) -> None:
    data_frame = read_csv(input_filename, sep=";", low_memory=False)

    data_frame["area_total"] = normalize_area_total(data_frame["area_total"])

    data_frame["ceiling_height"] = normalize_ceiling_height(
        data_frame["ceiling_height"]
    )

    data_frame["komunal_cost"] = normalize_kommunal_cost(data_frame["komunal_cost"])

    data_frame["bathrooms_cnt"] = data_frame["bathrooms_cnt"].fillna(1.0)

    data_frame["sold_price"].fillna(data_frame["sold_price"].mean())

    data_frame["two_levels"] = data_frame["two_levels"].astype("category").cat.codes

    # data_frame["territory"] = vectorize_territory(data_frame["territory"])

    # drops
    data_frame = data_frame.drop(
        columns=[
            "id",
            "status",
            "date_sold",
            "area_live",
            "area_kitchen",
            "area_balcony",
            "loggia",
            "closed_yard",
            "territory",
        ],
        axis=1,
    )

    data_frame = data_frame.dropna()

    # category params convert
    data_frame = get_dummies(
        data_frame,
        columns=["balcon", "bathroom", "type", "windows", "keep", "plate"],
    )

    # writing the output
    data_frame.to_csv(output_filename, sep=";", index=False)


def normalize_area_total(df):
    return (
        to_numeric(df, errors="coerce")
        .fillna(0.0)
        .apply(lambda value: value if value >= 10 else 0.0)
    )


def normalize_ceiling_height(df):
    return to_numeric(df, errors="coerce").fillna(0.0)


def normalize_kommunal_cost(df):
    # cost preparing func
    def process_kommunal_cost(value):
        if isinstance(value, str):
            value = value.replace(" ", "").replace(",", ".")
            if value == "" or search("[а-яА-ЯёЁ_]+", value):
                return 0.0

        def float_check(values: List[str]) -> bool:
            try:
                [float(value) for value in values]
                return True
            except:
                return False

        sign = (
            "-"
            if "-" in str(value)
            else "/"
            if "/" in str(value)
            else "*"
            if "*" in str(value)
            else "="
            if "=" in str(value)
            else None
        )

        if sign and value:
            parts = value.split(sign)
            if float_check(parts):
                return (float(parts[0]) + float(parts[1])) / 2
            else:
                return 0.0
        elif str(value) != "nan":
            return float(value)
        else:
            return 0.0

    return df.apply(process_kommunal_cost)


if __name__ == "__main__":
    input_filename = "raw_data.csv"
    output_filename = "preprocessed_data.csv"
    data_preprocessing(input_filename, output_filename)
