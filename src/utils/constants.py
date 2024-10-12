import copy

WINDOW_SIZE = 2

datasets = [
    {
        "raw": "../data/raw/Afghanistan-2000-2024.csv",
        "processed": "../data/processed/Afghanistan-2000-2024",
        "name": "Afghanistan",
        "criteria": "Rice (high quality)",
    },
    {
        "raw": "../data/raw/Argentina-2005-2022.csv",
        "processed": "../data/processed/Argentina-2005-2022",
        "name": "Argentina",
        "criteria": None,
    },
    {
        "raw": "../data/raw/Bangladesh-1998-2024.csv",
        "processed": "../data/processed/Bangladesh-1998-2024",
        "name": "Bangladesh",
        "criteria": None,
    },
    {
        "raw": "../data/raw/India-1994-2024.csv",
        "processed": "../data/processed/India-1994-2024",
        "name": "India",
        "criteria": None,
    },
    {
        "raw": "../data/raw/Indonesia-2007-2023.csv",
        "processed": "../data/processed/Indonesia-2007-2023",
        "name": "Indonesia",
        "criteria": None,
    },
    {
        "raw": "../data/raw/Iran-2012-2022.csv",
        "processed": "../data/processed/Iran-2012-2022",
        "name": "Iran",
        "criteria": None,
    },
    {
        "raw": "../data/raw/Japan-2011-2020.csv",
        "processed": "../data/processed/Japan-2011-2020",
        "name": "Japan",
        "criteria": None,
    },
    {
        "raw": "../data/raw/Kyrgyzstan-2005-2020.csv",
        "processed": "../data/processed/Kyrgyzstan-2005-2020",
        "name": "Kyrgyzstan",
        "criteria": "Rice",
    },
    {
        "raw": "../data/raw/Lao-2020-2024.csv",
        "processed": "../data/processed/Lao-2020-2024",
        "name": "Lao",
        "criteria": "Rice (ordinary, first quality)",
    },
    {
        "raw": "../data/raw/Lebanon-2012-2024.csv",
        "processed": "../data/processed/Lebanon-2012-2024",
        "name": "Lebanon",
        "criteria": None,
    },
    {
        "raw": "../data/raw/Moldova-2008-2024.csv",
        "processed": "../data/processed/Moldova-2008-2024",
        "name": "Moldova",
        "criteria": None,
    },
    {
        "raw": "../data/raw/Nepal-2001-2024.csv",
        "processed": "../data/processed/Nepal-2001-2024",
        "name": "Nepal",
        "criteria": None,
    },
    {
        "raw": "../data/raw/Nigeria-2002-2024.csv",
        "processed": "../data/processed/Nigeria-2002-2024",
        "name": "Nigeria",
        "criteria": None,
    },
    {
        "raw": "../data/raw/Pakistan-2004-2024.csv",
        "processed": "../data/processed/Pakistan-2004-2024",
        "name": "Pakistan",
        "criteria": None,
    },
    {
        "raw": "../data/raw/Palestine-2007-2024.csv",
        "processed": "../data/processed/Palestine-2007-2024",
        "name": "Palestine",
        "criteria": "Rice (short grain, low quality, local)",
    },
    {
        "raw": "../data/raw/Philipines-2000-2024.csv",
        "processed": "../data/processed/Philipines-2000-2024",
        "name": "Philipines",
        "criteria": None,
    },
    {
        "raw": "../data/raw/Senegal-2000-2023.csv",
        "processed": "../data/processed/Senegal-2000-2023",
        "name": "Senegal",
        "criteria": None,
    },
    {
        "raw": "../data/raw/SriLanka-2004-2024.csv",
        "processed": "../data/processed/SriLanka-2004-2024",
        "name": "SriLanka",
        "criteria": "Rice (long grain)",
    },
    {
        "raw": "../data/raw/Tajikistan-2002-2024.csv",
        "processed": "../data/processed/Tajikistan-2002-2024",
        "name": "Tajikistan",
        "criteria": None,
    },
    {
        "raw": "../data/raw/Turkey-2013-2022.csv",
        "processed": "../data/processed/Turkey-2013-2022",
        "name": "Turkey",
        "criteria": None,
    },
    {
        "raw": "../data/raw/Ukraine-2014-2023.csv",
        "processed": "../data/processed/Ukraine-2014-2023",
        "name": "Ukraine",
        "criteria": None,
    },
]


def get_countries(countries, commodity):
    country_list = []

    for country in countries:
        country_list.append(
            next(
                (dataset for dataset in datasets if dataset["name"] == country),
                None,
            ).copy()
        )

    for country in country_list:
        country["processed"] = country["processed"] + "-" + commodity + ".csv"

    return country_list


def get_country(country, commodity):
    data = next(
        (dataset for dataset in datasets if dataset["name"] == country), None
    ).copy()

    if data is not None:
        data["processed"] = data["processed"] + "-" + commodity + ".csv"

    return data


def get_scaler_filename(country, commodity):
    return f"../scalers/{country}-{commodity}-StandardScaler.pkl"


def get_model_filename(country, commodity, final=False, market=None):
    if final and market is None:
        return f"../models/{country}-{commodity}-final-model.h5"
    elif market is not None and final is False:
        return f"../models/Market-{market}-best-model.h5"
    elif market is not None and final is True:
        return f"../models/Market-{market}-final-model.h5"
    return f"../models/{country}-{commodity}-best-model.h5"


def get_tl_model_filename(base, country, commodity):
    return f"../models/{base}-{country}-{commodity}.h5"


def get_large_model_results():
    return "../reports/large-models.json"


def get_small_model_results():
    return "../reports/small-models.json"


def get_tl_model_results():
    return "../reports/tl-models.json"


def get_market_data(name):
    return "../data/processed/Market-" + name + ".csv"


def get_market_results():
    return "../reports/market-models.json"


def get_market_tl_results():
    return "../reports/tl-market-models.json"
