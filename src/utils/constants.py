WINDOW_SIZE = 2


def get_countries(commodity, country=None):
    afghanistan = {
        "raw": "../data/raw/Afghanistan-2000-2024.csv",
        "processed": "../data/processed/Afghanistan-2000-2024",
        "name": "Afghanistan",
        "criteria": "Rice (high quality)",
    }
    argentina = {
        "raw": "../data/raw/Argentina-2005-2022.csv",
        "processed": "../data/processed/Argentina-2005-2022",
        "name": "Argentina",
        "criteria": None,
    }
    bangladesh = {
        "raw": "../data/raw/Bangladesh-1998-2024.csv",
        "processed": "../data/processed/Bangladesh-1998-2024",
        "name": "Bangladesh",
        "criteria": None,
    }
    india = {
        "raw": "../data/raw/India-1994-2024.csv",
        "processed": "../data/processed/India-1994-2024",
        "name": "India",
        "criteria": None,
    }
    indonesia = {
        "raw": "../data/raw/Indonesia-2007-2023.csv",
        "processed": "../data/processed/Indonesia-2007-2023",
        "name": "Indonesia",
        "criteria": None,
    }
    iran = {
        "raw": "../data/raw/Iran-2012-2022.csv",
        "processed": "../data/processed/Iran-2012-2022",
        "name": "Iran",
        "criteria": None,
    }
    japan = {
        "raw": "../data/raw/Japan-2011-2020.csv",
        "processed": "../data/processed/Japan-2011-2020",
        "name": "Japan",
        "criteria": None,
    }
    kyrgyzstan = {
        "raw": "../data/raw/Kyrgyzstan-2005-2020.csv",
        "processed": "../data/processed/Kyrgyzstan-2005-2020",
        "name": "Kyrgyzstan",
        "criteria": "Rice",
    }
    moldova = {
        "raw": "../data/raw/Moldova-2008-2024.csv",
        "processed": "../data/processed/Moldova-2008-2024",
        "name": "Moldova",
        "criteria": None,
    }
    nepal = {
        "raw": "../data/raw/Nepal-2001-2024.csv",
        "processed": "../data/processed/Nepal-2001-2024",
        "name": "Nepal",
        "criteria": None,
    }
    nigeria = {
        "raw": "../data/raw/Nigeria-2002-2024.csv",
        "processed": "../data/processed/Nigeria-2002-2024",
        "name": "Nigeria",
        "criteria": None,
    }
    pakistan = {
        "raw": "../data/raw/Pakistan-2004-2024.csv",
        "processed": "../data/processed/Pakistan-2004-2024",
        "name": "Pakistan",
        "criteria": None,
    }
    philipines = {
        "raw": "../data/raw/Philipines-2000-2024.csv",
        "processed": "../data/processed/Philipines-2000-2024",
        "name": "Philipines",
        "criteria": None,
    }
    senegal = {
        "raw": "../data/raw/Senegal-2000-2023.csv",
        "processed": "../data/processed/Senegal-2000-2023",
        "name": "Senegal",
        "criteria": None,
    }
    srilanka = {
        "raw": "../data/raw/SriLanka-2004-2024.csv",
        "processed": "../data/processed/SriLanka-2004-2024",
        "name": "SriLanka",
        "criteria": "Rice (long grain)",
    }
    tajikistan = {
        "raw": "../data/raw/Tajikistan-2002-2024.csv",
        "processed": "../data/processed/Tajikistan-2002-2024",
        "name": "Tajikistan",
        "criteria": None,
    }
    turkey = {
        "raw": "../data/raw/Turkey-2013-2022.csv",
        "processed": "../data/processed/Turkey-2013-2022",
        "name": "Turkey",
        "criteria": None,
    }
    ukraine = {
        "raw": "../data/raw/Ukraine-2014-2023.csv",
        "processed": "../data/processed/Ukraine-2014-2023",
        "name": "Ukraine",
        "criteria": None,
    }

    datasets = [
        afghanistan,
        bangladesh,
        india,
        indonesia,
        kyrgyzstan,
        pakistan,
        tajikistan,
        srilanka,
    ]
    for dataset in datasets:
        dataset["processed"] = dataset["processed"] + "-" + commodity + ".csv"

    if country is not None:
        return next(
            (dataset for dataset in datasets if dataset["name"] == country), None
        )

    return datasets


def get_scaler_filename(country, commodity):
    return f"../scalers/{country}-{commodity}-StandardScaler.pkl"


def get_model_filename(country, commodity, final=False):
    if final:
        return f"../models/{country}-{commodity}-final-model.h5"
    return f"../models/{country}-{commodity}-best-model.h5"


def get_tl_model_filename(base, country, commodity):
    return f"../models/{base}-{country}-{commodity}.h5"


def get_large_model_results():
    return "../reports/large-models.json"


def get_small_model_results():
    return "../reports/small-models.json"


def get_tl_model_results():
    return "../reports/tl-models.json"
