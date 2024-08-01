def get_countries(commodity, country=None):
    argentina = { 'raw': '../data/raw/Argentina-2005-2022.csv', 'processed': '../data/processed/Argentina-2005-2022', 'name': 'Argentina' }
    bangladesh = { 'raw': '../data/raw/Bangladesh-1998-2024.csv', 'processed': '../data/processed/Bangladesh-1998-2024', 'name': 'Bangladesh' }
    india = { 'raw': '../data/raw/India-1994-2024.csv', 'processed': '../data/processed/India-1994-2024', 'name': 'India' }
    indonesia = { 'raw': '../data/raw/Indonesia-2007-2023.csv', 'processed': '../data/processed/Indonesia-2007-2023', 'name': 'Indonesia' }
    iran = { 'raw': '../data/raw/Iran-2012-2022.csv', 'processed': '../data/processed/Iran-2012-2022', 'name': 'Iran' }
    moldova = { 'raw': '../data/raw/Moldova-2008-2024.csv', 'processed': '../data/processed/Moldova-2008-2024', 'name': 'Moldova' }
    nepal = { 'raw': '../data/raw/Nepal-2001-2024.csv', 'processed': '../data/processed/Nepal-2001-2024', 'name': 'Nepal' }
    nigeria = { 'raw': '../data/raw/Nigeria-2002-2024.csv', 'processed': '../data/processed/Nigeria-2002-2024', 'name': 'Nigeria' }
    pakistan = { 'raw': '../data/raw/Pakistan-2004-2024.csv', 'processed': '../data/processed/Pakistan-2004-2024', 'name': 'Pakistan' }
    philipines = { 'raw': '../data/raw/Philipines-2000-2024.csv', 'processed': '../data/processed/Philipines-2000-2024', 'name': 'Philipines' }
    senegal = { 'raw': '../data/raw/Senegal-2000-2023.csv', 'processed': '../data/processed/Senegal-2000-2023', 'name': 'Senegal' }
    turkey = { 'raw': '../data/raw/Turkey-2013-2022.csv', 'processed': '../data/processed/Turkey-2013-2022', 'name': 'Turkey' }
    ukraine = { 'raw': '../data/raw/Ukraine-2014-2023.csv', 'processed': '../data/processed/Ukraine-2014-2023', 'name': 'Ukraine' }

    datasets = [argentina, bangladesh, india, indonesia, iran, moldova, nepal, nigeria, pakistan, philipines, senegal, turkey, ukraine]
    for dataset in datasets:
        dataset['processed'] = dataset['processed'] + '-' + commodity + '.csv'
    
    if country is not None:
        return next((dataset for dataset in datasets if dataset['name'] == country), None)
    
    return datasets

def get_scaler_filename(country, commodity):
    return f'../scalers/{country}-{commodity}-StandardScaler.pkl'

def get_model_filename(country, commodity):
    return f'../models/{country}-{commodity}-best-model.h5'

def get_tl_model_filename(base, country, commodity):
    return f'../models/{base}-{country}-{commodity}-transfer-learning.h5'

